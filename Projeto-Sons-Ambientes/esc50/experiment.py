from __future__ import annotations

import gc
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .config import ExperimentConfig
from .data import (
    ALL_FOLDS,
    assert_screening_balance,
    label_maps,
    limit_splits,
    load_esc50,
    make_splits,
    read_metadata,
    validate_metadata,
)
from .evaluation import (
    append_registry,
    classification_metrics,
    fit_temperature,
    save_json,
    save_prediction_artifacts,
    trainer_metrics,
)
from .models import (
    MODEL_SPECS,
    AudioCollator,
    build_model_bundle,
    load_model_bundle,
    parameter_count,
)


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_logits(predictions: Any) -> np.ndarray:
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return np.asarray(predictions)


def _mark_augmentation(splits: Any, enabled: bool) -> None:
    for name in splits:
        values = [bool(enabled and name == "train")] * len(splits[name])
        if "augment" in splits[name].column_names:
            splits[name] = splits[name].remove_columns("augment")
        splits[name] = splits[name].add_column("augment", values)


def _training_arguments(
    config: ExperimentConfig,
    has_validation: bool,
    train_size: int,
) -> TrainingArguments:
    spec = MODEL_SPECS[config.model_family]
    checkpoints = config.run_dir / "checkpoints"
    evaluation_strategy = "epoch" if has_validation else "no"
    batch_size = config.train_batch_size or spec.train_batch_size
    batches_per_epoch = int(np.ceil(train_size / batch_size))
    updates_per_epoch = int(np.ceil(batches_per_epoch / config.gradient_accumulation_steps))
    warmup_steps = int(updates_per_epoch * config.epochs * config.warmup_ratio)
    return TrainingArguments(
        output_dir=str(checkpoints),
        eval_strategy=evaluation_strategy,
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=config.learning_rate or spec.learning_rate,
        per_device_train_batch_size=config.train_batch_size or spec.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size or spec.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        warmup_steps=warmup_steps,
        logging_steps=10,
        load_best_model_at_end=has_validation,
        metric_for_best_model="f1_macro" if has_validation else None,
        greater_is_better=True if has_validation else None,
        seed=config.seed,
        data_seed=config.seed,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
    )


def _best_epoch(trainer: Trainer, fallback: int) -> int:
    history = [row for row in trainer.state.log_history if "eval_f1_macro" in row]
    if not history:
        return fallback
    return int(round(max(history, key=lambda row: row["eval_f1_macro"])["epoch"]))


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _roundtrip_check(
    family: str,
    trainer: Trainer,
    processor: Any,
    model_dir: Path,
    example: dict[str, Any],
) -> float:
    collator = AudioCollator(family, processor, augment=False)
    batch = collator([example])
    batch.pop("labels")
    device = next(trainer.model.parameters()).device
    inputs = {key: value.to(device) for key, value in batch.items()}
    trainer.model.eval()
    with torch.inference_mode():
        before = trainer.model(**inputs).logits.detach().cpu()

    loaded_family, loaded_model, loaded_processor = load_model_bundle(model_dir)
    if loaded_family != family:
        raise RuntimeError("Família mudou após salvar e recarregar.")
    loaded_model.to(device).eval()
    loaded_batch = AudioCollator(family, loaded_processor, augment=False)([example])
    loaded_batch.pop("labels")
    loaded_inputs = {key: value.to(device) for key, value in loaded_batch.items()}
    with torch.inference_mode():
        after = loaded_model(**loaded_inputs).logits.detach().cpu()
    difference = float(torch.max(torch.abs(before - after)))
    del loaded_model
    gc.collect()
    if difference > 1e-4:
        raise RuntimeError(f"Round-trip alterou logits (diferença máxima {difference:.6g}).")
    return difference


def _write_model_card(
    path: Path,
    config: ExperimentConfig,
    metrics: dict[str, Any],
    labels: list[str],
) -> None:
    validation = config.validation_fold if config.validation_fold is not None else "nenhum"
    test = config.test_fold if config.test_fold is not None else "nenhum"
    body = f"""# ESC-50 — {config.model_family}

Modelo de classificação fechada para as 50 categorias do ESC-50.

## Protocolo

- Folds de treino: {list(config.train_folds)}
- Fold de validação: {validation}
- Fold de teste: {test}
- Semente: {config.seed}
- Augmentação: {config.augment}

## Resultados

```json
{json.dumps(metrics, indent=2, ensure_ascii=False)}
```

## Limitações

O modelo reconhece apenas as 50 classes do ESC-50. Confiança alta não garante que um
áudio externo pertença a uma dessas classes. Um modelo treinado nos cinco folds serve
para inferência, mas não possui uma métrica de teste independente associada.

## Classes

{', '.join(labels)}
"""
    path.write_text(body, encoding="utf-8")


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    if config.model_family not in MODEL_SPECS:
        raise ValueError(f"Família desconhecida: {config.model_family}")
    existing_metrics = config.run_dir / "metrics.json"
    if existing_metrics.exists() and not config.force:
        return json.loads(existing_metrics.read_text(encoding="utf-8"))

    set_reproducibility(config.seed)
    metadata = read_metadata()
    validate_metadata(metadata)
    labels, _, _ = label_maps(metadata)
    dataset = load_esc50(config.dataset_name)
    splits = make_splits(
        dataset,
        config.train_folds,
        config.validation_fold,
        config.test_fold,
        config.seed,
    )
    if (
        tuple(config.train_folds) == (1, 2, 3)
        and config.validation_fold == 4
        and config.test_fold == 5
        and config.max_train_samples is None
        and config.max_eval_samples is None
    ):
        assert_screening_balance(splits)
    splits = limit_splits(splits, config.max_train_samples, config.max_eval_samples)
    _mark_augmentation(splits, config.augment)

    config.run_dir.mkdir(parents=True, exist_ok=True)
    run_config = config.to_dict()
    run_config["model_id"] = config.model_id or MODEL_SPECS[config.model_family].model_id
    run_config["labels"] = labels
    save_json(config.run_dir / "run_config.json", run_config)

    model, processor = build_model_bundle(
        config.model_family,
        labels,
        config.model_id,
        config.local_files_only,
    )
    total_parameters, trainable_parameters = parameter_count(model)
    collator = AudioCollator(config.model_family, processor, config.augment, config.seed)
    has_validation = "validation" in splits
    callbacks = []
    if has_validation and config.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(config.early_stopping_patience))
    trainer = Trainer(
        model=model,
        args=_training_arguments(config, has_validation, len(splits["train"])),
        train_dataset=splits["train"],
        eval_dataset=splits.get("validation"),
        data_collator=collator,
        processing_class=processor,
        compute_metrics=trainer_metrics,
        callbacks=callbacks,
    )

    checkpoint_dir = config.run_dir / "checkpoints"
    resume_checkpoint = None
    if config.resume and not config.force and checkpoint_dir.exists():
        resume_checkpoint = get_last_checkpoint(str(checkpoint_dir))
    started = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_seconds = time.perf_counter() - started

    model_dir = config.run_dir / "model"
    trainer.save_model(str(model_dir))
    processor.save_pretrained(model_dir)
    shutil.copy2(config.run_dir / "run_config.json", model_dir / "run_config.json")

    results: dict[str, Any] = {
        "run_name": config.resolved_run_name(),
        "model_family": config.model_family,
        "model_id": run_config["model_id"],
        "train_folds": list(config.train_folds),
        "validation_fold": config.validation_fold,
        "test_fold": config.test_fold,
        "seed": config.seed,
        "epochs_requested": config.epochs,
        "best_epoch": _best_epoch(trainer, config.epochs),
        "augment": config.augment,
        "train_seconds": train_seconds,
        "train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "checkpoint": str(model_dir),
    }

    temperature = 1.0
    if has_validation:
        validation_output = trainer.predict(splits["validation"], metric_key_prefix="validation")
        validation_logits = _extract_logits(validation_output.predictions)
        validation_metrics = classification_metrics(validation_output.label_ids, validation_logits)
        temperature = fit_temperature(validation_output.label_ids, validation_logits)
        results.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        results["temperature"] = temperature

    if "test" in splits:
        test_started = time.perf_counter()
        test_output = trainer.predict(splits["test"], metric_key_prefix="test")
        prediction_seconds = time.perf_counter() - test_started
        test_logits = _extract_logits(test_output.predictions)
        raw_metrics = classification_metrics(test_output.label_ids, test_logits)
        calibrated_metrics = classification_metrics(test_output.label_ids, test_logits / temperature)
        results.update({f"test_{key}": value for key, value in raw_metrics.items()})
        results["test_calibrated_ece"] = calibrated_metrics["ece"]
        results["test_calibrated_log_loss"] = calibrated_metrics["log_loss"]
        results["latency_ms_per_audio"] = prediction_seconds * 1000 / len(splits["test"])
        filenames = splits["test"]["filename"] if "filename" in splits["test"].column_names else None
        save_prediction_artifacts(
            config.run_dir / "analysis",
            test_output.label_ids,
            test_logits,
            labels,
            filenames,
            temperature,
        )

    results["model_size_bytes"] = _directory_size(model_dir)
    if config.verify_roundtrip:
        example_split = splits.get("test", splits["train"])
        results["roundtrip_max_logit_difference"] = _roundtrip_check(
            config.model_family, trainer, processor, model_dir, example_split[0]
        )

    save_json(config.run_dir / "metrics.json", results)
    _write_model_card(model_dir / "MODEL_CARD.md", config, results, labels)
    registry_row = {
        key: value
        for key, value in results.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    append_registry(config.output_root / "registry.csv", registry_row)
    return results


def screening(
    models: tuple[str, ...] = tuple(MODEL_SPECS),
    output_root: Path = Path("artifacts/screening"),
    seed: int = 42,
    **overrides: Any,
) -> pd.DataFrame:
    rows = []
    for family in models:
        config = ExperimentConfig(
            model_family=family,
            train_folds=(1, 2, 3),
            validation_fold=4,
            test_fold=5,
            output_root=output_root,
            seed=seed,
            **overrides,
        )
        rows.append(run_experiment(config))
    frame = pd.DataFrame(rows).sort_values("validation_f1_macro", ascending=False)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "summary.csv", index=False)
    save_json(output_root / "selected_models.json", frame.head(2)["model_family"].tolist())
    return frame


def cross_validate(
    models: tuple[str, ...],
    output_root: Path = Path("artifacts/cross_validation"),
    seed: int = 42,
    epochs_by_model: dict[str, int] | None = None,
    **overrides: Any,
) -> pd.DataFrame:
    epochs_by_model = epochs_by_model or {}
    rows = []
    for family in models:
        for test_fold in ALL_FOLDS:
            train_folds = tuple(fold for fold in ALL_FOLDS if fold != test_fold)
            config = ExperimentConfig(
                model_family=family,
                train_folds=train_folds,
                validation_fold=None,
                test_fold=test_fold,
                output_root=output_root,
                run_name=f"{family}-cv-fold{test_fold}-s{seed}",
                seed=seed,
                epochs=epochs_by_model.get(family, 3),
                early_stopping_patience=0,
                **overrides,
            )
            rows.append(run_experiment(config))
    runs = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_root / "fold_results.csv", index=False)
    summary = (
        runs.groupby("model_family")
        .agg(
            folds=("test_fold", "count"),
            f1_macro_mean=("test_f1_macro", "mean"),
            f1_macro_std=("test_f1_macro", "std"),
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            latency_ms_mean=("latency_ms_per_audio", "mean"),
            model_size_bytes=("model_size_bytes", "mean"),
        )
        .reset_index()
        .sort_values(
            ["f1_macro_mean", "accuracy_mean", "latency_ms_mean", "model_size_bytes"],
            ascending=[False, False, True, True],
        )
    )
    summary.to_csv(output_root / "summary.csv", index=False)
    save_json(output_root / "winner.json", summary.iloc[0].to_dict())
    return summary


def train_final_model(
    model_family: str,
    epochs: int,
    output_root: Path = Path("artifacts/final"),
    seed: int = 42,
    **overrides: Any,
) -> dict[str, Any]:
    return run_experiment(
        ExperimentConfig(
            model_family=model_family,
            train_folds=ALL_FOLDS,
            output_root=output_root,
            run_name=f"{model_family}-final-all-folds-s{seed}",
            seed=seed,
            epochs=epochs,
            early_stopping_patience=0,
            **overrides,
        )
    )


def augmentation_ablation(
    model_family: str,
    epochs: int,
    output_root: Path = Path("artifacts/ablation"),
    seed: int = 42,
    **overrides: Any,
) -> pd.DataFrame:
    rows = []
    overrides.pop("augment", None)
    for augment in (False, True):
        rows.append(
            run_experiment(
                ExperimentConfig(
                    model_family=model_family,
                    train_folds=(1, 2, 3),
                    validation_fold=4,
                    test_fold=5,
                    output_root=output_root,
                    run_name=f"{model_family}-ablation-{'augmented' if augment else 'plain'}-s{seed}",
                    seed=seed,
                    epochs=epochs,
                    augment=augment,
                    **overrides,
                )
            )
        )
    frame = pd.DataFrame(rows).sort_values("augment")
    frame.to_csv(output_root / "summary.csv", index=False)
    plain = frame.loc[frame["augment"] == False].iloc[0]  # noqa: E712
    augmented = frame.loc[frame["augment"] == True].iloc[0]  # noqa: E712
    save_json(
        output_root / "effect.json",
        {
            "model_family": model_family,
            "validation_f1_macro_delta": float(
                augmented["validation_f1_macro"] - plain["validation_f1_macro"]
            ),
            "test_f1_macro_delta": float(augmented["test_f1_macro"] - plain["test_f1_macro"]),
        },
    )
    return frame
