from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .experiment import (
    augmentation_ablation,
    cross_validate,
    run_experiment,
    screening,
    train_final_model,
)
from .inference import evaluate_robustness, explain_file, format_predictions, predict_file
from .models import SUPPORTED_MODELS
from .report import summarize_registry


def _folds(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use folds separados por vírgula, por exemplo 1,2,3.") from error


def _models(value: str) -> tuple[str, ...]:
    models = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = set(models) - set(SUPPORTED_MODELS)
    if invalid:
        raise argparse.ArgumentTypeError(f"Modelos inválidos: {', '.join(sorted(invalid))}")
    return models


def _epochs_by_model(value: str) -> dict[str, int]:
    result = {}
    try:
        for item in value.split(","):
            model, epochs = item.split("=", 1)
            result[model] = int(epochs)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use o formato ast=3,whisper=5.") from error
    return result


def _add_runtime_options(
    parser: argparse.ArgumentParser,
    *,
    include_epochs: bool = True,
    include_augment: bool = True,
) -> None:
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    if include_epochs:
        parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    if include_augment:
        parser.add_argument("--augment", action="store_true")
    else:
        parser.set_defaults(augment=False)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-roundtrip", action="store_true")


def _overrides(args: argparse.Namespace, include_epochs: bool = True) -> dict[str, Any]:
    values: dict[str, Any] = {
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "augment": args.augment,
        "local_files_only": args.local_files_only,
        "resume": not args.no_resume,
        "force": args.force,
        "verify_roundtrip": not args.no_roundtrip,
    }
    if include_epochs and args.epochs is not None:
        values["epochs"] = args.epochs
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m esc50", description="Experimentos ESC-50")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Treina uma arquitetura")
    train.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    train.add_argument("--model-id")
    train.add_argument("--train-folds", type=_folds, required=True)
    train.add_argument("--validation-fold", type=int)
    train.add_argument("--test-fold", type=int)
    train.add_argument("--run-name")
    _add_runtime_options(train)

    screen = subparsers.add_parser("screen", help="Executa a triagem das quatro arquiteturas")
    screen.add_argument("--models", type=_models, default=SUPPORTED_MODELS)
    _add_runtime_options(screen)

    cross = subparsers.add_parser("cross-validate", help="Executa os cinco folds oficiais")
    cross.add_argument("--models", type=_models, required=True)
    cross.add_argument("--model-epochs", type=_epochs_by_model, default={})
    _add_runtime_options(cross, include_epochs=False)
    cross.set_defaults(epochs=None)

    ablation = subparsers.add_parser("ablation", help="Compara treino com e sem augmentação")
    ablation.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    _add_runtime_options(ablation, include_augment=False)

    final = subparsers.add_parser("train-final", help="Treina o vencedor usando os cinco folds")
    final.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    final.add_argument("--epochs", type=int, required=True)
    final.add_argument("--output-root", type=Path, default=Path("artifacts/final"))
    final.add_argument("--seed", type=int, default=42)
    final.add_argument("--augment", action="store_true")
    final.add_argument("--local-files-only", action="store_true")

    predict = subparsers.add_parser("predict", help="Classifica WAV, MP3 ou FLAC")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--audio", type=Path, required=True)
    predict.add_argument("--top-k", type=int, default=5)

    robust = subparsers.add_parser("robustness", help="Avalia perturbações determinísticas")
    robust.add_argument("--checkpoint", type=Path, required=True)
    robust.add_argument("--test-fold", type=int, required=True)
    robust.add_argument("--output-dir", type=Path, required=True)
    robust.add_argument("--batch-size", type=int, default=8)
    robust.add_argument("--seed", type=int, default=42)
    robust.add_argument("--max-samples", type=int)

    explain = subparsers.add_parser("explain", help="Gera explicação por oclusão temporal")
    explain.add_argument("--checkpoint", type=Path, required=True)
    explain.add_argument("--audio", type=Path, required=True)
    explain.add_argument("--output-dir", type=Path, required=True)
    explain.add_argument("--window-seconds", type=float, default=0.5)

    report = subparsers.add_parser("summarize", help="Consolida o registro de execuções")
    report.add_argument("--registry", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, default=Path("artifacts/report"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        config = ExperimentConfig(
            model_family=args.model,
            model_id=args.model_id,
            train_folds=args.train_folds,
            validation_fold=args.validation_fold,
            test_fold=args.test_fold,
            run_name=args.run_name,
            output_root=args.output_root or Path("artifacts/runs"),
            seed=args.seed,
            **_overrides(args),
        )
        result = run_experiment(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "screen":
        frame = screening(
            models=args.models,
            output_root=args.output_root or Path("artifacts/screening"),
            seed=args.seed,
            **_overrides(args),
        )
        print(frame.to_string(index=False))
    elif args.command == "cross-validate":
        overrides = _overrides(args, include_epochs=False)
        frame = cross_validate(
            args.models,
            args.output_root or Path("artifacts/cross_validation"),
            args.seed,
            args.model_epochs,
            **overrides,
        )
        print(frame.to_string(index=False))
    elif args.command == "ablation":
        overrides = _overrides(args, include_epochs=False)
        frame = augmentation_ablation(
            args.model,
            args.epochs or 3,
            args.output_root or Path("artifacts/ablation"),
            args.seed,
            **overrides,
        )
        print(frame.to_string(index=False))
    elif args.command == "train-final":
        result = train_final_model(
            args.model,
            args.epochs,
            args.output_root,
            args.seed,
            augment=args.augment,
            local_files_only=args.local_files_only,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "predict":
        print(format_predictions(predict_file(args.checkpoint, args.audio, args.top_k)))
    elif args.command == "robustness":
        evaluate_robustness(
            args.checkpoint,
            args.test_fold,
            args.output_dir,
            args.batch_size,
            args.seed,
            args.max_samples,
        )
    elif args.command == "explain":
        explain_file(
            args.checkpoint,
            args.audio,
            args.output_dir,
            args.window_seconds,
        )
    elif args.command == "summarize":
        print(summarize_registry(args.registry, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
