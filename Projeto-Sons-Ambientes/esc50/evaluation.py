from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from .data import SAMPLE_RATE
from .models import AudioCollator, normalize_audio


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits)
    predictions = logits.argmax(axis=1)
    top_k = min(5, logits.shape[1])
    top_indices = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
    top5 = np.mean([label in row for label, row in zip(labels, top_indices)])
    probabilities = probabilities_from_logits(logits)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(logits.shape[1]),
                average="macro",
                zero_division=0,
            )
        ),
        "top5_accuracy": float(top5),
        "log_loss": float(log_loss(labels, probabilities, labels=np.arange(logits.shape[1]))),
        "ece": float(expected_calibration_error(labels, probabilities)),
    }


def trainer_metrics(eval_prediction: Any) -> dict[str, float]:
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return classification_metrics(eval_prediction.label_ids, logits)


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def fit_temperature(validation_labels: np.ndarray, validation_logits: np.ndarray) -> float:
    """Escolhe temperatura apenas na validação, nunca no teste."""
    candidates = np.geomspace(0.25, 4.0, 121)
    losses = [
        log_loss(
            validation_labels,
            probabilities_from_logits(validation_logits, value),
            labels=np.arange(validation_logits.shape[1]),
        )
        for value in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_prediction_artifacts(
    output_dir: str | Path,
    labels: np.ndarray,
    logits: np.ndarray,
    class_names: list[str],
    filenames: Iterable[str] | None = None,
    temperature: float = 1.0,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits)
    probabilities = probabilities_from_logits(logits, temperature)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    names = list(filenames or [f"example-{index}" for index in range(len(labels))])
    metrics = classification_metrics(labels, logits / temperature)

    top_indices = np.argsort(probabilities, axis=1)[:, -5:][:, ::-1]
    rows = []
    for index, (true, predicted) in enumerate(zip(labels, predictions)):
        rows.append(
            {
                "filename": names[index],
                "true_id": int(true),
                "true_class": class_names[true],
                "predicted_id": int(predicted),
                "predicted_class": class_names[predicted],
                "confidence": float(confidence[index]),
                "correct": bool(true == predicted),
                "top5": "|".join(class_names[item] for item in top_indices[index]),
            }
        )
    predictions_frame = pd.DataFrame(rows)
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)
    predictions_frame.query("not correct").sort_values("confidence", ascending=False).to_csv(
        output_dir / "high_confidence_errors.csv", index=False
    )

    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=90, fontsize=6)
    axis.set_yticklabels(class_names, fontsize=6)
    axis.set_xlabel("Classe prevista")
    axis.set_ylabel("Classe real")
    axis.set_title("Matriz de confusão normalizada — ESC-50")
    fig.colorbar(image, ax=axis, fraction=0.025)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    calibration_rows = []
    edges = np.linspace(0, 1, 11)
    correctness = predictions == labels
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        calibration_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else math.nan,
                "accuracy": float(correctness[mask].mean()) if mask.any() else math.nan,
            }
        )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(output_dir / "calibration.csv", index=False)
    nonempty = calibration.query("count > 0")
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="calibração perfeita")
    axis.plot(nonempty["mean_confidence"], nonempty["accuracy"], marker="o")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confiança", ylabel="Acurácia")
    axis.set_title(f"Diagrama de confiabilidade — ECE {metrics['ece']:.3f}")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=160)
    plt.close(fig)
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def _stable_seed(value: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def perturb_audio(
    array: np.ndarray,
    condition: str,
    identifier: str,
    seed: int = 42,
) -> np.ndarray:
    output = np.asarray(array, dtype=np.float32).copy()
    if condition == "clean":
        return output
    if condition.startswith("shift_"):
        seconds = float(condition.split("_", 1)[1])
        amount = int(seconds * SAMPLE_RATE)
        shifted = np.zeros_like(output)
        if amount > 0:
            shifted[amount:] = output[:-amount]
        else:
            shifted[:amount] = output[-amount:]
        return shifted
    if condition.startswith("gain_"):
        decibels = float(condition.split("_", 1)[1])
        return np.clip(output * 10 ** (decibels / 20), -1, 1).astype(np.float32)
    if condition.startswith("noise_"):
        snr = float(condition.split("_", 1)[1])
        rng = np.random.default_rng(_stable_seed(identifier, seed))
        noise = rng.normal(size=output.shape).astype(np.float32)
        signal_rms = np.sqrt(np.mean(output**2) + 1e-12)
        noise_rms = np.sqrt(np.mean(noise**2) + 1e-12)
        scaled = noise * signal_rms / (noise_rms * 10 ** (snr / 20))
        return np.clip(output + scaled, -1, 1).astype(np.float32)
    raise ValueError(f"Condição de robustez desconhecida: {condition}")


ROBUSTNESS_CONDITIONS = (
    "clean",
    "shift_-0.5",
    "shift_0.5",
    "gain_-6",
    "gain_6",
    "noise_20",
    "noise_10",
    "noise_0",
)


def predict_examples(
    model: torch.nn.Module,
    collator: AudioCollator,
    examples: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 8,
    transform: Callable[[np.ndarray, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    model.eval()
    started = time.perf_counter()
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        if transform:
            transformed = []
            for index, example in enumerate(batch_examples):
                item = dict(example)
                identifier = str(item.get("filename", start + index))
                item["audio"] = {
                    "array": transform(normalize_audio(item["audio"]), identifier),
                    "sampling_rate": SAMPLE_RATE,
                }
                transformed.append(item)
            batch_examples = transformed
        batch = collator(batch_examples)
        labels = batch.pop("labels")
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.numpy())
    elapsed = time.perf_counter() - started
    return np.concatenate(all_labels), np.concatenate(all_logits), elapsed * 1000 / len(examples)


def robustness_evaluation(
    model: torch.nn.Module,
    collator: AudioCollator,
    examples: list[dict[str, Any]],
    device: torch.device,
    output_dir: str | Path,
    batch_size: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for condition in ROBUSTNESS_CONDITIONS:
        transform = lambda array, identifier, name=condition: perturb_audio(
            array, name, identifier, seed
        )
        labels, logits, latency = predict_examples(
            model, collator, examples, device, batch_size, transform
        )
        row = {"condition": condition, **classification_metrics(labels, logits)}
        row["latency_ms_per_audio"] = latency
        rows.append(row)
    frame = pd.DataFrame(rows)
    clean_accuracy = frame.loc[frame["condition"] == "clean", "accuracy"].iloc[0]
    clean_f1 = frame.loc[frame["condition"] == "clean", "f1_macro"].iloc[0]
    frame["accuracy_drop"] = clean_accuracy - frame["accuracy"]
    frame["f1_macro_drop"] = clean_f1 - frame["f1_macro"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "robustness.csv", index=False)
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(frame["condition"], frame["f1_macro"], color="tab:blue")
    axis.set(ylim=(0, 1), ylabel="F1 macro", title="Robustez a perturbações")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness.png", dpi=160)
    plt.close(fig)
    return frame


def occlusion_explanation(
    model: torch.nn.Module,
    collator: AudioCollator,
    audio: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
    window_seconds: float = 0.5,
) -> pd.DataFrame:
    array = normalize_audio(audio)
    example = {"audio": {"array": array, "sampling_rate": SAMPLE_RATE}, "target": 0}
    _, original_logits, _ = predict_examples(model, collator, [example], device)
    original_probabilities = probabilities_from_logits(original_logits)[0]
    target = int(original_probabilities.argmax())
    window = int(window_seconds * SAMPLE_RATE)
    rows = []
    for start in range(0, len(array), window):
        occluded = array.copy()
        occluded[start : start + window] = 0
        item = {"audio": {"array": occluded, "sampling_rate": SAMPLE_RATE}, "target": target}
        _, logits, _ = predict_examples(model, collator, [item], device)
        probability = probabilities_from_logits(logits)[0, target]
        rows.append(
            {
                "start_seconds": start / SAMPLE_RATE,
                "end_seconds": min(start + window, len(array)) / SAMPLE_RATE,
                "importance": float(original_probabilities[target] - probability),
            }
        )
    frame = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "occlusion.csv", index=False)
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(frame["start_seconds"], frame["importance"], width=window_seconds * 0.9)
    axis.set(xlabel="Tempo (s)", ylabel="Queda de probabilidade", title="Importância por oclusão")
    axis.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "occlusion.png", dpi=160)
    plt.close(fig)
    save_json(
        output_dir / "occlusion_metadata.json",
        {"predicted_id": target, "original_probability": float(original_probabilities[target])},
    )
    return frame


def append_registry(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
