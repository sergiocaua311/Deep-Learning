from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from .data import SAMPLE_RATE, load_esc50
from .evaluation import (
    occlusion_explanation,
    probabilities_from_logits,
    robustness_evaluation,
)
from .experiment import best_device
from .models import AudioCollator, load_model_bundle


def load_audio_file(path: str | Path) -> dict[str, Any]:
    array, sampling_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return {"array": np.asarray(array, dtype=np.float32), "sampling_rate": sampling_rate}


def predict_file(checkpoint: str | Path, audio_path: str | Path, top_k: int = 5) -> list[dict[str, Any]]:
    family, model, processor = load_model_bundle(checkpoint)
    device = best_device()
    model.to(device).eval()
    collator = AudioCollator(family, processor, augment=False)
    example = {"audio": load_audio_file(audio_path), "target": 0}
    batch = collator([example])
    batch.pop("labels")
    inputs = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits.detach().cpu().numpy()
    probabilities = probabilities_from_logits(logits)[0]
    indices = np.argsort(probabilities)[-top_k:][::-1]
    return [
        {
            "class_id": int(index),
            "class": model.config.id2label[int(index)],
            "probability": float(probabilities[index]),
        }
        for index in indices
    ]


def evaluate_robustness(
    checkpoint: str | Path,
    test_fold: int,
    output_dir: str | Path,
    batch_size: int = 8,
    seed: int = 42,
    max_samples: int | None = None,
) -> None:
    family, model, processor = load_model_bundle(checkpoint)
    device = best_device()
    model.to(device).eval()
    dataset = load_esc50().filter(
        lambda fold: fold == test_fold,
        input_columns=["fold"],
        desc=f"Selecionando fold {test_fold}",
    )
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    examples = [dataset[index] for index in range(len(dataset))]
    collator = AudioCollator(family, processor, augment=False, seed=seed)
    frame = robustness_evaluation(
        model,
        collator,
        examples,
        device,
        output_dir,
        batch_size,
        seed,
    )
    print(frame.to_string(index=False))


def explain_file(
    checkpoint: str | Path,
    audio_path: str | Path,
    output_dir: str | Path,
    window_seconds: float = 0.5,
) -> None:
    family, model, processor = load_model_bundle(checkpoint)
    device = best_device()
    model.to(device).eval()
    collator = AudioCollator(family, processor, augment=False)
    frame = occlusion_explanation(
        model,
        collator,
        load_audio_file(audio_path),
        device,
        output_dir,
        window_seconds,
    )
    print(frame.to_string(index=False))


def format_predictions(predictions: list[dict[str, Any]]) -> str:
    return json.dumps(predictions, indent=2, ensure_ascii=False)
