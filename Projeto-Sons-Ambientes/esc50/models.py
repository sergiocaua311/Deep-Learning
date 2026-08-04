from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch import nn
import torchaudio
from transformers import (
    ASTForAudioClassification,
    AutoConfig,
    AutoFeatureExtractor,
    PretrainedConfig,
    PreTrainedModel,
    Wav2Vec2ForSequenceClassification,
    WhisperForAudioClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from .data import CLIP_SECONDS, SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class ModelSpec:
    family: str
    model_id: str | None
    input_key: str
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int


MODEL_SPECS = {
    "ast": ModelSpec(
        "ast", "MIT/ast-finetuned-audioset-10-10-0.4593", "input_values", 3e-5, 4, 4
    ),
    "wav2vec2": ModelSpec(
        "wav2vec2", "facebook/wav2vec2-base", "input_values", 3e-5, 2, 4
    ),
    "whisper": ModelSpec(
        "whisper", "openai/whisper-tiny", "input_features", 3e-5, 2, 4
    ),
    "cnn": ModelSpec("cnn", None, "input_features", 1e-3, 16, 32),
}
SUPPORTED_MODELS = tuple(MODEL_SPECS)


class Esc50CNNConfig(PretrainedConfig):
    model_type = "esc50-cnn"

    def __init__(
        self,
        num_labels: int = 50,
        channels: tuple[int, ...] | list[int] = (32, 64, 128),
        dropout: float = 0.3,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        )
        self.channels = list(channels)
        self.dropout = dropout


class Esc50CNNForAudioClassification(PreTrainedModel):
    config_class = Esc50CNNConfig
    base_model_prefix = "esc50_cnn"

    def __init__(self, config: Esc50CNNConfig) -> None:
        super().__init__(config)
        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in config.channels:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(in_channels, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> SequenceClassifierOutput:
        hidden = self.encoder(input_features)
        hidden = self.pool(hidden).flatten(1)
        logits = self.classifier(self.dropout(hidden))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


class CNNFeatureExtractor:
    model_input_names = ["input_features"]

    def __init__(
        self,
        sampling_rate: int = SAMPLE_RATE,
        n_fft: int = 1024,
        hop_length: int = 160,
        n_mels: int = 128,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def __call__(self, arrays: list[np.ndarray]) -> dict[str, torch.Tensor]:
        target_length = self.sampling_rate * CLIP_SECONDS
        normalized = []
        for array in arrays:
            waveform = torch.as_tensor(array, dtype=torch.float32).flatten()
            waveform = waveform[:target_length]
            waveform = nn.functional.pad(waveform, (0, max(0, target_length - waveform.numel())))
            normalized.append(waveform)
        waveforms = torch.stack(normalized)
        features = self.to_db(self.mel(waveforms)).unsqueeze(1)
        means = features.mean(dim=(-2, -1), keepdim=True)
        stds = features.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return {"input_features": (features - means) / stds}

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_extractor_type": type(self).__name__,
            "sampling_rate": self.sampling_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
        }
        (path / "preprocessor_config.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "CNNFeatureExtractor":
        payload = json.loads((Path(path) / "preprocessor_config.json").read_text())
        allowed = {key: payload[key] for key in ("sampling_rate", "n_fft", "hop_length", "n_mels")}
        return cls(**allowed)


def _label_config(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    return {label: index for index, label in enumerate(labels)}, dict(enumerate(labels))


def build_model_bundle(
    family: str,
    labels: list[str],
    model_id: str | None = None,
    local_files_only: bool = False,
) -> tuple[PreTrainedModel, Any]:
    if family not in MODEL_SPECS:
        raise ValueError(f"Modelo inválido: {family}. Opções: {', '.join(SUPPORTED_MODELS)}")
    label2id, id2label = _label_config(labels)
    if family == "cnn":
        config = Esc50CNNConfig(num_labels=len(labels), label2id=label2id, id2label=id2label)
        return Esc50CNNForAudioClassification(config), CNNFeatureExtractor()

    source = model_id or MODEL_SPECS[family].model_id
    config = AutoConfig.from_pretrained(
        source,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        local_files_only=local_files_only,
    )
    processor = AutoFeatureExtractor.from_pretrained(source, local_files_only=local_files_only)
    model_class = {
        "ast": ASTForAudioClassification,
        "wav2vec2": Wav2Vec2ForSequenceClassification,
        "whisper": WhisperForAudioClassification,
    }[family]
    model = model_class.from_pretrained(
        source,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )
    return model, processor


def load_model_bundle(checkpoint: str | Path) -> tuple[str, PreTrainedModel, Any]:
    checkpoint = Path(checkpoint)
    if (checkpoint / "model").is_dir():
        checkpoint = checkpoint / "model"
    metadata_path = checkpoint / "run_config.json"
    if not metadata_path.exists():
        metadata_path = checkpoint.parent / "run_config.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        family = metadata["model_family"]
    else:
        # Compatibilidade com os checkpoints AST produzidos pelo notebook antigo.
        config_path = checkpoint / "config.json"
        if not config_path.exists():
            raise FileNotFoundError("run_config.json/config.json não encontrado no checkpoint.")
        model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
        family_by_type = {
            "audio-spectrogram-transformer": "ast",
            "wav2vec2": "wav2vec2",
            "whisper": "whisper",
            "esc50-cnn": "cnn",
        }
        if model_type not in family_by_type:
            raise ValueError(f"Tipo de checkpoint não reconhecido: {model_type}")
        family = family_by_type[model_type]
    if family == "cnn":
        model = Esc50CNNForAudioClassification.from_pretrained(checkpoint)
        processor = CNNFeatureExtractor.from_pretrained(checkpoint)
    else:
        model_class = {
            "ast": ASTForAudioClassification,
            "wav2vec2": Wav2Vec2ForSequenceClassification,
            "whisper": WhisperForAudioClassification,
        }[family]
        model = model_class.from_pretrained(checkpoint)
        processor = AutoFeatureExtractor.from_pretrained(checkpoint)
    return family, model, processor


def normalize_audio(audio: dict[str, Any], target_rate: int = SAMPLE_RATE) -> np.ndarray:
    array = np.asarray(audio["array"], dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=0)
    rate = int(audio["sampling_rate"])
    if rate != target_rate:
        array = librosa.resample(array, orig_sr=rate, target_sr=target_rate)
    target_length = target_rate * CLIP_SECONDS
    array = np.asarray(array[:target_length], dtype=np.float32)
    if len(array) < target_length:
        array = np.pad(array, (0, target_length - len(array)))
    return array


def augment_waveform(array: np.ndarray, seed: int, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.array(array, dtype=np.float32, copy=True)
    shift = int(rng.uniform(-0.5, 0.5) * sampling_rate)
    shifted = np.zeros_like(output)
    if shift > 0:
        shifted[shift:] = output[:-shift]
    elif shift < 0:
        shifted[:shift] = output[-shift:]
    else:
        shifted = output
    gain = 10 ** (rng.uniform(-6.0, 6.0) / 20.0)
    output = np.clip(shifted * gain, -1.0, 1.0)
    if rng.random() < 0.5:
        rms = np.sqrt(np.mean(output**2) + 1e-12)
        noise = rng.normal(size=output.shape).astype(np.float32)
        noise_rms = np.sqrt(np.mean(noise**2) + 1e-12)
        output = output + noise * (rms / (noise_rms * 10 ** (20 / 20)))
    return np.clip(output, -1.0, 1.0).astype(np.float32)


class AudioCollator:
    def __init__(self, family: str, processor: Any, augment: bool = False, seed: int = 42) -> None:
        self.family = family
        self.processor = processor
        self.augment = augment
        self.seed = seed
        self.calls = 0

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        arrays = []
        for index, example in enumerate(examples):
            array = normalize_audio(example["audio"])
            if self.augment and example.get("augment", False):
                array = augment_waveform(array, self.seed + self.calls * 10_007 + index)
            arrays.append(array)
        self.calls += 1

        if self.family == "cnn":
            batch = self.processor(arrays)
        else:
            options: dict[str, Any] = {
                "sampling_rate": SAMPLE_RATE,
                "return_tensors": "pt",
            }
            if self.family == "wav2vec2":
                options.update(padding=True, return_attention_mask=True)
            batch = dict(self.processor(arrays, **options))
            allowed = {
                "ast": {"input_values"},
                "wav2vec2": {"input_values", "attention_mask"},
                "whisper": {"input_features"},
            }[self.family]
            batch = {key: value for key, value in batch.items() if key in allowed}

        augment_rows = [bool(example.get("augment", False)) for example in examples]
        if self.augment and any(augment_rows) and self.family in {"ast", "whisper", "cnn"}:
            self._apply_specaugment(batch, augment_rows)

        labels = [example.get("labels", example.get("target")) for example in examples]
        if any(label is None for label in labels):
            raise ValueError("Exemplo sem target/labels.")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        for key, tensor in batch.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Tensor {key} contém valores não finitos.")
        return batch

    def _apply_specaugment(self, batch: dict[str, torch.Tensor], rows: list[bool]) -> None:
        """Aplica máscaras leves de tempo/frequência somente em lotes de treino."""
        key = "input_values" if self.family == "ast" else "input_features"
        features = batch[key]
        rng = np.random.default_rng(self.seed + self.calls * 65_537)
        for row, enabled in enumerate(rows):
            if not enabled:
                continue
            if self.family == "ast":
                time_axis, frequency_axis = 0, 1
                view = features[row]
            elif self.family == "whisper":
                frequency_axis, time_axis = 0, 1
                view = features[row]
            else:
                frequency_axis, time_axis = 1, 2
                view = features[row]
            time_size = view.shape[time_axis]
            frequency_size = view.shape[frequency_axis]
            time_width = max(1, int(time_size * 0.08))
            frequency_width = max(1, int(frequency_size * 0.05))
            time_start = int(rng.integers(0, max(1, time_size - time_width + 1)))
            frequency_start = int(rng.integers(0, max(1, frequency_size - frequency_width + 1)))
            time_slices = [slice(None)] * view.ndim
            time_slices[time_axis] = slice(time_start, time_start + time_width)
            frequency_slices = [slice(None)] * view.ndim
            frequency_slices[frequency_axis] = slice(
                frequency_start, frequency_start + frequency_width
            )
            view[tuple(time_slices)] = 0
            view[tuple(frequency_slices)] = 0


def parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
