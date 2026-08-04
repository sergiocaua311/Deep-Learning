from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    model_family: str
    train_folds: tuple[int, ...]
    validation_fold: int | None = None
    test_fold: int | None = None
    model_id: str | None = None
    run_name: str | None = None
    output_root: Path = Path("artifacts/runs")
    dataset_name: str = "ashraq/esc50"
    seed: int = 42
    epochs: int = 10
    learning_rate: float | None = None
    train_batch_size: int | None = None
    eval_batch_size: int | None = None
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    augment: bool = False
    resume: bool = True
    force: bool = False
    local_files_only: bool = False
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    verify_roundtrip: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        train = "-".join(map(str, self.train_folds))
        validation = self.validation_fold if self.validation_fold is not None else "none"
        test = self.test_fold if self.test_fold is not None else "none"
        suffix = "-aug" if self.augment else ""
        return f"{self.model_family}-train{train}-val{validation}-test{test}-s{self.seed}{suffix}"

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["train_folds"] = list(self.train_folds)
        return data
