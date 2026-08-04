from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from datasets import Audio, Dataset, DatasetDict, load_dataset


SAMPLE_RATE = 16_000
CLIP_SECONDS = 5
ALL_FOLDS = (1, 2, 3, 4, 5)


def validate_fold_spec(
    train_folds: Iterable[int],
    validation_fold: int | None,
    test_fold: int | None,
) -> tuple[int, ...]:
    train = tuple(sorted(set(train_folds)))
    if not train:
        raise ValueError("É necessário informar ao menos um fold de treino.")
    invalid = set(train) - set(ALL_FOLDS)
    if invalid:
        raise ValueError(f"Folds inválidos: {sorted(invalid)}")
    held_out = [fold for fold in (validation_fold, test_fold) if fold is not None]
    if any(fold not in ALL_FOLDS for fold in held_out):
        raise ValueError("Folds de validação e teste devem estar entre 1 e 5.")
    if len(held_out) != len(set(held_out)):
        raise ValueError("Validação e teste não podem usar o mesmo fold.")
    overlap = set(train).intersection(held_out)
    if overlap:
        raise ValueError(f"Vazamento entre treino e folds retidos: {sorted(overlap)}")
    return train


def read_metadata(path: str | Path = "data/esc50.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def validate_metadata(metadata: pd.DataFrame) -> None:
    required = {"filename", "fold", "target", "category", "src_file"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadados sem colunas obrigatórias: {sorted(missing)}")
    if len(metadata) != 2_000 or metadata["target"].nunique() != 50:
        raise ValueError("O ESC-50 deve conter 2.000 exemplos e 50 classes.")
    counts = metadata.groupby(["fold", "target"]).size()
    if len(counts) != 250 or not np.all(counts.to_numpy() == 8):
        raise ValueError("Cada fold oficial deve conter 8 exemplos de cada classe.")


def label_maps(metadata: pd.DataFrame) -> tuple[list[str], dict[str, int], dict[int, str]]:
    pairs = (
        metadata[["target", "category"]]
        .drop_duplicates()
        .sort_values("target")
    )
    labels = pairs["category"].tolist()
    if pairs["target"].tolist() != list(range(50)):
        raise ValueError("Os identificadores das classes devem cobrir o intervalo 0..49.")
    return labels, {label: index for index, label in enumerate(labels)}, dict(enumerate(labels))


def fold_indices(
    metadata: pd.DataFrame,
    folds: Iterable[int],
    seed: int = 42,
) -> list[int]:
    """Retorna índices estáveis; a ordem muda de forma reproduzível com a semente."""
    indices = metadata.index[metadata["fold"].isin(tuple(folds))].tolist()
    random.Random(seed).shuffle(indices)
    return indices


def load_esc50(dataset_name: str = "ashraq/esc50", sampling_rate: int = SAMPLE_RATE) -> Dataset:
    dataset = load_dataset(dataset_name)["train"]
    return dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))


def _select_folds(dataset: Dataset, folds: Iterable[int], description: str) -> Dataset:
    selected = set(folds)
    # Restringir a função à coluna ``fold`` evita decodificar os 2.000 áudios
    # apenas para construir os índices da divisão.
    return dataset.filter(
        lambda fold: fold in selected,
        input_columns=["fold"],
        desc=description,
    )


def make_splits(
    dataset: Dataset,
    train_folds: Iterable[int],
    validation_fold: int | None,
    test_fold: int | None,
    seed: int = 42,
) -> DatasetDict:
    train = validate_fold_spec(train_folds, validation_fold, test_fold)
    splits: dict[str, Dataset] = {
        "train": _select_folds(dataset, train, "Selecionando treino").shuffle(seed=seed)
    }
    if validation_fold is not None:
        splits["validation"] = _select_folds(
            dataset, (validation_fold,), "Selecionando validação"
        )
    if test_fold is not None:
        splits["test"] = _select_folds(dataset, (test_fold,), "Selecionando teste")
    return DatasetDict(splits)


def limit_splits(
    splits: DatasetDict,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> DatasetDict:
    limited = DatasetDict()
    for name, split in splits.items():
        limit = max_train_samples if name == "train" else max_eval_samples
        limited[name] = split.select(range(min(limit, len(split)))) if limit else split
    return limited


def assert_screening_balance(splits: DatasetDict) -> None:
    expected = {"train": 24, "validation": 8, "test": 8}
    for name, per_class in expected.items():
        if name not in splits:
            raise ValueError(f"Split obrigatório ausente: {name}")
        counts = np.bincount(splits[name]["target"], minlength=50)
        if not np.all(counts == per_class):
            raise ValueError(f"Split {name} não está balanceado em {per_class} exemplos/classe.")
