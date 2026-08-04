from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def summarize_registry(registry: str | Path, output_dir: str | Path) -> pd.DataFrame:
    registry = Path(registry)
    if not registry.exists():
        raise FileNotFoundError(f"Registro não encontrado: {registry}")
    frame = pd.read_csv(registry)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        column
        for column in (
            "validation_f1_macro",
            "test_f1_macro",
            "test_accuracy",
            "latency_ms_per_audio",
            "model_size_bytes",
        )
        if column in frame
    ]
    columns = ["run_name", "model_family", *metrics]
    summary = frame[columns].sort_values(
        "test_f1_macro" if "test_f1_macro" in frame else metrics[0], ascending=False
    )
    summary.to_csv(output_dir / "experiment_summary.csv", index=False)

    if {"test_f1_macro", "latency_ms_per_audio"}.issubset(frame.columns):
        fig, axis = plt.subplots(figsize=(8, 5))
        for family, group in frame.groupby("model_family"):
            axis.scatter(group["latency_ms_per_audio"], group["test_f1_macro"], label=family)
        axis.set(xlabel="Latência (ms/áudio)", ylabel="F1 macro", title="Qualidade versus custo")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "quality_vs_latency.png", dpi=160)
        plt.close(fig)
    return summary
