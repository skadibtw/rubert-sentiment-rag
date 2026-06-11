from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.plots import plot_confusion_matrix, plot_model_comparison


def test_plot_confusion_matrix_writes_png(tmp_path: Path) -> None:
    confusion_csv = tmp_path / "confusion.csv"
    pd.DataFrame(
        [[4, 1, 0], [1, 3, 1], [0, 1, 5]],
        index=["negative", "neutral", "positive"],
        columns=["negative", "neutral", "positive"],
    ).to_csv(confusion_csv)
    output_path = tmp_path / "confusion.png"

    plot_confusion_matrix(confusion_csv, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_model_comparison_writes_png(tmp_path: Path) -> None:
    comparison_csv = tmp_path / "comparison.csv"
    pd.DataFrame(
        [
            {"model": "baseline", "f1_macro": 0.75},
            {"model": "bert", "f1_macro": 0.79},
        ]
    ).to_csv(comparison_csv, index=False)
    output_path = tmp_path / "comparison.png"

    plot_model_comparison(comparison_csv, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
