from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".tmp/matplotlib")

import matplotlib
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

matplotlib.use("Agg")


def plot_confusion_matrix(
    confusion_csv: Path,
    output_path: Path,
    *,
    title: str = "Baseline confusion matrix",
) -> None:
    frame = pd.read_csv(confusion_csv, index_col=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(frame.to_numpy(), cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(frame.columns)), labels=frame.columns)
    ax.set_yticks(range(len(frame.index)), labels=frame.index)

    threshold = frame.to_numpy().max() / 2
    for row_index, row_name in enumerate(frame.index):
        for column_index, column_name in enumerate(frame.columns):
            value = int(frame.loc[row_name, column_name])
            color = "white" if value > threshold else "black"
            ax.text(
                column_index,
                row_index,
                f"{value:,}",
                ha="center",
                va="center",
                color=color,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_model_comparison(
    comparison_csv: Path,
    output_path: Path,
    *,
    metric_column: str = "f1_macro",
    title: str = "Model comparison by macro F1",
) -> None:
    frame = pd.read_csv(comparison_csv)
    if metric_column not in frame.columns:
        msg = f"Metric column does not exist in comparison CSV: {metric_column}"
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = frame.sort_values(metric_column, ascending=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(plot_frame["model"], plot_frame[metric_column], color="#2f6f8f")
    ax.set_title(title)
    ax.set_xlabel(metric_column)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PNG plots from saved evaluation artifacts"
    )
    parser.add_argument(
        "--baseline-confusion-csv",
        type=Path,
        default=Path("artifacts/baseline/confusion_matrix.csv"),
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("artifacts/comparison/model_comparison.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_confusion_matrix(
        args.baseline_confusion_csv,
        args.output_dir / "baseline_confusion_matrix.png",
    )
    plot_model_comparison(
        args.comparison_csv,
        args.output_dir / "model_comparison_macro_f1.png",
    )
    print(f"Saved evaluation plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
