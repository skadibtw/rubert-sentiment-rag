from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_MODEL_DIRS = {
    "baseline": Path("artifacts/baseline"),
    "bert_trainer": Path("artifacts/bert"),
    "bert_custom": Path("artifacts/bert_custom"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved evaluation artifacts across models"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/comparison"),
        help="Directory where comparison files will be saved",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_MODEL_DIRS["baseline"],
        help="Artifacts directory for the classical baseline model",
    )
    parser.add_argument(
        "--bert-dir",
        type=Path,
        default=DEFAULT_MODEL_DIRS["bert_trainer"],
        help="Artifacts directory for the Hugging Face Trainer model",
    )
    parser.add_argument(
        "--custom-dir",
        type=Path,
        default=DEFAULT_MODEL_DIRS["bert_custom"],
        help="Artifacts directory for the custom PyTorch training loop",
    )
    return parser.parse_args()


def _load_metrics(metrics_path: Path) -> dict[str, object]:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _extract_label_f1(metrics: dict[str, object], label: str) -> float | None:
    report_dict = metrics.get("report_dict", {})
    if not isinstance(report_dict, dict):
        return None
    label_metrics = report_dict.get(label)
    if not isinstance(label_metrics, dict):
        return None
    f1_score = label_metrics.get("f1-score")
    return float(f1_score) if f1_score is not None else None


def _collect_rows(
    model_dirs: dict[str, Path],
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    missing_models: list[str] = []

    for model_name, model_dir in model_dirs.items():
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            missing_models.append(model_name)
            continue

        metrics = _load_metrics(metrics_path)
        rows.append(
            {
                "model": model_name,
                "artifacts_dir": str(model_dir),
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "f1_negative": _extract_label_f1(metrics, "0"),
                "f1_neutral": _extract_label_f1(metrics, "1"),
                "f1_positive": _extract_label_f1(metrics, "2"),
            }
        )

    rows.sort(key=lambda row: row["f1_macro"], reverse=True)
    return rows, missing_models


def _format_console_table(frame: pd.DataFrame) -> str:
    printable = frame.copy()
    for column in [
        "accuracy",
        "f1_macro",
        "f1_negative",
        "f1_neutral",
        "f1_positive",
    ]:
        if column in printable.columns:
            printable[column] = printable[column].map(
                lambda value: f"{value:.4f}" if pd.notna(value) else "-"
            )
    return printable.to_string(index=False)


def _build_summary(
    rows: list[dict[str, object]],
    missing_models: list[str],
) -> dict[str, object]:
    best_model = rows[0]["model"] if rows else None
    best_f1 = rows[0]["f1_macro"] if rows else None
    return {
        "best_model": best_model,
        "best_f1_macro": best_f1,
        "compared_models": [row["model"] for row in rows],
        "missing_models": missing_models,
    }


def main() -> None:
    args = parse_args()
    model_dirs = {
        "baseline": args.baseline_dir,
        "bert_trainer": args.bert_dir,
        "bert_custom": args.custom_dir,
    }
    rows, missing_models = _collect_rows(model_dirs)

    if not rows:
        raise FileNotFoundError(
            "No metrics.json files were found in the provided artifacts directories."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows)
    summary = _build_summary(rows, missing_models)

    (output_dir / "model_comparison.csv").write_text(
        frame.to_csv(index=False),
        encoding="utf-8",
    )
    (output_dir / "model_comparison.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "Model comparison",
        "",
        _format_console_table(frame),
        "",
        (
            "Best model by macro F1: "
            f"{summary['best_model']} ({summary['best_f1_macro']:.4f})"
        ),
    ]
    if missing_models:
        report_lines.append(f"Missing artifacts: {', '.join(missing_models)}")
    report_text = "\n".join(report_lines)
    (output_dir / "model_comparison.txt").write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nSaved comparison artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
