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
    "baseline_binary": Path("artifacts/baseline_binary"),
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
        "--baseline-binary-dir",
        type=Path,
        default=DEFAULT_MODEL_DIRS["baseline_binary"],
        help="Artifacts directory for the binary polarity baseline model",
    )
    parser.add_argument(
        "--custom-dir",
        type=Path,
        default=DEFAULT_MODEL_DIRS["bert_custom"],
        help="Artifacts directory for the custom PyTorch training loop",
    )
    parser.add_argument(
        "--fallback-metrics-path",
        type=Path,
        default=None,
        help="Optional JSON file with previous-run metrics for missing artifacts",
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


def _load_artifact_task(model_dir: Path, metrics: dict[str, object]) -> str:
    if "task" in metrics:
        return str(metrics["task"])
    metadata_path = model_dir / "sklearn_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return str(metadata.get("label_mode", "multiclass"))
    return "multiclass"


def _load_fallback_metrics(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    raw_metrics = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_metrics, dict):
        msg = "Fallback metrics file must contain a JSON object keyed by model name"
        raise ValueError(msg)
    fallback_metrics: dict[str, dict[str, object]] = {}
    for model_name, metrics in raw_metrics.items():
        if not isinstance(metrics, dict):
            msg = f"Fallback metrics for {model_name} must be a JSON object"
            raise ValueError(msg)
        fallback_metrics[str(model_name)] = metrics
    return fallback_metrics


def _row_from_metrics(
    model_name: str,
    model_dir: Path,
    metrics: dict[str, object],
    *,
    source: str,
    task: str,
) -> dict[str, object]:
    positive_label = "1" if task == "binary_polarity" else "2"
    return {
        "model": model_name,
        "artifacts_dir": str(model_dir),
        "task": task,
        "source": source,
        "accuracy": float(metrics["accuracy"]),
        "f1_macro": float(metrics["f1_macro"]),
        "f1_negative": _extract_label_f1(metrics, "0"),
        "f1_neutral": None
        if task == "binary_polarity"
        else _extract_label_f1(metrics, "1"),
        "f1_positive": _extract_label_f1(metrics, positive_label),
    }


def _collect_rows(
    model_dirs: dict[str, Path],
    fallback_metrics: dict[str, dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    missing_models: list[str] = []
    fallback_metrics = fallback_metrics or {}

    for model_name, model_dir in model_dirs.items():
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            metrics = fallback_metrics.get(model_name)
            if metrics is None:
                missing_models.append(model_name)
                continue
            rows.append(
                _row_from_metrics(
                    model_name,
                    model_dir,
                    metrics,
                    source=str(metrics.get("source", "fallback")),
                    task=str(metrics.get("task", "multiclass")),
                )
            )
            continue

        metrics = _load_metrics(metrics_path)
        task = _load_artifact_task(model_dir, metrics)
        rows.append(
            _row_from_metrics(
                model_name,
                model_dir,
                metrics,
                source="artifact",
                task=task,
            )
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
    best_by_task = {}
    for task, task_rows in pd.DataFrame(rows).groupby("task"):
        best_row = task_rows.sort_values("f1_macro", ascending=False).iloc[0]
        best_by_task[str(task)] = {
            "best_model": best_row["model"],
            "best_f1_macro": float(best_row["f1_macro"]),
        }
    return {
        "best_model": best_model,
        "best_f1_macro": best_f1,
        "best_by_task": best_by_task,
        "compared_models": [row["model"] for row in rows],
        "missing_models": missing_models,
    }


def main() -> None:
    args = parse_args()
    model_dirs = {
        "baseline": args.baseline_dir,
        "baseline_binary": args.baseline_binary_dir,
        "bert_trainer": args.bert_dir,
        "bert_custom": args.custom_dir,
    }
    fallback_metrics = _load_fallback_metrics(args.fallback_metrics_path)
    rows, missing_models = _collect_rows(model_dirs, fallback_metrics)

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
    for task, task_summary in summary["best_by_task"].items():
        report_lines.append(
            "Best "
            f"{task} model by macro F1: "
            f"{task_summary['best_model']} ({task_summary['best_f1_macro']:.4f})"
        )
    if missing_models:
        report_lines.append(f"Missing artifacts: {', '.join(missing_models)}")
    report_text = "\n".join(report_lines)
    (output_dir / "model_comparison.txt").write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nSaved comparison artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
