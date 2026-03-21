from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "report_text": classification_report(y_true, y_pred, zero_division=0),
        "report_dict": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def confusion_matrix_frame(
    y_true: list[int], y_pred: list[int], labels: list[str] | None = None
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = [str(index) for index in range(len(matrix))]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def build_error_table(
    texts: list[str],
    y_true: list[int],
    y_pred: list[int],
    id2label: dict[int, str] | None = None,
) -> pd.DataFrame:
    rows = []
    for text, true_label, pred_label in zip(texts, y_true, y_pred):
        if true_label == pred_label:
            continue
        rows.append(
            {
                "text": text,
                "true_label": true_label,
                "pred_label": pred_label,
                "true_label_text": id2label.get(true_label, str(true_label))
                if id2label
                else str(true_label),
                "pred_label_text": id2label.get(pred_label, str(pred_label))
                if id2label
                else str(pred_label),
            }
        )
    return pd.DataFrame(rows)


def save_evaluation_artifacts(
    output_dir: Path,
    *,
    metrics: dict[str, object],
    confusion: pd.DataFrame,
    errors: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion.to_csv(output_dir / "confusion_matrix.csv", index=True)
    errors.to_csv(output_dir / "errors.csv", index=False)

    serializable_metrics = dict(metrics)
    report_dict = serializable_metrics.get("report_dict")
    if report_dict is not None:
        serializable_metrics["report_dict"] = report_dict
    (output_dir / "classification_report.txt").write_text(
        str(metrics["report_text"]),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(serializable_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
