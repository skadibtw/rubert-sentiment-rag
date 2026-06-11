from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.compare_models import _collect_rows, _load_fallback_metrics


def test_collect_rows_uses_fallback_metrics_for_missing_artifacts(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "metrics.json").write_text(
        json.dumps({"accuracy": 0.75, "f1_macro": 0.76, "report_dict": {}}),
        encoding="utf-8",
    )
    binary_dir = tmp_path / "baseline_binary"
    binary_dir.mkdir()
    (binary_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": 0.94,
                "f1_macro": 0.94,
                "task": "binary_polarity",
                "report_dict": {
                    "0": {"f1-score": 0.93},
                    "1": {"f1-score": 0.95},
                },
            }
        ),
        encoding="utf-8",
    )

    fallback_path = tmp_path / "fallback.json"
    fallback_path.write_text(
        json.dumps(
            {
                "bert_trainer": {
                    "source": "previous_full_local_run",
                    "accuracy": 0.78,
                    "f1_macro": 0.79,
                }
            }
        ),
        encoding="utf-8",
    )

    rows, missing_models = _collect_rows(
        {
            "baseline": baseline_dir,
            "baseline_binary": binary_dir,
            "bert_trainer": tmp_path / "bert",
            "bert_custom": tmp_path / "bert_custom",
        },
        _load_fallback_metrics(fallback_path),
    )

    assert [row["model"] for row in rows] == [
        "baseline_binary",
        "bert_trainer",
        "baseline",
    ]
    assert rows[0]["task"] == "binary_polarity"
    assert rows[0]["f1_neutral"] is None
    assert rows[0]["f1_positive"] == 0.95
    assert rows[1]["source"] == "previous_full_local_run"
    assert rows[2]["source"] == "artifact"
    assert missing_models == ["bert_custom"]
