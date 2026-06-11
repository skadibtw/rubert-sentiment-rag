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
            "bert_trainer": tmp_path / "bert",
            "bert_custom": tmp_path / "bert_custom",
        },
        _load_fallback_metrics(fallback_path),
    )

    assert [row["model"] for row in rows] == ["bert_trainer", "baseline"]
    assert rows[0]["source"] == "previous_full_local_run"
    assert rows[1]["source"] == "artifact"
    assert missing_models == ["bert_custom"]
