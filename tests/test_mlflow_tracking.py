from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from src.experiments import MLflowRunConfig, log_run


class FakeRun:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events

    def __enter__(self) -> None:
        self.events.append(("enter_run", None))
        return None

    def __exit__(self, *args: object) -> None:
        del args
        self.events.append(("exit_run", None))


class FakeMLflow(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, object]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.events.append(("tracking_uri", uri))

    def set_experiment(self, name: str) -> None:
        self.events.append(("experiment", name))

    def start_run(self, run_name: str | None = None) -> FakeRun:
        self.events.append(("run_name", run_name))
        return FakeRun(self.events)

    def log_params(self, params: dict[str, object]) -> None:
        self.events.append(("params", params))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.events.append(("metrics", metrics))

    def log_artifacts(self, path: str) -> None:
        self.events.append(("artifacts", path))


def test_log_run_does_nothing_when_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)

    log_run(
        MLflowRunConfig(enabled=False),
        params={"model": "baseline"},
        metrics={"f1": 0.75},
        artifacts_dir=tmp_path,
    )


def test_log_run_sends_params_metrics_and_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    fake_mlflow = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    log_run(
        MLflowRunConfig(
            enabled=True,
            experiment_name="nlp-exp",
            run_name="baseline",
            tracking_uri="file:///tmp/mlruns",
        ),
        params={"path": Path("artifacts/baseline"), "none": None, "c": 0.7},
        metrics={"f1": 0.75, "nested": {"skip": True}},
        artifacts_dir=artifacts_dir,
    )

    assert ("tracking_uri", "file:///tmp/mlruns") in fake_mlflow.events
    assert ("experiment", "nlp-exp") in fake_mlflow.events
    assert ("run_name", "baseline") in fake_mlflow.events
    assert ("params", {"path": "artifacts/baseline", "c": 0.7}) in fake_mlflow.events
    assert ("metrics", {"f1": 0.75}) in fake_mlflow.events
    assert ("artifacts", str(artifacts_dir)) in fake_mlflow.events
