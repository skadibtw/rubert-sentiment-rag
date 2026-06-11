from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MLflowRunConfig:
    enabled: bool = False
    experiment_name: str = "rubert-sentiment-rag"
    run_name: str | None = None
    tracking_uri: str | None = None


def _flatten_params(params: dict[str, Any]) -> dict[str, str | int | float | bool]:
    flat: dict[str, str | int | float | bool] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float | str):
            flat[key] = value
            continue
        flat[key] = str(value)
    return flat


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            flat[key] = float(value)
    return flat


def log_run(
    config: MLflowRunConfig,
    *,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts_dir: Path | None = None,
) -> None:
    if not config.enabled:
        return

    try:
        import mlflow
    except ImportError as exc:
        msg = "MLflow tracking is enabled, but the 'mlflow' package is not installed"
        raise RuntimeError(msg) from exc

    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(_flatten_params(params))
        mlflow.log_metrics(_flatten_metrics(metrics))
        if artifacts_dir is not None and artifacts_dir.exists():
            mlflow.log_artifacts(str(artifacts_dir))
