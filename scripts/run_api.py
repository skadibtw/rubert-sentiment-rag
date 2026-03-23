from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from src.utils import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/api.yaml"))
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    model_dir = config.get("model_dir")
    if model_dir is not None:
        os.environ["MODEL_DIR"] = str(model_dir)

    uvicorn.run(
        "src.api.app:app",
        host=str(config.get("host", "127.0.0.1")),
        port=int(config.get("port", 8000)),
        reload=bool(config.get("reload", False)),
    )


if __name__ == "__main__":
    main()
