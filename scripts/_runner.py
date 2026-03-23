from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.utils import load_yaml_config


def _normalize_flag(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _flatten_cli_args(config: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    for key, value in config.items():
        flag = _normalize_flag(key)
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                cli_args.extend([flag, str(item)])
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def run_module_from_config(default_config: str, module_name: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(default_config))
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    command = [sys.executable, "-m", module_name, *_flatten_cli_args(config)]
    subprocess.run(command, check=True)
