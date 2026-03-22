from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference import load_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the custom ruBERT checkpoint"
    )
    parser.add_argument("text", nargs="+", help="Text to classify")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/bert_custom"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = load_predictor(args.model_dir)
    predictions = predictor.predict(args.text)
    payload = [asdict(prediction) for prediction in predictions]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
