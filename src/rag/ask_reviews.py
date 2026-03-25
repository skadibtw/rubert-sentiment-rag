from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.rag.pipeline import DEFAULT_RAG_DIR, load_review_rag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question over indexed reviews")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--question", dest="question_flag", default=None)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_RAG_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sentiment-focus", default=None)
    parser.add_argument("--generation-mode", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.question = args.question or args.question_flag
    if not args.question:
        parser.error("question is required")
    return args


def main() -> None:
    args = parse_args()
    rag = load_review_rag(args.index_dir, device=args.device)
    response = rag.answer(
        args.question,
        top_k=args.top_k,
        sentiment_focus=args.sentiment_focus,
        generation_mode=args.generation_mode,
    )
    print(json.dumps(asdict(response), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
