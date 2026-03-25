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

    rag_index_dir = config.get("rag_index_dir")
    if rag_index_dir is not None:
        os.environ["RAG_INDEX_DIR"] = str(rag_index_dir)

    rag_generation_mode = config.get("rag_generation_mode")
    if rag_generation_mode is not None:
        os.environ["RAG_GENERATION_MODE"] = str(rag_generation_mode)

    rag_llm_model = config.get("rag_llm_model")
    if rag_llm_model is not None:
        os.environ["RAG_LLM_MODEL"] = str(rag_llm_model)

    rag_llm_base_url = config.get("rag_llm_base_url")
    if rag_llm_base_url is not None:
        os.environ["RAG_LLM_BASE_URL"] = str(rag_llm_base_url)

    uvicorn.run(
        "src.api.app:app",
        host=str(config.get("host", "127.0.0.1")),
        port=int(config.get("port", 8000)),
        reload=bool(config.get("reload", False)),
    )


if __name__ == "__main__":
    main()
