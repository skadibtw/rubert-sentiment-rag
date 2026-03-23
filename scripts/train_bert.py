from __future__ import annotations

from scripts._runner import run_module_from_config

if __name__ == "__main__":
    run_module_from_config("configs/train_bert.yaml", "src.bert.train_bert")
