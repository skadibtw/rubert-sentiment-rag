from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (
    DEFAULT_CACHE_DIR,
    get_dataset,
    maybe_sample_dataset,
    prepare_text_classification_dataset,
)
from src.evaluation.metrics import (
    build_error_table,
    classification_metrics,
    confusion_matrix_frame,
    save_evaluation_artifacts,
)
from src.models import BertSentimentClassifier
from src.training import (
    TrainerConfig,
    create_dataloader,
    detect_device,
    evaluate_epoch,
    fit,
)


MODEL_NAME = "ai-forever/ruBERT-base"
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def prepare_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    keep_label_text: bool,
) -> DatasetDict:
    dataset = prepare_text_classification_dataset(
        dataset, keep_label_text=keep_label_text
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_batch, batched=True)
    columns_to_keep = ["text", "input_ids", "attention_mask", "label"]
    if keep_label_text and "label_text" in tokenized["train"].column_names:
        columns_to_keep.append("label_text")
    columns_to_remove = [
        column
        for column in tokenized["train"].column_names
        if column not in columns_to_keep
    ]
    tokenized = tokenized.remove_columns(columns_to_remove)
    tokenized = tokenized.rename_column("label", "labels")
    return tokenized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ruBERT with a custom PyTorch loop"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/bert_custom")
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--keep-label-text", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device()

    dataset = maybe_sample_dataset(
        get_dataset(cache_dir=args.cache_dir), args.sample_size
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        fix_mistral_regex=True,
    )
    tokenized_dataset = prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        keep_label_text=args.keep_label_text,
    )

    print("Device:", device)
    print("Columns before tokenization:", dataset["train"].column_names)
    print("Columns after tokenization:", tokenized_dataset["train"].column_names)

    if args.dry_run:
        return

    model = BertSentimentClassifier(
        model_name=args.model_name,
        num_labels=len(ID2LABEL),
        dropout=args.dropout,
    )
    trainer_config = TrainerConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        device=str(device),
    )

    history, best_checkpoint = fit(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        validation_dataset=tokenized_dataset["validation"],
        config=trainer_config,
    )

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    model.to(device)
    test_loader = create_dataloader(
        tokenized_dataset["test"],
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_metrics = evaluate_epoch(model, test_loader, device)
    report_metrics = classification_metrics(
        test_metrics["labels"], test_metrics["predictions"]
    )
    confusion = confusion_matrix_frame(
        test_metrics["labels"],
        test_metrics["predictions"],
        labels=[ID2LABEL[index] for index in sorted(ID2LABEL)],
    )
    errors = build_error_table(
        test_metrics["texts"],
        test_metrics["labels"],
        test_metrics["predictions"],
        id2label=ID2LABEL,
    )
    save_evaluation_artifacts(
        args.output_dir,
        metrics=report_metrics,
        confusion=confusion,
        errors=errors,
    )

    summary = {
        "best_checkpoint": str(best_checkpoint),
        "history": history,
        "test_accuracy": report_metrics["accuracy"],
        "test_f1_macro": report_metrics["f1_macro"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
