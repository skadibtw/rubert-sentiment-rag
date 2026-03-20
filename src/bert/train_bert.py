from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (
    DEFAULT_CACHE_DIR,
    get_dataset,
    maybe_sample_dataset,
    prepare_text_classification_dataset,
)


MODEL_NAME = "ai-forever/ruBERT-base"


def prepare_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int,
    keep_label_text: bool,
) -> DatasetDict:
    dataset = prepare_text_classification_dataset(
        dataset,
        keep_label_text=keep_label_text,
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)

    columns_to_keep = ["input_ids", "attention_mask", "label"]
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


def compute_metrics(eval_prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ruBERT for sentiment classification"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--keep-label-text",
        action="store_true",
        help="Keep label_text after tokenization for debugging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only prepare dataset and print columns without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = get_dataset(cache_dir=args.cache_dir)
    dataset = maybe_sample_dataset(dataset, args.sample_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_dataset = prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        keep_label_text=args.keep_label_text,
    )

    print("Columns before tokenization:", dataset["train"].column_names)
    print("Columns after tokenization:", tokenized_dataset["train"].column_names)

    if args.dry_run:
        return

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )

    training_args = TrainingArguments(
        output_dir="artifacts/bert",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(tokenized_dataset["test"])
    print(metrics)


if __name__ == "__main__":
    main()
