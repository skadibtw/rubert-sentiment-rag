from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from src.evaluation.metrics import classification_metrics


@dataclass(slots=True)
class TrainerConfig:
    output_dir: Path
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    device: str = "cpu"


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_dataloader(
    dataset: object,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collator(features: list[dict[str, Any]]) -> dict[str, Any]:
        texts = (
            [feature.pop("text") for feature in features]
            if "text" in features[0]
            else None
        )
        label_texts = (
            [feature.pop("label_text") for feature in features]
            if "label_text" in features[0]
            else None
        )
        batch = base_collator(features)
        if texts is not None:
            batch["text"] = texts
        if label_texts is not None:
            batch["label_text"] = label_texts
        return batch

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="train", leave=False):
        batch = _move_batch(batch, device)
        optimizer.zero_grad()
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = loss_fn(logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    texts: list[str] = []
    predictions: list[int] = []
    labels: list[int] = []
    loss_fn = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch_texts = batch.pop("text", None)
        batch.pop("label_text", None)
        batch = _move_batch(batch, device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = loss_fn(logits, batch["labels"])
        total_loss += loss.item()
        predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
        if batch_texts is not None:
            texts.extend(batch_texts)

    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    metrics["predictions"] = predictions
    metrics["labels"] = labels
    metrics["texts"] = texts
    return metrics


def fit(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: object,
    validation_dataset: object,
    config: TrainerConfig,
) -> tuple[list[dict[str, float]], Path]:
    output_dir = config.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)
    model.to(device)

    train_loader = create_dataloader(
        train_dataset,
        tokenizer,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        tokenizer,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = max(len(train_loader) * config.epochs, 1)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history: list[dict[str, float]] = []
    best_metric = float("-inf")
    best_checkpoint = checkpoint_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config.max_grad_norm,
        )
        validation_metrics = evaluate_epoch(model, validation_loader, device)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(validation_metrics["loss"]),
            "val_accuracy": float(validation_metrics["accuracy"]),
            "val_f1_macro": float(validation_metrics["f1_macro"]),
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        if epoch_metrics["val_f1_macro"] > best_metric:
            best_metric = epoch_metrics["val_f1_macro"]
            torch.save(model.state_dict(), best_checkpoint)
            tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

    (output_dir / "training_history.json").write_text(
        json.dumps(
            {"config": asdict(config), "history": history},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return history, best_checkpoint
