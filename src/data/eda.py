from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict

from src.data.dataset import clean_text


def split_to_frame(split: Dataset, clean: bool = False) -> pd.DataFrame:
    frame = split.to_pandas()
    if clean:
        frame = frame.copy()
        frame["text"] = frame["text"].map(clean_text)
    return frame


def dataset_to_frames(
    dataset: DatasetDict, clean: bool = False
) -> dict[str, pd.DataFrame]:
    return {
        split_name: split_to_frame(split, clean=clean)
        for split_name, split in dataset.items()
    }


def class_distribution(
    frame: pd.DataFrame, label_column: str = "label_text"
) -> pd.DataFrame:
    counts = frame[label_column].value_counts(dropna=False).rename_axis(label_column)
    distribution = counts.reset_index(name="count")
    distribution["share"] = distribution["count"] / distribution["count"].sum()
    return distribution


def text_length_summary(frame: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    lengths = frame[text_column].fillna("").astype(str).str.len()
    summary = lengths.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_frame(
        name="length"
    )
    return summary


def duplicated_texts(frame: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    duplicated = frame[frame.duplicated(subset=[text_column], keep=False)].copy()
    return duplicated.sort_values(text_column)


def empty_texts(frame: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    mask = frame[text_column].fillna("").astype(str).str.strip().eq("")
    return frame[mask].copy()


def sample_examples(
    frame: pd.DataFrame,
    *,
    label_value: str | int | None = None,
    label_column: str = "label_text",
    size: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    filtered = (
        frame if label_value is None else frame[frame[label_column] == label_value]
    )
    if filtered.empty:
        return filtered
    size = min(size, len(filtered))
    return filtered.sample(n=size, random_state=random_state)


def dataset_report(dataset: DatasetDict) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for split_name, split in dataset.items():
        frame = split_to_frame(split, clean=True)
        report[split_name] = {
            "rows": len(frame),
            "class_distribution": class_distribution(frame).to_dict(orient="records"),
            "length_summary": text_length_summary(frame)["length"].round(2).to_dict(),
            "empty_texts": int(empty_texts(frame).shape[0]),
            "duplicated_texts": int(duplicated_texts(frame).shape[0]),
        }
    return report
