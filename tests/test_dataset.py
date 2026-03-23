from __future__ import annotations

from datasets import Dataset, DatasetDict

from src.data.dataset import clean_text, prepare_text_classification_dataset


def test_clean_text_removes_html_and_extra_spaces() -> None:
    dirty_text = "  Hello&nbsp;<b>world</b>   from   test  "

    assert clean_text(dirty_text) == "Hello world from test"


def test_prepare_text_classification_dataset_drops_id_and_label_text() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "id": [1],
                    "text": [" <b>great</b> app "],
                    "label": [2],
                    "label_text": ["positive"],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "id": [2],
                    "text": [" bad app "],
                    "label": [0],
                    "label_text": ["negative"],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "id": [3],
                    "text": [" ok app "],
                    "label": [1],
                    "label_text": ["neutral"],
                }
            ),
        }
    )

    prepared = prepare_text_classification_dataset(dataset)

    assert prepared["train"].column_names == ["text", "label"]
    assert prepared["train"][0]["text"] == "great app"


def test_prepare_text_classification_dataset_keeps_label_text_when_requested() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "id": [1],
                    "text": ["nice"],
                    "label": [2],
                    "label_text": ["positive"],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "id": [2],
                    "text": ["fine"],
                    "label": [1],
                    "label_text": ["neutral"],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "id": [3],
                    "text": ["bad"],
                    "label": [0],
                    "label_text": ["negative"],
                }
            ),
        }
    )

    prepared = prepare_text_classification_dataset(dataset, keep_label_text=True)

    assert prepared["train"].column_names == ["text", "label", "label_text"]
