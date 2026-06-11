from __future__ import annotations

import pytest
from datasets import Dataset
from sklearn.pipeline import FeatureUnion

from src.baselines.tfidf_logreg import (
    BaselineConfig,
    build_pipeline,
    dataset_to_xy,
    id2label_for_mode,
)


def test_build_pipeline_uses_word_char_features() -> None:
    config = BaselineConfig(feature_mode="word_char")

    pipeline = build_pipeline(config)

    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(features, FeatureUnion)
    assert [name for name, _ in features.transformer_list] == ["word", "char"]
    assert classifier.C == config.regularization_c


def test_build_pipeline_rejects_unknown_feature_mode() -> None:
    config = BaselineConfig(feature_mode="unknown")

    with pytest.raises(ValueError, match="feature_mode"):
        build_pipeline(config)


def test_dataset_to_xy_can_build_binary_polarity_task() -> None:
    dataset = Dataset.from_dict(
        {
            "text": ["bad app", "ok app", "great app"],
            "label": [0, 1, 2],
        }
    )

    texts, labels = dataset_to_xy(dataset, "text", "label", "binary_polarity")

    assert texts == ["bad app", "great app"]
    assert labels == [0, 1]
    assert id2label_for_mode("binary_polarity") == {0: "negative", 1: "positive"}
