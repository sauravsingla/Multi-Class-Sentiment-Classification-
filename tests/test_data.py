from pathlib import Path

import pandas as pd
import pytest

from multiclass_sentiment.data import (
    LABELS,
    clean_tweet,
    load_emotion_data,
    validate_paper_distribution,
)


def test_clean_tweet_normalizes_social_tokens() -> None:
    cleaned = clean_tweet("@Sam I'm HAPPY! See https://example.com")
    assert cleaned == "i am happy see"


def test_loader_keeps_paper_labels_and_targets(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "tweet_id": [1, 2, 3],
            "author": ["a", "b", "c"],
            "sentiment": ["love", "empty", "worry"],
            "content": ["Great", "", "Oh no"],
        }
    ).to_csv(source, index=False)

    frame = load_emotion_data(source)
    assert frame["sentiment"].astype(str).tolist() == ["love", "worry"]
    assert frame["target"].tolist() == [5, 3]
    assert tuple(frame["sentiment"].cat.categories) == LABELS


def test_loader_preserves_duplicates_by_default(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "sentiment": ["love", "love"],
            "content": ["same tweet", "same tweet"],
        }
    ).to_csv(source, index=False)

    assert len(load_emotion_data(source)) == 2
    assert len(load_emotion_data(source, deduplicate=True)) == 1


def test_distribution_validator_reports_mismatches(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    pd.DataFrame({"sentiment": ["love"], "content": ["hello"]}).to_csv(source, index=False)
    differences = validate_paper_distribution(load_emotion_data(source))
    assert differences["love"] == (3842, 1)
    assert differences["anger"] == (110, 0)


def test_loader_rejects_invalid_schema(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="required columns"):
        load_emotion_data(source)
