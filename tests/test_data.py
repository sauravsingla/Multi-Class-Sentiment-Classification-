from pathlib import Path

import pandas as pd
import pytest

from multiclass_sentiment.data import LABELS, clean_tweet, load_emotion_data


def test_clean_tweet_normalizes_social_tokens() -> None:
    cleaned = clean_tweet("@Sam I'm HAPPY! See https://example.com")
    assert cleaned == "USER i am happy! see URL"


def test_loader_keeps_paper_labels(tmp_path: Path) -> None:
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
    assert tuple(frame["sentiment"].cat.categories) == LABELS


def test_loader_rejects_invalid_schema(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="required columns"):
        load_emotion_data(source)
