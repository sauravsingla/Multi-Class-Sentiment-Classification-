"""Dataset loading and text normalization for the CrowdFlower emotion data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

LABELS = (
    "sadness",
    "boredom",
    "neutral",
    "worry",
    "surprise",
    "love",
    "fun",
    "hate",
    "happiness",
    "anger",
    "relief",
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
_SPACE_RE = re.compile(r"\s+")


def clean_tweet(text: str) -> str:
    """Apply conservative normalization while retaining sentiment-bearing tokens."""
    value = str(text).lower().replace("’", "'")
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
    }
    for source, target in contractions.items():
        value = value.replace(source, target)
    value = _URL_RE.sub(" URL ", value)
    value = _MENTION_RE.sub(" USER ", value)
    return _SPACE_RE.sub(" ", value).strip()


def load_emotion_data(path: str | Path = "text_emotion.csv") -> pd.DataFrame:
    """Load the repository dataset and return the 11 classes used in the paper.

    The original CrowdFlower file contains 13 labels. Following the paper's
    experiments, ``empty`` and ``enthusiasm`` are removed.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")

    frame = pd.read_csv(source)
    required = {"sentiment", "content"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.loc[frame["sentiment"].isin(LABELS), ["content", "sentiment"]].copy()
    frame = frame.dropna().drop_duplicates()
    frame["content"] = frame["content"].map(clean_tweet)
    frame = frame.loc[frame["content"].str.len() > 0].reset_index(drop=True)
    frame["sentiment"] = pd.Categorical(frame["sentiment"], categories=LABELS)
    return frame
