"""Dataset loading and preprocessing for the CrowdFlower emotion data.

The default behaviour mirrors the 2020 paper: retain the 11 reported emotion
classes, preserve repeated tweets, expand common contractions, remove Twitter
handles/URLs, and optionally apply lemmatisation and stop-word removal.
"""

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
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
PAPER_CLASS_COUNTS = {
    "sadness": 5165,
    "boredom": 179,
    "neutral": 8638,
    "worry": 8459,
    "surprise": 2187,
    "love": 3842,
    "fun": 1776,
    "hate": 1323,
    "happiness": 5209,
    "anger": 110,
    "relief": 1526,
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z\s]")
_SPACE_RE = re.compile(r"\s+")


def clean_tweet(text: str) -> str:
    """Apply the common cleaning stage described in the paper."""
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
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "that's": "that is",
        "what's": "what is",
        "where's": "where is",
    }
    for source, target in contractions.items():
        value = value.replace(source, target)
    value = _URL_RE.sub(" ", value)
    value = _MENTION_RE.sub(" ", value)
    value = _NON_ALNUM_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def paper_preprocess(text: str) -> str:
    """Run the paper's cleaning, stop-word removal and lemmatisation stages.

    NLTK data is imported lazily so the basic package remains lightweight.
    Install ``multiclass-sentiment[paper]`` and download ``stopwords`` and
    ``wordnet`` before running the full reproduction pipeline.
    """
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'paper' extra to use paper_preprocess") from exc

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(token)
        for token in clean_tweet(text).split()
        if len(token) > 2 and token not in stop_words
    ]
    return " ".join(tokens)


def load_emotion_data(
    path: str | Path = "text_emotion.csv",
    *,
    preprocess: bool = False,
    deduplicate: bool = False,
) -> pd.DataFrame:
    """Load the 11-class dataset used in the paper.

    ``empty`` and ``enthusiasm`` are removed. Duplicates are retained by
    default because removing them changes the published class distribution.
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
    frame = frame.dropna()
    if deduplicate:
        frame = frame.drop_duplicates()
    cleaner = paper_preprocess if preprocess else clean_tweet
    frame["content"] = frame["content"].map(cleaner)
    frame = frame.loc[frame["content"].str.len() > 0].reset_index(drop=True)
    frame["target"] = frame["sentiment"].map(LABEL_TO_ID).astype("int64")
    frame["sentiment"] = pd.Categorical(frame["sentiment"], categories=LABELS)
    return frame


def validate_paper_distribution(frame: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Return any differences from Table 1 of the paper."""
    observed = frame["sentiment"].value_counts().to_dict()
    return {
        label: (PAPER_CLASS_COUNTS[label], int(observed.get(label, 0)))
        for label in LABELS
        if int(observed.get(label, 0)) != PAPER_CLASS_COUNTS[label]
    }
