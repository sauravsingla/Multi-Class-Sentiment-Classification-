"""Shared, deterministic data splitting and metric persistence."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from multiclass_sentiment.data import LABELS, load_emotion_data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def paper_split(data_path: str | Path, seed: int = 42):
    """Return the common 70:30 stratified split used by every model."""
    frame = load_emotion_data(data_path, preprocess=True)
    return train_test_split(
        frame["content"].astype(str).to_numpy(),
        frame["sentiment"].astype(str).to_numpy(),
        test_size=0.30,
        random_state=seed,
        stratify=frame["sentiment"].astype(str),
    )


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
    }


def save_metrics(metrics: dict, output_dir: str | Path, model_name: str) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{model_name}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return path
