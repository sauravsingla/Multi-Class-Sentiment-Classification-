"""Model construction and evaluation utilities."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


def build_random_forest_pipeline(random_state: int = 42) -> Pipeline:
    """Create the paper-aligned TF-IDF + Random Forest baseline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=30_000,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(model: Any, texts: list[str], labels: list[str]) -> dict[str, Any]:
    """Return accuracy, macro/weighted F1 and a per-class report."""
    predictions = model.predict(texts)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            labels, predictions, output_dict=True, zero_division=0
        ),
    }
