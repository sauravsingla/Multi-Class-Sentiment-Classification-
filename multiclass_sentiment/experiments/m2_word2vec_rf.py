"""M2: Word2Vec document vectors followed by Random Forest."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from .common import evaluate, paper_split, save_metrics, set_seed


def _tokens(texts):
    return [str(text).split() for text in texts]


def _document_vectors(model: Word2Vec, tokenized, vector_size: int) -> np.ndarray:
    rows = []
    for tokens in tokenized:
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        rows.append(np.mean(vectors, axis=0) if vectors else np.zeros(vector_size, dtype=np.float32))
    return np.asarray(rows)


def train(data: str | Path, output_dir: str | Path, seed: int = 42) -> dict:
    set_seed(seed)
    x_train, x_test, y_train, y_test = paper_split(data, seed)
    train_tokens = _tokens(x_train)
    test_tokens = _tokens(x_test)

    embedding = Word2Vec(
        sentences=train_tokens,
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        seed=seed,
        epochs=20,
    )
    train_vectors = _document_vectors(embedding, train_tokens, 100)
    test_vectors = _document_vectors(embedding, test_tokens, 100)

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    classifier.fit(train_vectors, y_train)
    metrics = evaluate(y_test, classifier.predict(test_vectors))

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    embedding.save(str(target / "m2_word2vec.model"))
    joblib.dump(classifier, target / "m2_random_forest.joblib")
    save_metrics(metrics, target, "m2")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="text_emotion.csv")
    parser.add_argument("--output-dir", default="artifacts/m2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
