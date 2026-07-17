"""TensorFlow implementations for paper models M1, M3 and M4."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from multiclass_sentiment.data import LABELS

from .common import evaluate, paper_split, save_metrics, set_seed


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install TensorFlow with: pip install -e '.[deep-learning]'") from exc
    return tf


def _glove_matrix(word_index, glove_path: str | Path, vocab_size: int, dimension: int = 100):
    embeddings = {}
    with Path(glove_path).open(encoding="utf-8") as handle:
        for line in handle:
            values = line.rstrip().split()
            if len(values) != dimension + 1:
                continue
            embeddings[values[0]] = np.asarray(values[1:], dtype=np.float32)
    matrix = np.zeros((vocab_size, dimension), dtype=np.float32)
    for word, index in word_index.items():
        if index < vocab_size and word in embeddings:
            matrix[index] = embeddings[word]
    return matrix


def train_m1(data, output_dir, seed=42, epochs=20):
    """M1: TF-IDF features reduced with SVD and classified by BiGRU."""
    tf = _tensorflow()
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    x_train, x_test, y_train, y_test = paper_split(data, seed)

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    train_tfidf = vectorizer.fit_transform(x_train)
    test_tfidf = vectorizer.transform(x_test)

    from sklearn.decomposition import TruncatedSVD

    components = min(500, train_tfidf.shape[1] - 1)
    reducer = TruncatedSVD(n_components=components, random_state=seed)
    train_features = reducer.fit_transform(train_tfidf)[..., None]
    test_features = reducer.transform(test_tfidf)[..., None]

    encoder = LabelEncoder().fit(list(LABELS))
    train_labels = encoder.transform(y_train)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((components, 1)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(48)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_features, train_labels, validation_split=0.1, epochs=epochs, batch_size=64, verbose=2)
    predictions = encoder.inverse_transform(np.argmax(model.predict(test_features, verbose=0), axis=1))
    metrics = evaluate(y_test, predictions)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save(target / "m1_bigru.keras")
    save_metrics(metrics, target, "m1")
    return metrics


def train_glove(model_id, data, glove_path, output_dir, seed=42, epochs=20):
    """Train M3 LSTM or M4 Bidirectional LSTM using 100d GloVe."""
    if model_id not in {"m3", "m4"}:
        raise ValueError("model_id must be m3 or m4")
    tf = _tensorflow()
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    x_train, x_test, y_train, y_test = paper_split(data, seed)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)
    train_seq = tf.keras.utils.pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=48)
    test_seq = tf.keras.utils.pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=48)
    vocab_size = min(20000, len(tokenizer.word_index) + 1)
    matrix = _glove_matrix(tokenizer.word_index, glove_path, vocab_size)

    encoder = LabelEncoder().fit(list(LABELS))
    train_labels = encoder.transform(y_train)
    recurrent = tf.keras.layers.LSTM(48)
    if model_id == "m4":
        recurrent = tf.keras.layers.Bidirectional(recurrent)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, 100, weights=[matrix], trainable=False),
            recurrent,
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_seq, train_labels, validation_split=0.1, epochs=epochs, batch_size=64, verbose=2)
    predictions = encoder.inverse_transform(np.argmax(model.predict(test_seq, verbose=0), axis=1))
    metrics = evaluate(y_test, predictions)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save(target / f"{model_id}.keras")
    save_metrics(metrics, target, model_id)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["m1", "m3", "m4"])
    parser.add_argument("--data", default="text_emotion.csv")
    parser.add_argument("--glove-path")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    target = Path(args.output_dir) / args.model
    if args.model == "m1":
        train_m1(args.data, target, args.seed, args.epochs)
    else:
        if not args.glove_path:
            parser.error("--glove-path is required for m3 and m4")
        train_glove(args.model, args.data, args.glove_path, target, args.seed, args.epochs)


if __name__ == "__main__":
    main()
