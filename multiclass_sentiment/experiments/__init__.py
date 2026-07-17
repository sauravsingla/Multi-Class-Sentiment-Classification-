"""Paper model implementations M1-M5."""

MODEL_NAMES = {
    "m1": "TF-IDF + TruncatedSVD + Bidirectional GRU",
    "m2": "Word2Vec + Random Forest",
    "m3": "GloVe + LSTM",
    "m4": "GloVe + Bidirectional LSTM",
    "m5": "BERT fine-tuning",
}
