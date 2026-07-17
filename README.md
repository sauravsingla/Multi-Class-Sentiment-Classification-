# Multi-Class Sentiment Classification

[![CI](https://github.com/sauravsingla/Multi-Class-Sentiment-Classification-/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/Multi-Class-Sentiment-Classification-/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.26438%2Fijcse%2Fv8i11.1420-blue)](https://doi.org/10.26438/ijcse/v8i11.1420)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible code accompanying the paper **“Multi-Class Sentiment Classification using Machine Learning and Deep Learning Techniques”**, published in the *International Journal of Computer Sciences and Engineering*, Volume 8, Issue 11, pages 14–20, November 2020.

The study compares five machine-learning and deep-learning approaches for fine-grained Twitter emotion classification on the CrowdFlower `text_emotion` dataset. The maintained implementation uses the same 11 classes, the paper preprocessing pipeline, and one deterministic 70:30 stratified split for all models.

## Paper models

| ID | Method | Maintained implementation |
|---|---|---|
| M1 | TF-IDF → dimensionality reduction → Bidirectional GRU | `experiments/keras_models.py` |
| M2 | Word2Vec → Random Forest | `experiments/m2_word2vec_rf.py` |
| M3 | 100-dimensional GloVe → LSTM | `experiments/keras_models.py` |
| M4 | 100-dimensional GloVe → Bidirectional LSTM | `experiments/keras_models.py` |
| M5 | Fine-tuned BERT | `experiments/m5_bert.py` |

The original `twitter_bilstm.ipynb` and `twitter_bert.ipynb` notebooks remain available as historical research provenance. The maintained scripts remove Google Drive paths, notebook-only state and hidden manual steps.

## Repository structure

- `multiclass_sentiment/data.py` — paper labels, preprocessing and dataset validation
- `multiclass_sentiment/experiments/` — executable M1–M5 implementations
- `docs/PAPER_REPRODUCTION.md` — methodology and reproducibility protocol
- `tests/` — automated validation tests
- `.github/workflows/ci.yml` — linting and tests on Python 3.10–3.12
- `text_emotion.csv` — CrowdFlower emotion dataset used in the paper

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev,paper]'
python -m nltk.downloader stopwords wordnet omw-1.4
pytest
```

For neural models:

```bash
pip install -e '.[deep-learning]'  # M1, M3 and M4
pip install -e '.[bert]'           # M5
```

## Run the paper experiments

All commands default to `text_emotion.csv`, seed `42`, and the common 70:30 stratified split.

### M1 — TF-IDF + Bidirectional GRU

```bash
python -m multiclass_sentiment.experiments.keras_models m1 \
  --data text_emotion.csv \
  --output-dir artifacts \
  --epochs 20
```

### M2 — Word2Vec + Random Forest

```bash
python -m multiclass_sentiment.experiments.m2_word2vec_rf \
  --data text_emotion.csv \
  --output-dir artifacts/m2
```

### M3 — GloVe + LSTM

Download `glove.6B.100d.txt`, then run:

```bash
python -m multiclass_sentiment.experiments.keras_models m3 \
  --data text_emotion.csv \
  --glove-path /path/to/glove.6B.100d.txt \
  --output-dir artifacts \
  --epochs 20
```

### M4 — GloVe + Bidirectional LSTM

```bash
python -m multiclass_sentiment.experiments.keras_models m4 \
  --data text_emotion.csv \
  --glove-path /path/to/glove.6B.100d.txt \
  --output-dir artifacts \
  --epochs 20
```

### M5 — BERT

```bash
python -m multiclass_sentiment.experiments.m5_bert \
  --data text_emotion.csv \
  --output-dir artifacts/m5 \
  --epochs 3
```

Each experiment saves model artifacts and a JSON report containing accuracy, macro F1, weighted F1 and precision/recall/F1 for every paper class.

## Reproducibility rules

- The class order is fixed to sadness, boredom, neutral, worry, surprise, love, fun, hate, happiness, anger and relief.
- `empty` and `enthusiasm` are excluded, matching the paper experiments.
- Every model uses the same stratified 70:30 split and random seed.
- Preprocessing removes Twitter mentions, URLs, punctuation and stop words, expands contractions and applies WordNet lemmatisation.
- The dataset is imbalanced, so macro F1 and per-class metrics must be reported alongside accuracy.
- Newly generated results must be labelled as reproduced results rather than copied paper results.

A lightweight TF-IDF + Random Forest command remains available as an engineering smoke test:

```bash
sentiment-train --data text_emotion.csv --output-dir artifacts/baseline --seed 42
```

It is not one of the paper's five M1–M5 methodologies.

## Citation

```bibtex
@article{singla2020multiclass,
  title   = {Multi-Class Sentiment Classification using Machine Learning and Deep Learning Techniques},
  author  = {Singla, Saurav and Kumar, Vikash},
  journal = {International Journal of Computer Sciences and Engineering},
  volume  = {8},
  number  = {11},
  pages   = {14--20},
  year    = {2020},
  month   = {November},
  doi     = {10.26438/ijcse/v8i11.1420}
}
```

## License

Code in this repository is available under the MIT License. The published article and dataset remain subject to their respective licences and terms.
