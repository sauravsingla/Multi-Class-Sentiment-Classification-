# Multi-Class Sentiment Classification

[![CI](https://github.com/sauravsingla/Multi-Class-Sentiment-Classification-/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/Multi-Class-Sentiment-Classification-/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.26438%2Fijcse%2Fv8i11.1420-blue)](https://doi.org/10.26438/ijcse/v8i11.1420)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and reproducibility material accompanying **“Multi-Class Sentiment Classification using Machine Learning and Deep Learning Techniques”**, published in the *International Journal of Computer Sciences and Engineering*, Volume 8, Issue 11, pages 14–20, November 2020.

The paper studies fine-grained Twitter emotion classification on the CrowdFlower `text_emotion` dataset. Unlike polarity-only classification, it predicts 11 emotions: sadness, boredom, neutral, worry, surprise, love, fun, hate, happiness, anger and relief.

## Methods reported in the paper

| ID | Representation | Classifier |
|---|---|---|
| M1 | TF-IDF + Truncated SVD | Bidirectional GRU network |
| M2 | Word2Vec | Random Forest |
| M3 | 100-dimensional GloVe | LSTM network |
| M4 | 100-dimensional GloVe | Bidirectional LSTM network |
| M5 | BERT contextual embeddings | Fine-tuned BERT classifier |

All five methods use the same preprocessing stage: abbreviation/contraction expansion, removal of Twitter noise, lemmatisation and English stop-word removal. The dataset is split 70:30 for training and testing. The paper reports BERT (M5) as the best model with 40% overall accuracy, followed by Bidirectional LSTM (M4).

See [`docs/PAPER_REPRODUCTION.md`](docs/PAPER_REPRODUCTION.md) for the exact model mapping, class counts and reproducibility rules.

## Repository structure

- `multiclass_sentiment/` — reusable dataset, preprocessing, training and evaluation code
- `docs/PAPER_REPRODUCTION.md` — paper-to-code methodology mapping
- `results/` — immutable metrics transcribed from the published paper
- `tests/` — preprocessing and dataset validation tests
- `.github/workflows/ci.yml` — linting and tests on Python 3.10–3.12
- `twitter_bert.ipynb` — original BERT experiment notebook
- `twitter_bilstm.ipynb` — original Bidirectional LSTM experiment notebook
- `text_emotion.csv` — CrowdFlower emotion dataset used in the study

The historical notebooks are preserved for research provenance. They contain 2020-era Colab paths and APIs; the package provides maintainable code without notebook-only state.

## Quick start

```bash
git clone https://github.com/sauravsingla/Multi-Class-Sentiment-Classification-.git
cd Multi-Class-Sentiment-Classification-
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
pytest
```

Install the complete paper preprocessing dependencies:

```bash
pip install -e '.[paper]'
python -m nltk.downloader stopwords wordnet omw-1.4
```

Install a deep-learning environment only when needed:

```bash
pip install -e '.[deep-learning]'  # M1, M3 and M4
pip install -e '.[bert]'           # M5
```

## Reproducible baseline

The maintained command-line baseline is intentionally separate from the five historical methods. It provides a fast smoke test for the complete data/evaluation pipeline:

```bash
sentiment-train --data text_emotion.csv --output-dir artifacts --seed 42
```

It writes `artifacts/random_forest.joblib` and `artifacts/metrics.json`, including accuracy, macro-F1, weighted-F1 and per-class precision/recall/F1. These newly generated values must not be represented as the published M1–M5 results.

## Paper-aligned data handling

The loader removes `empty` and `enthusiasm`, retains the 11 paper labels in their published numeric order and preserves duplicate tweets by default. Duplicate removal is optional because silently removing repeated rows changes the paper's Table 1 distribution of 38,414 samples.

```python
from multiclass_sentiment.data import load_emotion_data, validate_paper_distribution

frame = load_emotion_data("text_emotion.csv", preprocess=False)
differences = validate_paper_distribution(frame)
assert not differences, differences
```

Pass `preprocess=True` to apply the complete lemmatisation and stop-word stage after installing the `paper` extra and NLTK resources.

## Reproducibility boundary

The article did not publish every random seed and environment detail needed for bit-for-bit reproduction. Therefore, this repository distinguishes:

- **paper-reported results**, transcribed under `results/`;
- **historical notebooks**, retained unchanged as experimental provenance;
- **new reproductions**, generated under `artifacts/` with an explicit seed and current dependency versions.

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

Code is available under the MIT License. The article is published under the journal's stated licence. Dataset users must also comply with the dataset provider's terms.
