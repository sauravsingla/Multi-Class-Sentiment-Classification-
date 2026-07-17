# Multi-Class Sentiment Classification

[![CI](https://github.com/sauravsingla/Multi-Class-Sentiment-Classification-/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/Multi-Class-Sentiment-Classification-/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.26438%2Fijcse%2Fv8i11.1420-blue)](https://doi.org/10.26438/ijcse/v8i11.1420)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible code accompanying the paper **“Multi-Class Sentiment Classification using Machine Learning and Deep Learning Techniques”**, published in the *International Journal of Computer Sciences and Engineering*, Volume 8, Issue 11, pages 14–20, November 2020.

The study compares classical machine learning and deep-learning approaches for fine-grained Twitter emotion classification. It uses the CrowdFlower `text_emotion` dataset and evaluates labels such as sadness, boredom, neutral, worry, surprise, love, fun, hate, happiness, anger and relief. The paper reports BERT as the strongest approach, followed by bidirectional recurrent models.

## Repository structure

- `multiclass_sentiment/` — reusable, tested Python package
- `tests/` — unit tests for preprocessing and data validation
- `.github/workflows/ci.yml` — automated linting and tests on Python 3.10–3.12
- `twitter_bert.ipynb` — original BERT experiment notebook
- `twitter_bilstm.ipynb` — original BiLSTM experiment notebook
- `text_emotion.csv` — CrowdFlower emotion dataset used by the experiments

The original notebooks are preserved for research provenance. The package provides a clean, reproducible baseline without Google Drive paths or notebook-only state.

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

Train the paper-aligned TF-IDF + Random Forest baseline:

```bash
sentiment-train --data text_emotion.csv --output-dir artifacts --seed 42
```

This writes:

- `artifacts/random_forest.joblib`
- `artifacts/metrics.json`

The metrics file includes accuracy, macro F1, weighted F1 and a full per-class classification report. Macro F1 is included because the dataset is imbalanced and accuracy alone can hide weak minority-class performance.

## Deep-learning environments

Install only the framework required for an experiment:

```bash
pip install -e '.[deep-learning]'  # TensorFlow / BiLSTM
pip install -e '.[bert]'           # PyTorch / Transformers
```

For faithful comparison, use one fixed stratified split, report the random seed, preserve the same 11 labels across models, and compare accuracy together with macro F1 and per-class precision/recall.

## Reproducibility notes

The loader deliberately removes `empty` and `enthusiasm`, matching the 11-class configuration used by the paper’s BERT experiment. It validates the CSV schema, removes missing and duplicate rows, normalizes URLs and mentions, expands common contractions and performs a stratified train/test split.

The repository does **not** claim that newly generated metrics exactly reproduce the historical paper tables unless the same dependency versions, embeddings, split and hardware settings are used. Generated artifacts are intentionally excluded from source control.

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

Code in this repository is available under the MIT License. The published article is distributed by the journal under its stated publication license. Dataset users must also comply with the dataset provider’s terms.
