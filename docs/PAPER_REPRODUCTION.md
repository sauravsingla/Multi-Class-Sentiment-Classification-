# Paper reproduction protocol

This document maps the repository to the five methodologies in the paper
**Multi-Class Sentiment Classification using Machine Learning and Deep Learning
Techniques** (IJCSE 8(11), 14–20, 2020; DOI: 10.26438/ijcse/v8i11.1420).

## Shared experimental protocol

1. Load `text_emotion.csv` from the CrowdFlower Twitter emotion dataset.
2. Remove the original `empty` and `enthusiasm` labels.
3. Retain the 11 classes in the exact order used by the paper:
   sadness, boredom, neutral, worry, surprise, love, fun, hate, happiness,
   anger and relief.
4. Expand abbreviations/contractions and remove handles, URLs and punctuation.
5. Lemmatise tokens and remove English stop words.
6. Use a 70:30 train/test split. Use a recorded random seed for a reproducible
   rerun; the original article did not report its seed.
7. Report per-class precision, recall and F1 plus overall accuracy.

The published class distribution contains 38,414 examples. Do not silently
remove duplicate tweets when reproducing the paper because that changes Table 1.

## Model mapping

| ID | Vectorisation | Classifier described in the paper |
|---|---|---|
| M1 | TF-IDF followed by Truncated SVD | Bidirectional GRU, dropout, global-max pooling and dense layers |
| M2 | Word2Vec | Random Forest |
| M3 | 100-dimensional GloVe | LSTM with embedding, dropout, global-max pooling and dense layers |
| M4 | 100-dimensional GloVe | Bidirectional LSTM with embedding, dropout, global-max pooling and dense layers |
| M5 | BERT contextual representation | Fine-tuned BERT classifier |

## Reproduction integrity

The original notebooks are historical experimental artefacts. They contain
Google Drive paths and library APIs from the 2020 Colab environment. A modern
rerun may not exactly equal the article unless the original package versions,
random split, embeddings and hyperparameters are reconstructed.

For this reason:

- `results/paper_reported_metrics.csv` records the values printed in the paper.
- newly generated metrics must be written separately under `artifacts/`;
- reproduced results must never overwrite or be presented as the published
  results;
- all runs should record the seed, package versions and model configuration.

## Published finding

The article reports M5 (BERT) as the strongest model with 40% overall accuracy,
followed by M4 (Bidirectional LSTM). The low performance on boredom and anger is
consistent with the severe class imbalance shown in Table 1.
