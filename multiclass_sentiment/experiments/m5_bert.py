"""M5: fine-tune BERT for the paper's 11-class emotion task."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from multiclass_sentiment.data import LABELS

from .common import evaluate, paper_split, save_metrics, set_seed


def train(
    data: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    epochs: int = 3,
    model_name: str = "bert-base-uncased",
) -> dict:
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install BERT dependencies with: pip install -e '.[bert]'") from exc

    set_seed(seed)
    x_train, x_test, y_train, y_test = paper_split(data, seed)
    encoder = LabelEncoder().fit(list(LABELS))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = Dataset.from_dict({"text": x_train.tolist(), "label": encoder.transform(y_train).tolist()})
    test_dataset = Dataset.from_dict({"text": x_test.tolist(), "label": encoder.transform(y_test).tolist()})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["text"])

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(encoder.classes_)},
        label2id={label: index for index, label in enumerate(encoder.classes_)},
    )

    arguments = TrainingArguments(
        output_dir=str(target / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=seed,
        data_seed=seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    predictions = trainer.predict(test_dataset).predictions
    predicted_labels = encoder.inverse_transform(np.argmax(predictions, axis=1))
    metrics = evaluate(y_test, predicted_labels)
    trainer.save_model(str(target / "model"))
    tokenizer.save_pretrained(str(target / "model"))
    save_metrics(metrics, target, "m5")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="text_emotion.csv")
    parser.add_argument("--output-dir", default="artifacts/m5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model-name", default="bert-base-uncased")
    args = parser.parse_args()
    train(args.data, args.output_dir, args.seed, args.epochs, args.model_name)


if __name__ == "__main__":
    main()
