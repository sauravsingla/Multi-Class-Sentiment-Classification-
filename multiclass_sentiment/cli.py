"""Command-line training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from .data import load_emotion_data
from .modeling import build_random_forest_pipeline, evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the paper-aligned sentiment baseline")
    parser.add_argument("--data", default="text_emotion.csv", help="Path to CrowdFlower CSV")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model and metrics")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1")

    frame = load_emotion_data(args.data)
    train, test = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=frame["sentiment"],
    )

    model = build_random_forest_pipeline(random_state=args.seed)
    model.fit(train["content"].tolist(), train["sentiment"].astype(str).tolist())
    metrics = evaluate_model(
        model,
        test["content"].tolist(),
        test["sentiment"].astype(str).tolist(),
    )
    metrics["train_rows"] = len(train)
    metrics["test_rows"] = len(test)
    metrics["seed"] = args.seed

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "random_forest.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({key: metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1")}, indent=2))


if __name__ == "__main__":
    main()
