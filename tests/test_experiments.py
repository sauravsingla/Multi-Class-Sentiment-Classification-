import json

from multiclass_sentiment.data import LABELS
from multiclass_sentiment.experiments import MODEL_NAMES
from multiclass_sentiment.experiments.common import evaluate, save_metrics


def test_all_five_paper_models_are_registered() -> None:
    assert tuple(MODEL_NAMES) == ("m1", "m2", "m3", "m4", "m5")


def test_evaluation_reports_every_paper_class() -> None:
    metrics = evaluate(["love", "worry"], ["love", "neutral"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert all(label in metrics["classification_report"] for label in LABELS)


def test_metrics_are_written_as_json(tmp_path) -> None:
    path = save_metrics({"accuracy": 0.5}, tmp_path, "m1")
    assert json.loads(path.read_text(encoding="utf-8")) == {"accuracy": 0.5}
