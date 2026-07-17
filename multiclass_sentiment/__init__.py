"""Utilities for reproducible multi-class emotion classification experiments."""

from .data import LABELS, load_emotion_data
from .modeling import build_random_forest_pipeline, evaluate_model

__all__ = [
    "LABELS",
    "build_random_forest_pipeline",
    "evaluate_model",
    "load_emotion_data",
]

__version__ = "1.0.0"
