from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence, List

import numpy as np


class FeatureExtractor:
    """Transform runtime metrics into model-ready feature arrays."""

    def extract(self, metrics: Dict[str, Any]) -> np.ndarray:
        cpu = metrics.get("cpu_percent") or 0.0
        mem = metrics.get("memory_percent") or 0.0
        return np.array([cpu, mem], dtype=float)


class PolicyModel:
    """Simple linear policy loaded from a serialized JSON file."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path or Path(__file__).with_name("policy_model.json"))
        with open(self.model_path) as f:
            data = json.load(f)
        self.weights = np.array(data["weights"], dtype=float)
        self.biases = np.array(data["biases"], dtype=float)
        self.suggestions: List[str] = data["suggestions"]
        self.default: str = data.get("default", "")

    def predict(self, features: Sequence[float]) -> List[str]:
        feats = np.asarray(features, dtype=float)
        scores = self.weights @ feats + self.biases
        suggestions = [msg for score, msg in zip(scores, self.suggestions) if score > 0]
        if not suggestions and self.default:
            suggestions.append(self.default)
        return suggestions
