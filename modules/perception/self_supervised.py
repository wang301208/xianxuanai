from __future__ import annotations

"""Lightweight self-supervised learner for sensory streams.

This module introduces predictive coding for individual modalities and a
multimodal contrastive head that aligns co-occurring sensory inputs into a
shared concept embedding.  It deliberately keeps dependencies minimal (NumPy
only) and uses online updates so it can run inside existing perception loops.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def _as_array(values: Any, *, limit: int = 256) -> np.ndarray:
    """Coerce inputs to a 1-D float array with an optional length cap."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    if limit and arr.size > limit:
        arr = arr[:limit]
    return arr.astype(float, copy=False)


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(float, copy=False)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(_normalise(a), _normalise(b)))


@dataclass
class PredictionResult:
    predicted_next: list[float] | None = None
    instant_error: float = 0.0
    ema_error: float = 0.0


@dataclass
class ContrastiveResult:
    pair_losses: Dict[Tuple[str, str], float] = field(default_factory=dict)
    average_loss: float = 0.0
    positives: Dict[Tuple[str, str], float] = field(default_factory=dict)


@dataclass
class ConceptResult:
    key: str
    embedding: list[float]
    count: int
    alignment: float


class _PredictionHead:
    """Tiny online linear predictor updated from prediction error."""

    def __init__(self, *, learning_rate: float = 0.05, ema: float = 0.9, max_dim: int = 256) -> None:
        self.lr = float(learning_rate)
        self.ema = float(ema)
        self.max_dim = int(max_dim)
        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        self._last_input: np.ndarray | None = None
        self._ema_error = 0.0

    def step(self, observation: Sequence[float]) -> PredictionResult:
        vector = _as_array(observation, limit=self.max_dim)
        if vector.size == 0:
            return PredictionResult(None, 0.0, self._ema_error)

        if self._weights is None or self._weights.shape[0] != vector.size:
            # Initialise to an identity-like projection to keep early steps stable.
            self._weights = np.eye(vector.size, dtype=float)
            self._bias = np.zeros(vector.size, dtype=float)
            self._last_input = vector
            self._ema_error = 0.0
            return PredictionResult(predicted_next=None, instant_error=0.0, ema_error=self._ema_error)

        predicted = np.tanh(self._last_input @ self._weights + self._bias)
        error_vec = vector - predicted
        instant_error = float(np.linalg.norm(error_vec))
        self._ema_error = self.ema * self._ema_error + (1.0 - self.ema) * instant_error

        # Simple Hebbian-style update using the outer product of previous input and error.
        grad_w = np.outer(self._last_input, error_vec)
        self._weights += self.lr * grad_w
        self._bias += self.lr * error_vec
        self._last_input = vector

        return PredictionResult(
            predicted_next=predicted.astype(float).tolist(),
            instant_error=instant_error,
            ema_error=self._ema_error,
        )


class _ContrastiveQueue:
    """Maintain recent embeddings per modality for contrastive negatives."""

    def __init__(self, *, capacity: int = 64, temperature: float = 0.1) -> None:
        self.capacity = int(capacity)
        self.temperature = float(max(temperature, 1e-4))
        self._store: MutableMapping[str, list[np.ndarray]] = {}

    def push(self, modality: str, vector: np.ndarray) -> None:
        vector = _normalise(vector)
        bucket = self._store.setdefault(modality, [])
        bucket.append(vector)
        if len(bucket) > self.capacity:
            del bucket[0 : len(bucket) - self.capacity]

    def sample_negatives(self, exclude: Iterable[str]) -> list[np.ndarray]:
        negative_vectors: list[np.ndarray] = []
        excluded = set(exclude)
        for modality, vectors in self._store.items():
            if modality in excluded:
                continue
            negative_vectors.extend(vectors)
        return negative_vectors

    def info_nce(self, anchor: np.ndarray, positive: np.ndarray, negatives: list[np.ndarray]) -> Tuple[float, float]:
        """Return (loss, positive_similarity)."""

        anchor = _normalise(anchor)
        positive = _normalise(positive)
        pos_score = math.exp(float(np.dot(anchor, positive)) / self.temperature)
        if not negatives:
            return float(-math.log(max(pos_score, 1e-8))), float(np.dot(anchor, positive))

        neg_scores = [math.exp(float(np.dot(anchor, neg)) / self.temperature) for neg in negatives]
        denom = pos_score + float(sum(neg_scores))
        loss = -math.log(max(pos_score / denom, 1e-8))
        return float(loss), float(np.dot(anchor, positive))


class SelfSupervisedPerceptionLearner:
    """Coordinate predictive coding and cross-modal contrastive alignment."""

    def __init__(
        self,
        *,
        fusion_engine: Any | None = None,
        prediction_lr: float = 0.05,
        prediction_ema: float = 0.9,
        max_dim: int = 256,
        contrastive_temperature: float = 0.1,
        contrastive_capacity: int = 64,
        concept_ema: float = 0.85,
    ) -> None:
        self._fusion_engine = fusion_engine
        self._predictors: Dict[str, _PredictionHead] = {}
        self._contrastive = _ContrastiveQueue(capacity=contrastive_capacity, temperature=contrastive_temperature)
        self._concepts: Dict[str, Tuple[np.ndarray, int]] = {}
        self._concept_ema = float(concept_ema)
        self._prediction_lr = prediction_lr
        self._prediction_ema = prediction_ema
        self._max_dim = max_dim

    def observe(
        self,
        modalities: Mapping[str, Mapping[str, Any]],
        *,
        fused_embedding: Optional[Sequence[float]] = None,
        annotations: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Update predictive/contrastive heads and return diagnostics."""

        predictions: Dict[str, PredictionResult] = {}
        normalized: Dict[str, np.ndarray] = {}
        annotations = annotations or {}

        for name, payload in modalities.items():
            vector = payload.get("embedding") or payload.get("primary_embedding")
            if vector is None:
                continue
            predictor = self._predictors.setdefault(
                name,
                _PredictionHead(learning_rate=self._prediction_lr, ema=self._prediction_ema, max_dim=self._max_dim),
            )
            result = predictor.step(vector)
            predictions[name] = result
            normalized[name] = _normalise(_as_array(vector, limit=self._max_dim))
            self._contrastive.push(name, normalized[name])

        contrastive = self._compute_contrastive(normalized)
        concept = self._update_concept_embedding(normalized, fused_embedding, annotations)

        return {
            "predictions": predictions,
            "contrastive": contrastive,
            "concept": concept,
        }

    # ------------------------------------------------------------------ #
    def _compute_contrastive(self, normalized: Dict[str, np.ndarray]) -> ContrastiveResult:
        if len(normalized) < 2:
            return ContrastiveResult()

        losses: Dict[Tuple[str, str], float] = {}
        positives: Dict[Tuple[str, str], float] = {}
        negative_pool = self._contrastive.sample_negatives(exclude=normalized.keys())

        names = list(normalized.keys())
        for i, anchor_name in enumerate(names):
            for j in range(i + 1, len(names)):
                other_name = names[j]
                loss, pos_sim = self._contrastive.info_nce(
                    normalized[anchor_name],
                    normalized[other_name],
                    negative_pool,
                )
                losses[(anchor_name, other_name)] = loss
                positives[(anchor_name, other_name)] = pos_sim
                # Symmetric view
                losses[(other_name, anchor_name)] = loss
                positives[(other_name, anchor_name)] = pos_sim

        average = float(np.mean(list(losses.values()))) if losses else 0.0
        return ContrastiveResult(pair_losses=losses, average_loss=average, positives=positives)

    def _update_concept_embedding(
        self,
        normalized: Dict[str, np.ndarray],
        fused_embedding: Optional[Sequence[float]],
        annotations: Mapping[str, Mapping[str, Any]],
    ) -> ConceptResult | None:
        if not normalized:
            return None

        fused_vector = None
        if fused_embedding is not None:
            fused_vector = _normalise(_as_array(fused_embedding, limit=self._max_dim))
        elif self._fusion_engine is not None and len(normalized) > 1:
            try:
                fusion_inputs = {name: vec for name, vec in normalized.items()}
                fused_vector = _normalise(
                    _as_array(self._fusion_engine.fuse_sensory_modalities(**fusion_inputs), limit=self._max_dim)
                )
            except Exception:
                fused_vector = None
        elif len(normalized) > 1:
            fused_vector = _normalise(np.mean(list(normalized.values()), axis=0))

        if fused_vector is None:
            return None

        concept_key = self._derive_concept_key(annotations, normalized)
        stored, count = self._concepts.get(concept_key, (fused_vector, 0))
        updated = self._concept_ema * stored + (1.0 - self._concept_ema) * fused_vector
        updated = _normalise(updated)
        self._concepts[concept_key] = (updated, count + 1)
        alignment = _cosine(fused_vector, updated)

        return ConceptResult(
            key=concept_key,
            embedding=updated.astype(float).tolist(),
            count=count + 1,
            alignment=alignment,
        )

    @staticmethod
    def _derive_concept_key(
        annotations: Mapping[str, Mapping[str, Any]], normalized: Mapping[str, np.ndarray]
    ) -> str:
        tokens: list[str] = []
        for modality, meta in annotations.items():
            labels = meta.get("labels") if isinstance(meta, dict) else None
            if isinstance(labels, (list, tuple)):
                tokens.extend(str(label) for label in labels if label)
            elif isinstance(labels, str):
                tokens.append(labels)
            transcript = meta.get("transcript") if isinstance(meta, dict) else None
            if transcript:
                tokens.append(str(transcript))
        if tokens:
            return "|".join(sorted(set(tokens)))
        return "|".join(sorted(normalized.keys()))


__all__ = ["SelfSupervisedPerceptionLearner", "PredictionResult", "ContrastiveResult", "ConceptResult"]
