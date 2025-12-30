"""Auditory frontend that clusters acoustic frames into proto-phoneme IDs.

The module provides a lightweight streaming interface for turning spectral
representations (e.g., log-Mel or MFCC frames) into discrete identifiers that
can be consumed by :class:`AuditoryLexiconLearner` without manual phoneme
labels. It relies on a simple online k-means learner with optional
autoencoder-style reconstruction scoring when the dependency is available.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from sklearn.neural_network import MLPRegressor
except Exception:  # pragma: no cover - scikit-learn may be unavailable
    MLPRegressor = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass
class AuditoryFrontendConfig:
    """Configuration for the streaming auditory frontend."""

    buffer_size: int = 24
    cluster_count: int = 16
    min_activation: float = 0.0
    use_autoencoder: bool = False
    learning_rate: float = 0.15


class ProtoPhonemeClusterer:
    """Online k-means style clusterer with optional autoencoder scoring."""

    def __init__(self, config: AuditoryFrontendConfig) -> None:
        self.config = config
        self.centroids: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None
        self.autoencoder: Optional["MLPRegressor"] = None

        if self.config.use_autoencoder and MLPRegressor is not None:
            # Small autoencoder to capture coarse structure; keeps the same
            # interface even if the dependency is missing.
            self.autoencoder = MLPRegressor(
                hidden_layer_sizes=(64,),
                activation="relu",
                max_iter=25,
                warm_start=True,
            )

    def _ensure_initialized(self, frame: np.ndarray) -> None:
        if self.centroids is not None:
            return
        dim = frame.shape[-1]
        self.centroids = np.random.randn(self.config.cluster_count, dim).astype(np.float32)
        self.counts = np.zeros(self.config.cluster_count, dtype=np.float32)

    def partial_fit(self, frames: Sequence[np.ndarray]) -> None:
        if not frames:
            return
        self._ensure_initialized(frames[0])
        assert self.centroids is not None and self.counts is not None

        for frame in frames:
            distances = self._distance(frame, self.centroids)
            cid = int(np.argmin(distances))
            self.counts[cid] += 1.0
            lr = self.config.learning_rate / max(1.0, self.counts[cid])
            self.centroids[cid] = (1 - lr) * self.centroids[cid] + lr * frame

        if self.autoencoder is not None:
            try:  # pragma: no cover - depends on sklearn
                X = np.stack(frames)
                self.autoencoder.partial_fit(X, X)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Autoencoder partial fit failed: %s", exc)
                self.autoencoder = None

    def encode(self, frames: Sequence[np.ndarray]) -> List[int]:
        if not frames:
            return []
        self._ensure_initialized(frames[0])
        assert self.centroids is not None

        proto_ids: List[int] = []
        for frame in frames:
            distances = self._distance(frame, self.centroids)
            cid = int(np.argmin(distances))

            if self.autoencoder is not None:
                try:  # pragma: no cover - depends on sklearn
                    recon = self.autoencoder.predict([frame])[0]
                    score = float(np.mean(np.square(recon - frame)))
                    if score > 1.5:  # reconstruction too poor; skip activation
                        continue
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Autoencoder inference failed: %s", exc)
                    self.autoencoder = None
            proto_ids.append(cid)
        return proto_ids

    @staticmethod
    def _distance(frame: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        diff = centroids - frame.reshape(1, -1)
        return np.sum(diff * diff, axis=1)


class AuditoryStream:
    """Streaming buffer that filters and clusters incoming acoustic frames."""

    def __init__(self, clusterer: ProtoPhonemeClusterer, config: AuditoryFrontendConfig) -> None:
        self.clusterer = clusterer
        self.config = config
        self.buffer: Deque[np.ndarray] = deque(maxlen=config.buffer_size)

    def push(self, frames: Iterable[Sequence[float]]) -> List[int]:
        for frame in frames:
            np_frame = np.asarray(frame, dtype=np.float32).flatten()
            if not np_frame.size:
                continue
            energy = float(np.mean(np.abs(np_frame)))
            if energy < self.config.min_activation:
                continue
            self.buffer.append(np_frame)

        if not self.buffer:
            return []

        buffered = list(self.buffer)
        self.buffer.clear()

        proto_ids = self.clusterer.encode(buffered)
        self.clusterer.partial_fit(buffered)
        return proto_ids


class AuditoryFrontend:
    """Facade that emits proto-phoneme IDs and updates the lexicon learner."""

    def __init__(
        self,
        word_recognizer: Any,
        semantic_network: Any,
        lexicon_learner: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = AuditoryFrontendConfig(**(config or {}))
        self.config = cfg
        self.clusterer = ProtoPhonemeClusterer(cfg)
        self.stream = AuditoryStream(self.clusterer, cfg)
        self.lexicon_learner = lexicon_learner
        self.word_recognizer = word_recognizer
        self.semantic_network = semantic_network

    def stream_frames(
        self, frames: Iterable[Sequence[float]], *, context_tokens: Optional[Sequence[str]] = None
    ) -> List[int]:
        proto_ids = self.stream.push(frames)
        if proto_ids and self.lexicon_learner is not None:
            self.lexicon_learner.ingest_phonemes([str(pid) for pid in proto_ids], context_tokens or [])
        return proto_ids

