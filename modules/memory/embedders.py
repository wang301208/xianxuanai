from __future__ import annotations

import hashlib
import re
from typing import Iterable, Protocol

try:  # optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - dependency is optional
    SentenceTransformer = None  # type: ignore[assignment]

import numpy as np

TOKEN_PATTERN = re.compile(r"\w+")


class HashingEmbedder:
    """Deterministic, dependency-free text embedder using hashing tricks.

    This embedder is not a substitute for high-quality transformer embeddings,
    but it provides a consistent fallback so the vector memory stack works
    without external services. The output vectors are L2-normalised so they can
    be used with cosine/inner-product similarity.
    """

    def __init__(self, dimension: int = 384, window_size: int = 2):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.window_size = max(1, window_size)

    def __call__(self, text: str) -> np.ndarray:
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32)

        vector = np.zeros(self.dimension, dtype=np.float32)
        for i in range(len(tokens)):
            window_tokens = tokens[i : i + self.window_size]
            if not window_tokens:
                continue
            ngram = " ".join(window_tokens).encode("utf-8")
            digest = hashlib.sha256(ngram).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            vector[idx] += 1.0

        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def encode_batch(self, texts: Iterable[str]) -> np.ndarray:
        vectors = [self(text) for text in texts]
        if not vectors:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack(vectors)


class TransformerEmbedder:
    """Wrapper around ``sentence_transformers`` with graceful fallback."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        normalize: bool = True,
        batch_size: int | None = None,
    ) -> None:
        if SentenceTransformer is None:  # pragma: no cover - exercised in integration tests
            raise RuntimeError(
                "sentence-transformers package is required for TransformerEmbedder"
            )

        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        self._batch_size = batch_size

        dimension = getattr(self._model, "get_sentence_embedding_dimension", None)
        if callable(dimension):
            self.dimension = int(dimension())
        else:
            sample = np.asarray(self._model.encode("sample", normalize_embeddings=normalize))
            self.dimension = int(sample.shape[-1])

    def __call__(self, text: str) -> np.ndarray:
        return self.encode(text)

    def encode(self, text: str) -> np.ndarray:
        kwargs = {"normalize_embeddings": self._normalize}
        if self._batch_size is not None:
            kwargs["batch_size"] = self._batch_size
        vector = self._model.encode(text, **kwargs)
        return np.asarray(vector, dtype=np.float32)

    def encode_batch(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        kwargs = {"normalize_embeddings": self._normalize}
        if self._batch_size is not None:
            kwargs["batch_size"] = self._batch_size
        vectors = self._model.encode(texts, **kwargs)
        return np.asarray(vectors, dtype=np.float32)


def embed_batch(embedder: "TextEmbedder", texts: Iterable[str]) -> np.ndarray:
    texts = list(texts)
    if not texts:
        dimension = getattr(embedder, "dimension", 0)
        return np.empty((0, int(dimension)), dtype=np.float32)
    if hasattr(embedder, "encode_batch"):
        batch_fn = getattr(embedder, "encode_batch")
        if callable(batch_fn):
            return np.asarray(batch_fn(texts), dtype=np.float32)
    vectors = [np.asarray(embedder(text), dtype=np.float32) for text in texts]
    return np.vstack(vectors)


class TextEmbedder(Protocol):
    dimension: int

    def __call__(self, text: str) -> np.ndarray:
        ...

    def encode_batch(self, texts: Iterable[str]) -> np.ndarray:
        ...
