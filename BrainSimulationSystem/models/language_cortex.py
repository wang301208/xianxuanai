"""
Minimal internal language cortex model.

Simulates an in-house neural language module trained on project data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class LanguageCortex:
    """Lightweight stateful encoder/decoder for language patterns."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        params = params or {}
        self.embedding_dim = int(params.get("embedding_dim", 64))
        self.hidden_dim = int(params.get("hidden_dim", 64))
        self.learning_rate = float(params.get("learning_rate", 0.01))
        seed = int(params.get("seed", 17))
        self.rng = np.random.default_rng(seed)

        scale = 1.0 / max(4, self.embedding_dim)
        self.word_embeddings: Dict[str, np.ndarray] = {}
        self.W_enc = self.rng.normal(0, scale, (self.hidden_dim, self.embedding_dim))
        self.W_proj = self.rng.normal(0, scale, (self.hidden_dim, self.hidden_dim))
        self.W_out = self.rng.normal(0, scale, (self.hidden_dim, self.hidden_dim))
        self.context_state = np.zeros(self.hidden_dim)

    def encode_tokens(self, tokens: Any) -> np.ndarray:
        vectors = [self._embedding(tok) for tok in tokens]
        if not vectors:
            return np.zeros(self.hidden_dim)
        stacked = np.stack(vectors)
        mean_vec = np.mean(stacked, axis=0)
        context = np.tanh(self.W_enc @ mean_vec + self.W_proj @ self.context_state)
        self.context_state = 0.8 * self.context_state + 0.2 * context
        return context

    def predict_intent_vector(self, context: np.ndarray) -> np.ndarray:
        return np.tanh(self.W_out @ context)

    def update_from_example(
        self,
        tokens: Any,
        target_vector: np.ndarray,
        momentum: float = 0.9,
    ) -> None:
        context = self.encode_tokens(tokens)
        pred = self.predict_intent_vector(context)
        error = target_vector - pred
        grad = error * (1 - pred ** 2)

        self.W_out += self.learning_rate * np.outer(grad, context)
        grad_context = self.W_out.T @ grad * (1 - context ** 2)
        self.W_enc += self.learning_rate * np.outer(grad_context, context)
        self.context_state = momentum * self.context_state + (1 - momentum) * context

    def reset_state(self) -> None:
        self.context_state[:] = 0.0

    def _embedding(self, token: str) -> np.ndarray:
        if token not in self.word_embeddings:
            self.word_embeddings[token] = self.rng.normal(0, 0.1, self.embedding_dim)
        return self.word_embeddings[token]
