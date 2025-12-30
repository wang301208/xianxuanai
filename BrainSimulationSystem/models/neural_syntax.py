"""
Lightweight neural-assisted dependency parser used as a local fallback.

The implementation is intentionally modest: it relies on simple vector
representations derived from deterministic hashing, along with a perceptron-style
head scoring layer. This keeps the module fully self-contained (no external LLM
or large pretrained weights) while still allowing offline training against
project-provided dependency corpora.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NeuralDependencyParser:
    """
    Minimal neural dependency parser.

    It predicts head indices for each token using a single hidden layer
    perceptron. When no trained weights are available it degrades gracefully to
    heuristic scoring so the caller always receives a dependency structure.
    """

    def __init__(self, model_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self.embedding_dim = int(self.params.get("embedding_dim", 16))
        self.hidden_dim = int(self.params.get("hidden_dim", 32))
        self.lr = float(self.params.get("learning_rate", 0.05))
        self.relations = self.params.get(
            "relations", ["root", "subject", "object", "modifier", "aux", "attribute"]
        )

        self.model_path = Path(model_path) if model_path else None
        self.head_layer: Optional[np.ndarray] = None  # shape: (feature_dim,)
        self.root_vector = self._hash_vector("<ROOT>")
        self._feature_dim = self.embedding_dim * 3  # head, dep, element-wise product

        if self.model_path and self.model_path.exists():
            self._load(self.model_path)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def parse(self, tokens: Sequence[str], pos_tags: Sequence[str]) -> Dict[str, Any]:
        """
        Produce a dependency structure for the given sentence.

        Returns
        -------
        dict
            Dictionary with ``arcs`` containing tuples of (head_index, dep_index, relation)
            and a ``confidence`` score between 0 and 1.
        """

        length = len(tokens)
        if length == 0:
            return {"arcs": [], "confidence": 0.0}

        arcs: List[Tuple[int, int, str]] = []
        root_idx = self._predict_root(tokens, pos_tags)
        arcs.append((-1, root_idx, "root"))

        scores: List[float] = []
        for dep_idx in range(length):
            if dep_idx == root_idx:
                continue
            head_idx, score = self._predict_head(dep_idx, tokens, pos_tags, root_idx)
            relation = self._predict_relation(dep_idx, head_idx, tokens, pos_tags)
            arcs.append((head_idx, dep_idx, relation))
            scores.append(score)

        confidence = self._confidence_from_scores(scores)
        return {"arcs": arcs, "confidence": confidence}

    def train(
        self,
        samples: Iterable[Dict[str, Any]],
        epochs: int = 5,
    ) -> None:
        """
        Train the perceptron layer with labelled dependency data.

        Parameters
        ----------
        samples:
            Iterable of dictionaries containing ``tokens``, ``pos`` and ``arcs``.
            Arcs should be tuples (head_index, dep_index, relation).
        epochs:
            Number of training passes.
        """

        for epoch in range(max(1, epochs)):
            updates = 0
            for sample in samples:
                tokens = sample["tokens"]
                pos_tags = sample.get("pos") or sample.get("pos_tags")
                gold_arcs: List[Tuple[int, int, str]] = sample["arcs"]
                gold_heads = {dep: head for head, dep, _ in gold_arcs}

                for dep_idx in range(len(tokens)):
                    if dep_idx not in gold_heads:
                        continue

                    gold_head = gold_heads[dep_idx]
                    pred_head, _ = self._predict_head(dep_idx, tokens, pos_tags, gold_heads.get(-1, 0))
                    if pred_head != gold_head:
                        self._update_weights(gold_head, dep_idx, tokens, pos_tags, 1.0)
                        self._update_weights(pred_head, dep_idx, tokens, pos_tags, -1.0)
                        updates += 1

            if updates == 0:
                logger.info("NeuralDependencyParser converged after %d epoch(s).", epoch + 1)
                break

        if self.model_path:
            self._save(self.model_path)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _predict_root(self, tokens: Sequence[str], pos_tags: Sequence[str]) -> int:
        for idx, pos in enumerate(pos_tags):
            if pos in {"V", "Gerund"}:
                return idx
        return 0

    def _predict_head(
        self,
        dep_idx: int,
        tokens: Sequence[str],
        pos_tags: Sequence[str],
        root_idx: int,
    ) -> Tuple[int, float]:
        best_head = root_idx if dep_idx != root_idx else -1
        best_score = float("-inf")

        for head_idx in range(-1, len(tokens)):
            if head_idx == dep_idx:
                continue
            score = self._score_pair(head_idx, dep_idx, tokens, pos_tags)
            if score > best_score:
                best_score = score
                best_head = head_idx

        return best_head, best_score

    def _predict_relation(
        self,
        dep_idx: int,
        head_idx: int,
        tokens: Sequence[str],
        pos_tags: Sequence[str],
    ) -> str:
        dep_pos = pos_tags[dep_idx]
        if head_idx == -1:
            return "root"
        if dep_pos in {"Pronoun", "NP"} and head_idx < dep_idx:
            return "subject"
        if dep_pos in {"N", "ProperNoun"} and head_idx < dep_idx:
            return "object"
        if dep_pos == "Aux":
            return "aux"
        if dep_pos in {"Adj", "Adv", "P"}:
            return "modifier"
        return "attribute"

    def _score_pair(
        self,
        head_idx: int,
        dep_idx: int,
        tokens: Sequence[str],
        pos_tags: Sequence[str],
    ) -> float:
        features = self._pair_features(head_idx, dep_idx, tokens, pos_tags)
        if self.head_layer is None:
            # Lazy initialisation with small random values for smooth scores.
            rng = np.random.default_rng(42)
            self.head_layer = rng.normal(0, 0.05, size=self._feature_dim)

        return float(np.dot(self.head_layer, features))

    def _pair_features(
        self,
        head_idx: int,
        dep_idx: int,
        tokens: Sequence[str],
        pos_tags: Sequence[str],
    ) -> np.ndarray:
        dep_vec = self._token_vector(dep_idx, tokens, pos_tags)
        head_vec = self._token_vector(head_idx, tokens, pos_tags)
        product = dep_vec * head_vec
        return np.concatenate([head_vec, dep_vec, product])

    def _token_vector(
        self,
        idx: int,
        tokens: Sequence[str],
        pos_tags: Sequence[str],
    ) -> np.ndarray:
        if idx == -1:
            return self.root_vector
        key = f"{tokens[idx].lower()}::{pos_tags[idx]}"
        return self._hash_vector(key)

    def _hash_vector(self, key: str) -> np.ndarray:
        seed = abs(hash(key)) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.embedding_dim).astype(np.float32)

    def _confidence_from_scores(self, scores: Sequence[float]) -> float:
        if not scores:
            return 0.5
        scores = np.array(scores, dtype=np.float32)
        scores -= scores.mean()
        max_score = float(np.max(scores))
        return float(np.clip(0.5 + max_score / (1 + np.abs(max_score)), 0.0, 0.95))

    def _update_weights(
        self,
        head_idx: int,
        dep_idx: int,
        tokens: Sequence[str],
        pos_tags: Sequence[str],
        direction: float,
    ) -> None:
        features = self._pair_features(head_idx, dep_idx, tokens, pos_tags)
        if self.head_layer is None:
            self.head_layer = np.zeros(self._feature_dim, dtype=np.float32)
        self.head_layer += self.lr * direction * features

    def _save(self, path: Path) -> None:
        if self.head_layer is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, head_layer=self.head_layer)
        logger.info("Saved neural syntax weights to %s", path)

    def _load(self, path: Path) -> None:
        try:
            data = np.load(path)
            self.head_layer = data["head_layer"]
            logger.info("Loaded neural syntax weights from %s", path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load neural syntax weights: %s", exc)
            self.head_layer = None
