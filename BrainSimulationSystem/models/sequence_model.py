"""Lightweight recurrent sequence learner for tokens and POS tags.

This module keeps a tiny recurrent state and learns token/tag embeddings
online. It predicts next tokens/tags via similarity between the current hidden
state and learned embeddings, and stores frequent tag transitions for grammar
induction.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple


class LightweightSequenceModel:
    """A minimal recurrent model that supports online updates.

    The model blends token and tag embeddings into a running hidden state and
    nudges embeddings toward the hidden state seen before the next token/tag.
    It exposes next-token/POS probabilities and frequently observed transition
    patterns for downstream grammar induction.
    """

    def __init__(self, params: Optional[Dict[str, object]] = None) -> None:
        self.params = params or {}
        self.hidden_size = int(self.params.get("hidden_size", 48))
        self.learning_rate = float(self.params.get("learning_rate", 0.05))
        self.mix_rate = float(self.params.get("mix_rate", 0.35))
        self.max_history = int(self.params.get("max_history", 4))

        self.token_embeddings: Dict[str, List[float]] = {"<unk>": self._init_vector()}
        self.tag_embeddings: Dict[str, List[float]] = {"UNK": self._init_vector()}
        self.pattern_memory: MutableMapping[Tuple[str, ...], MutableMapping[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def observe_sequence(self, tokens: Sequence[str], pos_tags: Optional[Sequence[str]] = None) -> None:
        if not tokens:
            return
        tags = list(pos_tags) if pos_tags else ["UNK"] * len(tokens)
        hidden = [0.0 for _ in range(self.hidden_size)]
        recent_tags: List[str] = []

        for idx, token in enumerate(tokens):
            tag = tags[idx] if idx < len(tags) else "UNK"
            token_vec = self._get_token_vector(token)
            tag_vec = self._get_tag_vector(tag)
            hidden = self._blend_hidden(hidden, token_vec, tag_vec)

            if idx + 1 < len(tokens):
                next_token = tokens[idx + 1]
                self._update_embedding(next_token, hidden, is_tag=False)
            if idx + 1 < len(tags):
                next_tag = tags[idx + 1]
                self._update_embedding(next_tag, hidden, is_tag=True)
                recent_tags.append(tag)
                history = tuple(recent_tags[-self.max_history :])
                self.pattern_memory[history][next_tag] += 1.0

    def predict_next(self, context_tokens: Sequence[str], context_tags: Optional[Sequence[str]] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        hidden = self._roll_hidden_state(context_tokens, context_tags)
        token_probs = self._predict_from_hidden(hidden, self.token_embeddings)
        tag_probs = self._predict_from_hidden(hidden, self.tag_embeddings)
        return token_probs, tag_probs

    def predict_tags(self, context_tags: Sequence[str]) -> Dict[str, float]:
        hidden = self._roll_hidden_state([], context_tags)
        return self._predict_from_hidden(hidden, self.tag_embeddings)

    def get_rule_candidates(self, threshold: float = 0.35, top_k: int = 10) -> List[Dict[str, object]]:
        rules: List[Dict[str, object]] = []
        for history, outcomes in self.pattern_memory.items():
            total = sum(outcomes.values())
            if total <= 0:
                continue
            for tag, score in outcomes.items():
                probability = score / total
                if probability < threshold:
                    continue
                rules.append({"history": history, "next": tag, "probability": probability})
        rules.sort(key=lambda r: r["probability"], reverse=True)
        if top_k:
            rules = rules[:top_k]
        return rules

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _init_vector(self) -> List[float]:
        return [random.uniform(-0.1, 0.1) for _ in range(self.hidden_size)]

    def _get_token_vector(self, token: str) -> List[float]:
        if token not in self.token_embeddings:
            self.token_embeddings[token] = self._init_vector()
        return self.token_embeddings[token]

    def _get_tag_vector(self, tag: str) -> List[float]:
        if tag not in self.tag_embeddings:
            self.tag_embeddings[tag] = self._init_vector()
        return self.tag_embeddings[tag]

    def _blend_hidden(self, hidden: List[float], token_vec: List[float], tag_vec: List[float]) -> List[float]:
        combined = [(tv + tg) * 0.5 for tv, tg in zip(token_vec, tag_vec)]
        return [h * (1.0 - self.mix_rate) + c * self.mix_rate for h, c in zip(hidden, combined)]

    def _update_embedding(self, target: str, reference: List[float], *, is_tag: bool) -> None:
        store = self.tag_embeddings if is_tag else self.token_embeddings
        vector = self._get_tag_vector(target) if is_tag else self._get_token_vector(target)
        for i in range(self.hidden_size):
            vector[i] += self.learning_rate * (reference[i] - vector[i])
        store[target] = vector

    def _roll_hidden_state(self, tokens: Sequence[str], tags: Optional[Sequence[str]]) -> List[float]:
        hidden = [0.0 for _ in range(self.hidden_size)]
        tags = list(tags) if tags is not None else []
        for idx, token in enumerate(tokens):
            tag = tags[idx] if idx < len(tags) else "UNK"
            hidden = self._blend_hidden(hidden, self._get_token_vector(token), self._get_tag_vector(tag))
        if not tokens and tags:
            for tag in tags:
                hidden = self._blend_hidden(hidden, self._get_token_vector("<unk>"), self._get_tag_vector(tag))
        return hidden

    def _predict_from_hidden(self, hidden: Sequence[float], embeddings: Dict[str, List[float]]) -> Dict[str, float]:
        scores = {key: self._dot(hidden, vec) for key, vec in embeddings.items()}
        return self._softmax(scores)

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        return {k: v / total for k, v in exp_scores.items()}


__all__ = ["LightweightSequenceModel"]
