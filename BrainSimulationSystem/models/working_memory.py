"""
Lightweight working memory buffer for dialogue context.

This module aggregates recent comprehension/generation events into a compact
state dictionary that higher-level controllers can consume. The implementation
sticks to simple statistics so it can adapt online without external training.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class MemoryStateEncoder:
    """Simple online GRU encoder producing a contextual state vector."""

    def __init__(
        self,
        vector_size: int = 48,
        hidden_size: int = 32,
        learning_rate: float = 0.02,
        seed: int = 13,
    ) -> None:
        self.vector_size = vector_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        scale = 0.15
        self.Wz = self.rng.normal(0, scale, (hidden_size, vector_size))
        self.Uz = self.rng.normal(0, scale, (hidden_size, hidden_size))
        self.bz = np.zeros(hidden_size)

        self.Wr = self.rng.normal(0, scale, (hidden_size, vector_size))
        self.Ur = self.rng.normal(0, scale, (hidden_size, hidden_size))
        self.br = np.zeros(hidden_size)

        self.Wh = self.rng.normal(0, scale, (hidden_size, vector_size))
        self.Uh = self.rng.normal(0, scale, (hidden_size, hidden_size))
        self.bh = np.zeros(hidden_size)

        self.state = np.zeros(hidden_size)

    def encode_features(self, record: Dict[str, Any]) -> np.ndarray:
        vec = np.zeros(self.vector_size)

        def add_feature(token: str, weight: float = 1.0) -> None:
            if not token:
                return
            idx = hash(token) % self.vector_size
            vec[idx] += weight

        add_feature(f"type:{record.get('type', '')}", 0.8)
        add_feature(f"intent:{record.get('intent', '')}", 1.0)
        add_feature(f"polarity:{record.get('polarity', '')}", 0.6)
        add_feature(f"status:{record.get('status', '')}", 0.5)

        for term in record.get("key_terms") or []:
            add_feature(f"term:{term}", 0.7)
        for tone in record.get("tone") or []:
            add_feature(f"tone:{tone}", 0.6)
        for action in record.get("actions") or []:
            add_feature(f"action:{action}", 0.9)

        summary = record.get("summary")
        if isinstance(summary, str):
            add_feature(f"summary_len:{len(summary) // 20}", 0.4)

        return vec

    def update(self, record: Dict[str, Any]) -> None:
        x = self.encode_features(record)
        z = _sigmoid(self.Wz @ x + self.Uz @ self.state + self.bz)
        r = _sigmoid(self.Wr @ x + self.Ur @ self.state + self.br)
        candidate = np.tanh(self.Wh @ x + self.Uh @ (r * self.state) + self.bh)
        new_state = (1 - z) * self.state + z * candidate

        lr = self.learning_rate
        self.Wz = (1 - lr * 0.1) * self.Wz + lr * np.outer(z, x)
        self.Uz = (1 - lr * 0.1) * self.Uz + lr * np.outer(z, self.state)
        self.Wr = (1 - lr * 0.1) * self.Wr + lr * np.outer(r, x)
        self.Ur = (1 - lr * 0.1) * self.Ur + lr * np.outer(r, self.state)
        self.Wh = (1 - lr * 0.1) * self.Wh + lr * np.outer(candidate, x)
        self.Uh = (1 - lr * 0.1) * self.Uh + lr * np.outer(candidate, r * self.state)

        self.state = new_state

    def reset(self) -> None:
        self.state = np.zeros(self.hidden_size)



class WorkingMemory:
    """Maintain a rolling buffer of dialogue context."""

    def __init__(
        self,
        network: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(network, dict) and params is None:
            params = network
            network = None

        self.network = network
        self.params: Dict[str, Any] = params or {}

        self.capacity = int(self.params.get("capacity", 6))
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=self.capacity)
        self.turn_counter = 0
        encoder_params = self.params.get("encoder", {})
        self.encoder = (
            MemoryStateEncoder(**encoder_params)
            if self.params.get("enable_encoder", True)
            else None
        )
        self.entity_capacity = int(self.params.get("entity_capacity", 12))
        self.recent_entities: Deque[Dict[str, str]] = deque(maxlen=self.entity_capacity)
        self.pronoun_preferences: Dict[str, Sequence[str]] = self.params.get(
            "pronoun_preferences",
            {
                "he": ("person",),
                "she": ("person",),
                "him": ("person",),
                "her": ("person",),
                "they": ("person", "entity"),
                "them": ("person", "entity"),
                "their": ("person", "entity"),
                "theirs": ("person", "entity"),
                "it": ("object", "entity"),
                "its": ("object", "entity"),
                "itself": ("object", "entity"),
                "himself": ("person",),
                "herself": ("person",),
                "themselves": ("person", "entity"),
            },
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def add_record(self, record: Dict[str, Any]) -> None:
        snapshot = dict(record)
        snapshot["turn_id"] = self.turn_counter
        self.turn_counter += 1
        self.buffer.append(snapshot)
        if self.encoder:
            self.encoder.update(snapshot)
        self._update_entity_memory(snapshot)

    def get_state(self) -> Dict[str, Any]:
        """Return aggregated context features."""
        state_vector = (
            self.encoder.state.astype(float).tolist() if self.encoder is not None else []
        )
        if not self.buffer:
            return {
                "key_terms": [],
                "pending_actions": [],
                "tone_counts": {},
                "recent_intent": None,
                "recent_summary": "",
                "state_vector": state_vector,
            }

        key_term_counter: Counter[str] = Counter()
        action_counter: Counter[str] = Counter()
        tone_counter: Counter[str] = Counter()
        polarity_counter: Counter[str] = Counter()
        unresolved_actions: List[str] = []
        last_intent = None
        last_summary = ""

        for record in self.buffer:
            key_terms = record.get("key_terms") or []
            key_term_counter.update(key_terms)
            tones = record.get("tone") or []
            tone_counter.update(tones)
            polarity = record.get("polarity")
            if polarity:
                polarity_counter.update([polarity])
            actions = record.get("actions") or []
            action_counter.update(actions)
            if record.get("status") == "pending":
                unresolved_actions.extend(actions)
            if record.get("type") == "input":
                last_intent = record.get("intent", last_intent)
                last_summary = record.get("summary", last_summary)

        key_terms = [term for term, _ in key_term_counter.most_common(8)]
        pending = list(dict.fromkeys(unresolved_actions))

        return {
            "key_terms": key_terms,
            "pending_actions": pending,
            "tone_counts": dict(tone_counter),
            "polarity_counts": dict(polarity_counter),
            "recent_intent": last_intent,
            "recent_summary": last_summary,
            "state_vector": state_vector,
        }

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.buffer)

    def update_last_record(self, updates: Dict[str, Any]) -> None:
        if not self.buffer:
            return
        last = self.buffer.pop()
        last.update(updates)
        self.buffer.append(last)
        self._update_entity_memory(last)

    def resolve_pronoun(self, pronoun: str) -> Optional[str]:
        if not pronoun:
            return None
        pronoun_lower = pronoun.lower()
        preferred_groups = self.pronoun_preferences.get(pronoun_lower)
        if not preferred_groups:
            return None
        for entity in reversed(self.recent_entities):
            if "entity" in preferred_groups or "any" in preferred_groups:
                return entity["name"]
            if entity["category"] in preferred_groups:
                return entity["name"]
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _update_entity_memory(self, record: Dict[str, Any]) -> None:
        entities = record.get("entities")
        if not entities:
            semantic = record.get("semantic") or {}
            entities = semantic.get("entities")
        if not entities:
            return
        concept_stats = (record.get("semantic") or {}).get("concept_stats", {})
        for entity in entities:
            if not entity:
                continue
            category = self._infer_entity_category(entity, concept_stats)
            self.recent_entities.append({"name": entity, "category": category})

    @staticmethod
    def _infer_entity_category(entity: str, concept_stats: Dict[str, Dict[str, Any]]) -> str:
        key = entity.lower()
        info = concept_stats.get(key, {})
        pos = (info.get("pos") or info.get("part_of_speech") or "").lower()
        if pos in {"propernoun", "prp", "nnp"} or entity[:1].isupper():
            return "person"
        if pos in {"n", "nn", "noun"}:
            return "object"
        return "entity"


class AChModulatedWorkingMemory(WorkingMemory):
    """Working memory variant modulated by acetylcholine-like gain."""

    def __init__(
        self,
        network: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(network, params)
        try:
            self.ach_sensitivity = float(self.params.get("ach_sensitivity", 1.0))
        except (TypeError, ValueError):
            self.ach_sensitivity = 1.0
