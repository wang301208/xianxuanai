"""
Blackboard-based attention management for multimodal information integration.

This module provides:
    - BlackboardEntry: encapsulates candidate information with metadata.
    - BlackboardWorkspace: shared buffer for cross-module communication.
    - AttentionManager: configurable scoring controller (supports heuristic,
      transformer-style self-attention and optional RL-driven policies).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional RL backend
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover
    PPO = None  # type: ignore[assignment]


@dataclass
class BlackboardEntry:
    """Represents a candidate item written onto the blackboard."""

    source: str
    payload: Dict[str, Any]
    salience: float = 0.5
    confidence: float = 0.5
    novelty: float = 0.5
    timestamp: float = field(default_factory=time.time)

    def recency(self, now: Optional[float] = None) -> float:
        now = now or time.time()
        delta = max(now - self.timestamp, 1e-6)
        return float(1.0 / (1.0 + delta))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "payload": self.payload,
            "salience": self.salience,
            "confidence": self.confidence,
            "novelty": self.novelty,
            "timestamp": self.timestamp,
        }


class BlackboardWorkspace:
    """Shared workspace aggregating entries from multiple modalities."""

    def __init__(self, decay: float = 0.96) -> None:
        self.decay = float(decay)
        self.entries: List[BlackboardEntry] = []

    def add_entry(
        self,
        source: str,
        payload: Dict[str, Any],
        salience: float = 0.5,
        confidence: float = 0.5,
        novelty: float = 0.5,
    ) -> BlackboardEntry:
        entry = BlackboardEntry(
            source=source,
            payload=payload,
            salience=float(np.clip(salience, 0.0, 1.0)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            novelty=float(np.clip(novelty, 0.0, 1.0)),
            timestamp=time.time(),
        )
        self.entries.append(entry)
        return entry

    def iter_entries(self) -> List[BlackboardEntry]:
        return list(self.entries)

    def prune(
        self,
        max_entries: int = 128,
        *,
        frozen_sources: Optional[Iterable[str]] = None,
    ) -> None:
        frozen_set = {str(src) for src in (frozen_sources or [])}
        now = time.time()
        pruned: List[BlackboardEntry] = []
        for entry in self.entries:
            if entry.source not in frozen_set:
                age = max(now - entry.timestamp, 0.0)
                if age > 0.0:
                    decay_factor = self.decay ** age
                    entry.salience *= decay_factor
                    entry.confidence *= decay_factor
                    entry.novelty *= decay_factor
            if entry.salience > 0.05 or entry.confidence > 0.05:
                pruned.append(entry)

        pruned.sort(key=lambda e: e.timestamp, reverse=True)
        self.entries = pruned[:max_entries]

    def clear(self) -> None:
        self.entries.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlackboardWorkspace":
        workspace = cls(decay=float(data.get("decay", 0.96)))
        for entry_data in data.get("entries", []):
            workspace.entries.append(
                BlackboardEntry(
                    source=str(entry_data.get("source", "unknown")),
                    payload=entry_data.get("payload", {}),
                    salience=float(entry_data.get("salience", 0.5)),
                    confidence=float(entry_data.get("confidence", 0.5)),
                    novelty=float(entry_data.get("novelty", 0.5)),
                    timestamp=float(entry_data.get("timestamp", time.time())),
                )
            )
        return workspace


@dataclass
class AttentionConfig:
    """Configuration for the AttentionManager."""

    focus_count: int = 5
    salience_weight: float = 0.5
    confidence_weight: float = 0.3
    novelty_weight: float = 0.2
    motivation_weight: float = 0.3
    decay: float = 0.96
    rl_model_path: Optional[str] = None
    scoring: str = "heuristic"
    focus_capacity: Optional[int] = None
    transformer_hidden_dim: int = 32
    transformer_weights_path: Optional[str] = None
    history_size: int = 32
    freeze_sources: Tuple[str, ...] = field(default_factory=tuple)
    suppressed_decay: float = 0.85
    allow_custom_strategies: bool = True


ScoringStrategy = Callable[
    [List[BlackboardEntry], Dict[str, Any], Dict[str, float], float],
    List[Tuple[float, BlackboardEntry]],
]


class AttentionManager:
    """Selects salient entries from the blackboard using configurable scoring."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = AttentionConfig(**(config or {}))
        self.workspace = BlackboardWorkspace(decay=self.config.decay)
        self._rl_policy = None
        if self.config.rl_model_path and PPO is not None:
            try:  # pragma: no cover - requires trained model
                self._rl_policy = PPO.load(self.config.rl_model_path)
            except Exception:
                self._rl_policy = None

        self.active_scoring = str(self.config.scoring or "heuristic").lower()
        self.frozen_sources: Tuple[str, ...] = tuple(str(src) for src in self.config.freeze_sources)
        self.focus_limit = max(1, int(self.config.focus_capacity or self.config.focus_count))
        history_size = max(1, int(self.config.history_size))
        self.focus_history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self.last_focus: List[Dict[str, Any]] = []
        self._strategies: Dict[str, ScoringStrategy] = {
            "heuristic": self._heuristic_strategy,
            "transformer": self._transformer_strategy,
            "rl": self._rl_strategy,
            "hybrid": self._hybrid_strategy,
        }
        self._transformer_weights = self._initialize_transformer_weights()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add(self, *args, **kwargs) -> BlackboardEntry:
        return self.workspace.add_entry(*args, **kwargs)

    def clear(self) -> None:
        self.workspace.clear()
        self.focus_history.clear()
        self.last_focus = []

    def register_strategy(self, name: str, handler: ScoringStrategy) -> None:
        if not self.config.allow_custom_strategies:
            raise RuntimeError("Custom strategies are disabled by configuration.")
        key = str(name).lower()
        if not callable(handler):
            raise TypeError("Strategy handler must be callable.")
        self._strategies[key] = handler

    def set_scoring_strategy(self, name: str) -> None:
        key = str(name).lower()
        if key not in self._strategies:
            raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(self._strategies)}")
        self.active_scoring = key

    def select_focus(
        self,
        context: Optional[Dict[str, Any]] = None,
        motivation: Optional[Dict[str, float]] = None,
        *,
        apply_suppression: bool = True,
    ) -> Dict[str, Any]:
        context = context or {}
        motivation = {str(k): float(v) for k, v in (motivation or {}).items()}

        self.workspace.prune(frozen_sources=self.frozen_sources)
        entries = self.workspace.iter_entries()
        now = time.time()

        if not entries:
            snapshot = {
                "focus": [],
                "suppressed": [],
                "scores": [],
                "strategy": self.active_scoring,
                "timestamp": now,
                "workspace_size": 0,
            }
            self.last_focus = []
            return snapshot

        strategy = self._strategies.get(self.active_scoring, self._heuristic_strategy)
        scored_entries = strategy(entries, context, motivation, now)
        scored_entries.sort(key=lambda item: item[0], reverse=True)

        focus_entries = scored_entries[: self.focus_limit]
        focus: List[Dict[str, Any]] = [
            self._entry_to_focus_dict(entry, score) for score, entry in focus_entries
        ]
        suppressed_ids = {id(entry) for _, entry in focus_entries}
        suppressed = [
            self._entry_to_focus_dict(entry, score)
            for score, entry in scored_entries
            if id(entry) not in suppressed_ids
        ]

        if apply_suppression:
            decay = float(np.clip(self.config.suppressed_decay, 0.0, 1.0))
            for score, entry in scored_entries:
                if id(entry) in suppressed_ids or entry.source in self.frozen_sources:
                    continue
                entry.salience *= decay
                entry.confidence *= decay
                entry.novelty *= decay

        history_entry = {
            "time": now,
            "strategy": self.active_scoring,
            "focus": focus,
            "scores": [
                {
                    "source": entry.source,
                    "score": score,
                    "salience": entry.salience,
                    "confidence": entry.confidence,
                    "novelty": entry.novelty,
                }
                for score, entry in scored_entries
            ],
            "workspace_size": len(entries),
        }
        self.focus_history.append(history_entry)
        self.last_focus = focus

        return {
            "focus": focus,
            "suppressed": suppressed,
            "scores": history_entry["scores"],
            "strategy": self.active_scoring,
            "timestamp": now,
            "workspace_size": len(entries),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace": self.workspace.to_dict(),
            "last_focus": self.last_focus,
            "history": list(self.focus_history),
            "strategy": self.active_scoring,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self.workspace = BlackboardWorkspace.from_dict(data.get("workspace", {}))
        self.last_focus = list(data.get("last_focus", []))
        history = data.get("history", [])
        self.focus_history.clear()
        for item in history:
            if isinstance(item, dict):
                self.focus_history.append(item)
        strategy = data.get("strategy")
        if isinstance(strategy, str) and strategy.lower() in self._strategies:
            self.active_scoring = strategy.lower()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _initialize_transformer_weights(self) -> Dict[str, np.ndarray]:
        feature_dim = 6  # salience, confidence, novelty, recency, motivation, urgency
        hidden_dim = max(1, int(self.config.transformer_hidden_dim))

        if self.config.transformer_weights_path:
            path = Path(os.path.expanduser(self.config.transformer_weights_path))
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    return {
                        "w_q": np.asarray(raw["w_q"], dtype=np.float32),
                        "w_k": np.asarray(raw["w_k"], dtype=np.float32),
                        "w_v": np.asarray(raw["w_v"], dtype=np.float32),
                    }
                except Exception:
                    pass

        rng = np.random.default_rng(42)
        return {
            "w_q": rng.normal(0.0, 0.2, size=(feature_dim, hidden_dim)).astype(np.float32),
            "w_k": rng.normal(0.0, 0.2, size=(feature_dim, hidden_dim)).astype(np.float32),
            "w_v": rng.normal(0.0, 0.2, size=(feature_dim, hidden_dim)).astype(np.float32),
        }

    def _heuristic_strategy(
        self,
        entries: List[BlackboardEntry],
        context: Dict[str, Any],
        motivation: Dict[str, float],
        now: float,
    ) -> List[Tuple[float, BlackboardEntry]]:
        scored_entries: List[Tuple[float, BlackboardEntry]] = []
        for entry in entries:
            score = self._heuristic_score(entry, motivation, now)
            scored_entries.append((score, entry))
        return scored_entries

    def _transformer_strategy(
        self,
        entries: List[BlackboardEntry],
        context: Dict[str, Any],
        motivation: Dict[str, float],
        now: float,
    ) -> List[Tuple[float, BlackboardEntry]]:
        if not entries:
            return []

        features = []
        for entry in entries:
            features.append(self._build_feature_vector(entry, context, motivation, now))

        features_np = np.asarray(features, dtype=np.float32)
        w_q = self._transformer_weights["w_q"]
        w_k = self._transformer_weights["w_k"]
        w_v = self._transformer_weights["w_v"]

        q = features_np @ w_q
        k = features_np @ w_k
        v = features_np @ w_v

        scale = math.sqrt(float(q.shape[1])) if q.shape[1] else 1.0
        attn_logits = (q @ k.T) / max(scale, 1e-6)
        attn_weights = self._softmax(attn_logits.mean(axis=0))

        value_projection = (v * attn_weights[:, None]).sum(axis=1)
        heuristic_scores = np.asarray(
            [self._heuristic_score(entry, motivation, now) for entry in entries],
            dtype=np.float32,
        )
        combined = 0.6 * heuristic_scores + 0.4 * value_projection

        scored_entries: List[Tuple[float, BlackboardEntry]] = []
        for score, entry, weight in zip(combined.tolist(), entries, attn_weights.tolist()):
            scored_entries.append((float(score + 0.1 * weight), entry))
        return scored_entries

    def _rl_strategy(
        self,
        entries: List[BlackboardEntry],
        context: Dict[str, Any],
        motivation: Dict[str, float],
        now: float,
    ) -> List[Tuple[float, BlackboardEntry]]:
        if self._rl_policy is None:
            return self._heuristic_strategy(entries, context, motivation, now)

        scored_entries: List[Tuple[float, BlackboardEntry]] = []
        for entry in entries:
            score = self._rl_score(entry, context, motivation, now)
            scored_entries.append((score, entry))
        return scored_entries

    def _hybrid_strategy(
        self,
        entries: List[BlackboardEntry],
        context: Dict[str, Any],
        motivation: Dict[str, float],
        now: float,
    ) -> List[Tuple[float, BlackboardEntry]]:
        heuristic_scores = {
            id(entry): self._heuristic_score(entry, motivation, now) for entry in entries
        }
        rl_scores = {}
        if self._rl_policy is not None:
            rl_scores = {
                id(entry): self._rl_score(entry, context, motivation, now) for entry in entries
            }

        scored_entries: List[Tuple[float, BlackboardEntry]] = []
        for entry in entries:
            key = id(entry)
            heuristic = heuristic_scores[key]
            rl = rl_scores.get(key, heuristic)
            score = 0.5 * heuristic + 0.5 * rl
            scored_entries.append((score, entry))
        return scored_entries

    def _heuristic_score(
        self,
        entry: BlackboardEntry,
        motivation: Dict[str, float],
        now: float,
    ) -> float:
        motivation_score = self._motivation_alignment(entry, motivation)
        recency = entry.recency(now)
        score = (
            self.config.salience_weight * entry.salience
            + self.config.confidence_weight * entry.confidence
            + self.config.novelty_weight * entry.novelty
            + self.config.motivation_weight * motivation_score
            + 0.1 * recency
        )
        return float(score)

    def _rl_score(
        self,
        entry: BlackboardEntry,
        context: Dict[str, Any],
        motivation: Dict[str, float],
        now: float,
    ) -> float:  # pragma: no cover - requires RL
        if self._rl_policy is None:
            return 0.0
        vector = self._build_feature_vector(entry, context, motivation, now)
        action, _ = self._rl_policy.predict(vector.astype(np.float32), deterministic=True)
        if isinstance(action, np.ndarray):
            return float(np.mean(action))
        try:
            return float(action)
        except Exception:
            return 0.0

    def _build_feature_vector(
        self,
        entry: BlackboardEntry,
        context: Dict[str, Any],
        motivation: Dict[str, float],
        now: float,
    ) -> np.ndarray:
        motivation_score = self._motivation_alignment(entry, motivation)
        urgency = float(context.get("task_urgency", 0.5))
        goal = entry.payload.get("goal")
        goal_len = float(len(goal)) / 10.0 if isinstance(goal, str) else 0.0
        return np.asarray(
            [
                float(entry.salience),
                float(entry.confidence),
                float(entry.novelty),
                float(entry.recency(now)),
                motivation_score,
                max(urgency, goal_len),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if not arr.size:
            return np.asarray([], dtype=np.float32)
        shifted = arr - np.max(arr)
        exp = np.exp(shifted)
        denom = exp.sum()
        if denom <= 0:
            return np.ones_like(exp) / float(exp.size)
        return exp / denom

    @staticmethod
    def _motivation_alignment(entry: BlackboardEntry, motivation: Dict[str, float]) -> float:
        payload_goal = entry.payload.get("goal")
        goal_score = 0.0
        if payload_goal and payload_goal in motivation:
            goal_score = float(motivation[payload_goal])
        source_score = float(motivation.get(entry.source, 0.0))
        return max(goal_score, source_score)

    @staticmethod
    def _entry_to_focus_dict(entry: BlackboardEntry, score: float) -> Dict[str, Any]:
        payload = entry.payload if isinstance(entry.payload, dict) else {"value": entry.payload}
        return {
            "source": entry.source,
            "payload": payload,
            "score": float(score),
            "salience": float(entry.salience),
            "confidence": float(entry.confidence),
            "novelty": float(entry.novelty),
        }
