"""Helpers for piping rich experiences into :class:`MemoryLifecycleManager`."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .lifecycle import MemoryLifecycleManager
from .task_memory import ExperiencePayload, format_experience_payload

try:  # pragma: no cover - optional dependency chain
    from modules.learning import EpisodeRecord
except Exception:  # pragma: no cover - fallback for lightweight environments
    from modules.learning.experience_hub import EpisodeRecord  # type: ignore


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class ExperienceMemoryBridge:
    """High-level façade for recording interactions in the memory lifecycle."""

    def __init__(
        self,
        lifecycle: MemoryLifecycleManager,
        *,
        consolidate_every: int = 5,
        auto_consolidate: bool = True,
    ) -> None:
        self._lifecycle = lifecycle
        self._auto_consolidate = bool(auto_consolidate)
        self._consolidate_every = max(1, consolidate_every)
        self._pending_events = 0

    # ------------------------------------------------------------------ #
    @property
    def lifecycle(self) -> MemoryLifecycleManager:
        return self._lifecycle

    # ------------------------------------------------------------------ #
    def record_episode(
        self,
        episode: EpisodeRecord,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        curiosity_samples: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        """Persist an :class:`EpisodeRecord` into the lifecycle."""

        summary = (
            f"Episode '{episode.task_id}' using policy {episode.policy_version} "
            f"yielded reward {episode.total_reward:.2f} in {episode.steps} steps."
        )
        details = "success" if episode.success else "failure"
        reward_line = (
            f"Outcome: {details}. Avg reward/step={episode.total_reward / max(1, episode.steps):.3f}."
        )
        messages = [
            {"role": "system", "content": summary},
            {"role": "agent", "content": reward_line},
        ]
        metadata = dict(episode.metadata)
        metadata.update(
            {
                "policy_version": episode.policy_version,
                "reward": float(episode.total_reward),
                "steps": int(episode.steps),
                "success": bool(episode.success),
                "timestamp": metadata.get("timestamp", time.time()),
            }
        )
        if metrics:
            metadata["agent_metrics"] = metrics

        novelty = self._extract_novelty(curiosity_samples)
        if novelty is not None:
            metadata["novelty"] = novelty

        metadata.setdefault("importance_hint", 0.8 if episode.success else 0.45)
        importance = self._episode_importance(episode, metadata)
        symbols = {
            "episode": {
                "task_id": episode.task_id,
                "policy_version": episode.policy_version,
                "total_reward": episode.total_reward,
                "steps": episode.steps,
                "success": episode.success,
                "novelty": metadata.get("novelty", 0.0),
            }
        }
        payload = ExperiencePayload(
            task_id=episode.task_id,
            summary=summary,
            messages=messages,
            metadata=metadata,
        )
        return self.record_payload(
            payload,
            importance=importance,
            symbols=symbols,
            auto_promote=episode.success,
        )

    # ------------------------------------------------------------------ #
    def record_payload(
        self,
        payload: ExperiencePayload,
        *,
        importance: Optional[float] = None,
        symbols: Optional[Dict[str, Any]] = None,
        auto_promote: bool = False,
    ) -> str:
        """Record a generic experience payload."""

        computed = importance
        if computed is None:
            computed = self._metadata_importance(payload.metadata)
        entry_id = self._lifecycle.ingest_interaction(
            payload,
            importance=_clamp(computed),
            symbols=symbols,
            auto_promote=auto_promote,
        )
        if self._auto_consolidate:
            self._pending_events += 1
            if self._pending_events >= self._consolidate_every:
                self._lifecycle.consolidate()
                self._pending_events = 0
        return entry_id

    # ------------------------------------------------------------------ #
    def recall(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        """Proxy to :meth:`MemoryLifecycleManager.recall`."""

        return self._lifecycle.recall(query, top_k=top_k)

    # ------------------------------------------------------------------ #
    def flush(self) -> None:
        """Force a consolidation pass immediately."""

        self._lifecycle.consolidate()
        self._pending_events = 0

    # ------------------------------------------------------------------ #
    def _metadata_importance(self, metadata: Dict[str, Any]) -> float:
        reward = float(metadata.get("reward", 0.0) or 0.0)
        novelty = float(metadata.get("novelty", metadata.get("surprise", 0.0)) or 0.0)
        hint = float(metadata.get("importance_hint", 0.5) or 0.5)
        reward_component = 0.35 * math.tanh(reward / 5.0)
        novelty_component = 0.2 * _clamp(novelty)
        return _clamp(hint + reward_component + novelty_component)

    def _episode_importance(self, episode: EpisodeRecord, metadata: Dict[str, Any]) -> float:
        per_step = episode.total_reward / max(1, episode.steps)
        success_bonus = 0.15 if episode.success else 0.0
        base = self._metadata_importance(metadata)
        return _clamp(base + (0.25 * math.tanh(per_step)) + success_bonus)

    def _extract_novelty(self, samples: Optional[Sequence[Dict[str, Any]]]) -> Optional[float]:
        if not samples:
            return None
        best = 0.0
        for sample in samples:
            meta = sample.get("metadata") or {}
            novelty = meta.get("novelty", meta.get("surprise"))
            if isinstance(novelty, (int, float)):
                best = max(best, float(novelty))
        return best if best > 0.0 else None


def curiosity_weighted_summarizer(payloads: List[ExperiencePayload]) -> Dict[str, Any]:
    """Summarizer that prioritises high-importance or novel experiences."""

    if not payloads:
        return {
            "text": "No experiences available for summarisation.",
            "metadata": {"summary_strategy": "curiosity_weighted", "payload_count": 0},
            "key_facts": [],
            "symbols": {},
        }

    scored: List[tuple[float, str]] = []
    total_weight = 0.0
    for payload in payloads:
        text = payload.summary.strip() if payload.summary else format_experience_payload(payload).strip()
        if not text:
            continue
        meta = payload.metadata or {}
        reward = float(meta.get("reward", 0.0) or 0.0)
        novelty = float(meta.get("novelty", meta.get("surprise", 0.0)) or 0.0)
        importance = float(meta.get("importance_hint", 0.5) or 0.5)
        weight = (
            0.5 * importance
            + 0.3 * _clamp(abs(reward) / 10.0)
            + 0.2 * _clamp(novelty)
        )
        scored.append((weight, text))
        total_weight += weight

    if not scored:
        return {
            "text": "Curiosity summarizer could not extract meaningful text.",
            "metadata": {"summary_strategy": "curiosity_weighted", "payload_count": len(payloads)},
            "key_facts": [],
            "symbols": {},
        }

    scored.sort(key=lambda item: item[0], reverse=True)
    highlights = [text for _, text in scored[: min(5, len(scored))]]
    summary_text = "\n".join(highlights)
    avg_weight = total_weight / max(1, len(scored))
    symbols = {"highlights": highlights, "avg_weight": avg_weight}
    return {
        "text": summary_text,
        "metadata": {
            "summary_strategy": "curiosity_weighted",
            "payload_count": len(payloads),
            "mean_weight": avg_weight,
        },
        "key_facts": highlights,
        "symbols": symbols,
    }


__all__ = ["ExperienceMemoryBridge", "curiosity_weighted_summarizer"]
