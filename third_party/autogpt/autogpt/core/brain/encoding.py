"""Utilities for constructing brain observations and memory contexts."""
from __future__ import annotations

import math
from typing import Iterable, Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from autogpt.agents.base import BaseAgent


def _safe_hash(value: str | None) -> float:
    """Map string values to a stable floating point in [-1, 1]."""

    if not value:
        return 0.0
    total = 0
    for idx, byte in enumerate(value.encode("utf-8", errors="ignore")):
        total = (total + (byte << (idx % 8))) & 0xFFFFFFFF
    scaled = (total % 2000000) / 1000000.0  # -> [0, 2)
    return float(scaled - 1.0)


def _write_feature(vector: torch.Tensor, index: int, value: float) -> int:
    if index < vector.numel():
        vector[index] = float(value)
    return index + 1


def _encode_recent_actions(
    vector: torch.Tensor, index: int, actions: Sequence[str], limit: int
) -> int:
    for name in actions[:limit]:
        index = _write_feature(vector, index, _safe_hash(name))
    return index


def _encode_numeric_series(
    vector: torch.Tensor, index: int, series: Iterable[float], limit: int
) -> int:
    for value in series:
        if limit <= 0:
            break
        index = _write_feature(vector, index, float(value))
        limit -= 1
    return index


def build_brain_inputs(agent: "BaseAgent", dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct observation and memory tensors for the transformer brain."""

    observation = torch.zeros(dim, dtype=torch.float32)
    memory_ctx = torch.zeros(dim, dtype=torch.float32)

    count = agent.config.cycle_count
    budget = agent.config.cycle_budget or 0
    episodes = list(agent.event_history.episodes)
    successes = sum(
        1
        for ep in episodes
        if ep.result is not None and getattr(ep.result, "status", "") == "success"
    )
    failures = sum(
        1
        for ep in episodes
        if ep.result is not None and getattr(ep.result, "status", "") == "error"
    )

    idx = 0
    idx = _write_feature(observation, idx, math.tanh(count / 10 if count else 0.0))
    idx = _write_feature(observation, idx, math.tanh((budget or 0) / 10))
    idx = _write_feature(observation, idx, math.tanh(len(episodes) / 10))
    total = max(len(episodes), 1)
    idx = _write_feature(observation, idx, successes / total)
    idx = _write_feature(observation, idx, failures / total)

    recent_actions = [ep.action.name for ep in episodes[-10:] if ep.action]
    idx = _encode_recent_actions(observation, idx, recent_actions[::-1], limit=10)

    recent_reasoning = [ep.action.reasoning for ep in episodes[-5:] if ep.action]
    for reasoning in recent_reasoning[::-1]:
        idx = _write_feature(observation, idx, _safe_hash(reasoning))

    recent_status = [
        1.0 if ep.result and getattr(ep.result, "status", "") == "success" else 0.0
        for ep in episodes[-10:]
    ]
    idx = _encode_numeric_series(observation, idx, recent_status[::-1], limit=10)

    m_idx = 0
    task = getattr(agent.state, "task", None)
    if task is not None:
        m_idx = _write_feature(memory_ctx, m_idx, _safe_hash(task.input))
        if task.additional_input:
            m_idx = _write_feature(memory_ctx, m_idx, _safe_hash(str(task.additional_input)))

    goals = list(getattr(agent.ai_profile, "ai_goals", []))
    m_idx = _encode_recent_actions(memory_ctx, m_idx, goals, limit=15)

    directives = getattr(agent.directives, "general_guidelines", None)
    if directives:
        for line in directives[:5]:
            m_idx = _write_feature(memory_ctx, m_idx, _safe_hash(line))

    for episode in episodes[-5:]:
        snippet = episode.summary or episode.action.reasoning
        m_idx = _write_feature(memory_ctx, m_idx, _safe_hash(snippet))

    return observation, memory_ctx
