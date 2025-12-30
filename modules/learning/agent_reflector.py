"""Agent reflection and strategy-improvement helpers.

This module provides an `AgentReflector` that can:
- Summarize recent outcomes (episodes, metrics, regressions)
- Propose strategy tweaks (heuristic / diagnoser-driven)
- Persist a structured record into `KnowledgeBase` for long-term learning

LLM-backed reflection is intentionally optional; the default implementation is
deterministic and safe for offline/unit-test usage.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from modules.knowledge import KnowledgeBase
except Exception:  # pragma: no cover - optional dependency during minimal installs
    KnowledgeBase = None  # type: ignore[assignment]


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _episode_summary(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(episodes)
    if total == 0:
        return {"count": 0, "success_rate": None, "avg_reward": None, "avg_steps": None}
    successes = 0
    reward_sum = 0.0
    steps_sum = 0.0
    for ep in episodes:
        if bool(ep.get("success", False)):
            successes += 1
        reward_sum += _safe_float(ep.get("total_reward", 0.0))
        steps_sum += _safe_float(ep.get("steps", 0.0))
    return {
        "count": total,
        "success_rate": successes / max(total, 1),
        "avg_reward": reward_sum / max(total, 1),
        "avg_steps": steps_sum / max(total, 1),
    }


def _heuristic_evaluation(
    *,
    success_rate: float | None,
    issue_count: int,
    regression_count: int,
) -> Dict[str, Any]:
    # Confidence is used as a reward signal in the retraining pipeline. Keep it
    # stable and interpretable without any model calls.
    base = 0.55
    if success_rate is not None:
        base = 0.2 + 0.8 * _clamp01(float(success_rate))
    penalty = min(0.35, 0.03 * max(issue_count, 0) + 0.05 * max(regression_count, 0))
    confidence = _clamp01(base - penalty)
    sentiment = "neutral"
    if success_rate is not None and success_rate >= 0.85 and issue_count == 0:
        sentiment = "positive"
    elif success_rate is not None and success_rate < 0.5:
        sentiment = "negative"
    elif issue_count >= 3 or regression_count >= 1:
        sentiment = "negative"
    return {
        "confidence": confidence,
        "sentiment": sentiment,
        "raw": f"heuristic:success_rate={success_rate},issues={issue_count},regressions={regression_count}",
    }


class AgentReflector:
    """Create and persist self-monitoring reflections."""

    def __init__(
        self,
        *,
        knowledge_base: Any | None = None,
        enabled: bool | None = None,
        llm: Optional[Callable[[str], str]] = None,
        max_episodes: int = 12,
        max_issues: int = 12,
        max_regressions: int = 12,
    ) -> None:
        self.enabled = _parse_bool(os.getenv("AGENT_REFLECTOR_ENABLED"), default=True) if enabled is None else bool(enabled)
        self._llm = llm
        self.max_episodes = max(0, int(max_episodes))
        self.max_issues = max(0, int(max_issues))
        self.max_regressions = max(0, int(max_regressions))

        kb = knowledge_base
        if kb is None and KnowledgeBase is not None:
            try:
                kb = KnowledgeBase.from_env()
            except Exception:
                kb = None
        self._kb = kb

    @classmethod
    def from_env(cls) -> "AgentReflector | None":
        if not _parse_bool(os.getenv("AGENT_REFLECTOR_ENABLED"), default=True):
            return None
        return cls()

    @property
    def knowledge_base(self) -> Any | None:
        return self._kb

    def reflect(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Generate a structured reflection record from coordinator payload."""

        now = time.time()
        regressions = payload.get("regressions") if isinstance(payload.get("regressions"), list) else []
        issues = payload.get("issues") if isinstance(payload.get("issues"), list) else []
        episodes = payload.get("episodes") if isinstance(payload.get("episodes"), list) else []
        metrics_summary = payload.get("metrics_summary") if isinstance(payload.get("metrics_summary"), dict) else {}
        strategy = payload.get("strategy_suggestions") if isinstance(payload.get("strategy_suggestions"), dict) else {}

        regressions = list(regressions)[: self.max_regressions]
        issues = list(issues)[: self.max_issues]
        episodes = list(episodes)[: self.max_episodes]

        episode_stats = _episode_summary(episodes)
        success_rate = episode_stats.get("success_rate")
        evaluation = _heuristic_evaluation(
            success_rate=success_rate if isinstance(success_rate, (int, float)) else None,
            issue_count=len(issues),
            regression_count=len(regressions),
        )

        summary_lines: List[str] = []
        summary_lines.append(f"episodes={episode_stats.get('count', 0)}")
        if isinstance(success_rate, (int, float)):
            summary_lines.append(f"success_rate={success_rate:.3f}")
        avg_reward = episode_stats.get("avg_reward")
        if isinstance(avg_reward, (int, float)):
            summary_lines.append(f"avg_reward={avg_reward:.3f}")
        avg_latency = metrics_summary.get("avg_latency")
        if avg_latency is not None:
            summary_lines.append(f"avg_latency={_safe_float(avg_latency):.3f}s")
        if issues:
            kinds = [str(item.get("kind", "issue")) for item in issues if isinstance(item, dict)]
            if kinds:
                summary_lines.append("issues=" + ",".join(kinds[:6]))
        if regressions:
            summary_lines.append(f"regressions={len(regressions)}")
        summary = " | ".join(summary_lines)

        recommendations: List[str] = []
        updates = strategy.get("updates") if isinstance(strategy.get("updates"), dict) else {}
        actions = strategy.get("actions") if isinstance(strategy.get("actions"), list) else []
        for action in actions:
            if not isinstance(action, dict):
                continue
            param = action.get("parameter")
            value = action.get("value")
            reason = action.get("reason")
            if param is None:
                continue
            recommendations.append(f"Set {param}={value} ({reason})")
        if not recommendations and issues:
            recommendations.append("Investigate top issues and adjust strategy parameters.")
        if not recommendations and regressions:
            recommendations.append("Review recent regressions and add a remediation checklist.")
        if not recommendations:
            recommendations.append("Maintain current strategy; continue monitoring.")

        revision = "\n".join(recommendations)

        record: Dict[str, Any] = {
            "timestamp": now,
            "summary": summary,
            "revision": revision,
            "evaluation": evaluation,
            "metrics_summary": dict(metrics_summary),
            "episodes": episodes,
            "issues": issues,
            "regressions": regressions,
            "strategy_updates": dict(updates),
        }

        if self._llm is not None:
            try:  # pragma: no cover - optional LLM refinement
                prompt = (
                    "Summarize the agent self-monitoring record and propose concise improvements.\n"
                    "Return JSON: {\"summary\": str, \"revision\": str}.\n\n"
                    f"Input:\n{_json_dumps(record)}"
                )
                raw = (self._llm(prompt) or "").strip()
                data = json.loads(raw) if raw.startswith("{") else {}
                if isinstance(data, dict):
                    if isinstance(data.get("summary"), str) and data["summary"].strip():
                        record["summary"] = data["summary"].strip()
                    if isinstance(data.get("revision"), str) and data["revision"].strip():
                        record["revision"] = data["revision"].strip()
            except Exception:
                pass

        memory_id = None
        kb = self._kb
        if self.enabled and kb is not None and hasattr(kb, "save_memory"):
            try:
                memory_id = kb.save_memory(
                    "self_monitoring",
                    _json_dumps(record),
                    tags=["self_monitoring", "reflection"],
                    metadata={"source": "agent_reflector"},
                )
            except Exception:
                memory_id = None
        record["memory_id"] = memory_id
        return record


def build_reflection_callback(
    reflector: AgentReflector,
) -> Callable[[Dict[str, Any]], Any]:
    def _callback(payload: Dict[str, Any]) -> Any:
        reflector.reflect(payload)
        return True

    return _callback


def normalise_strategy_suggestions(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Make `StrategyAdjuster.propose()` output JSON-friendly."""

    updates = payload.get("updates") if isinstance(payload.get("updates"), dict) else {}
    actions_raw = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    actions: List[Dict[str, Any]] = []
    for action in actions_raw:
        if isinstance(action, dict):
            actions.append(dict(action))
            continue
        try:
            actions.append(asdict(action))
        except Exception:
            continue
    return {"updates": dict(updates), "actions": actions}


__all__ = ["AgentReflector", "build_reflection_callback", "normalise_strategy_suggestions"]

