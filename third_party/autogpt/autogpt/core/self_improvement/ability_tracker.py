"""Ability-oriented scoring utilities for self-improvement."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


@dataclass
class AbilityWeakSpot:
    """Descriptor for a repeatedly underperforming capability."""

    name: str
    streak: int
    latest_score: float
    trend: float | None = None


class AbilityScoreTracker:
    """Track ability-centric performance trends and flag weak spots."""

    _BENCHMARK_KEYWORDS: dict[str, tuple[str, ...]] = {
        "creativity": ("creativity", "novel", "divergent"),
        "logic_reasoning": (
            "logic",
            "reason",
            "math",
            "analysis",
            "deduction",
            "puzzle",
        ),
        "planning": ("plan", "strategy", "planning", "schedule", "roadmap"),
        "knowledge": ("knowledge", "retrieval", "facts", "research", "memory"),
        "tool_usage": ("tool", "api", "integration", "browser"),
        "multimodal": ("vision", "image", "audio", "multimodal"),
    }

    def __init__(
        self,
        history_path: Path,
        *,
        logger: logging.Logger | None = None,
        low_score_threshold: float = 0.6,
        streak_length: int = 3,
        history_limit: int = 50,
        latency_normaliser: float = 1.0,
    ) -> None:
        self._history_path = history_path
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logger or logging.getLogger(__name__)
        self._low_score_threshold = max(0.0, min(1.0, low_score_threshold))
        self._streak_length = max(1, streak_length)
        self._history_limit = max(1, history_limit)
        self._latency_normaliser = max(latency_normaliser, 1e-6)

    # Public API ---------------------------------------------------------
    def update(
        self,
        evaluation_summary: dict[str, float] | None = None,
        benchmark_results: Any | None = None,
    ) -> dict[str, Any] | None:
        """Append a new ability snapshot and return the latest report."""

        if evaluation_summary is None:
            evaluation_summary = {}
        elif hasattr(evaluation_summary, "summary") and callable(
            evaluation_summary.summary  # type: ignore[attr-defined]
        ):
            evaluation_summary = evaluation_summary.summary()  # type: ignore[assignment]

        benchmark_payload = self._normalise_benchmark_payload(benchmark_results)
        ability_scores = self._compute_scores(evaluation_summary, benchmark_payload)
        if not ability_scores:
            # Even if we have no new scores, expose existing history if available.
            return self._build_report(self._load_history())

        history = self._load_history()
        history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "scores": ability_scores,
            }
        )
        history = history[-self._history_limit :]
        self._save_history(history)
        return self._build_report(history)

    def latest_report(self) -> dict[str, Any] | None:
        """Return the most recent ability report without mutating history."""

        history = self._load_history()
        if not history:
            return None
        return self._build_report(history)

    # Internal helpers ---------------------------------------------------
    def _load_history(self) -> list[dict[str, Any]]:
        if not self._history_path.exists():
            return []
        try:
            return json.loads(self._history_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - best effort logging
            self._logger.exception("Failed to load ability history")
            return []

    def _save_history(self, history: list[dict[str, Any]]) -> None:
        try:
            payload = json.dumps(history, ensure_ascii=False, indent=2)
            self._history_path.write_text(payload, encoding="utf-8")
        except Exception:  # pragma: no cover - best effort logging
            self._logger.exception("Failed to persist ability history")

    def _normalise_benchmark_payload(self, payload: Any | None) -> dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, (str, Path)):
            try:
                data = json.loads(Path(payload).read_text(encoding="utf-8"))
            except Exception:
                self._logger.exception("Unable to read benchmark payload", path=str(payload))
                return {}
            return data if isinstance(data, dict) else {}
        if isinstance(payload, dict):
            return payload
        self._logger.debug("Unsupported benchmark payload type", type=type(payload))
        return {}

    def _compute_scores(
        self,
        evaluation_summary: dict[str, float],
        benchmark_payload: dict[str, Any],
    ) -> dict[str, float]:
        ability_scores: dict[str, float] = {}
        ability_scores.update(self._scores_from_evaluation(evaluation_summary))
        benchmark_scores = self._scores_from_benchmarks(benchmark_payload)
        for ability, score in benchmark_scores.items():
            if ability in ability_scores:
                ability_scores[ability] = (ability_scores[ability] + score) / 2
            else:
                ability_scores[ability] = score
        return ability_scores

    def _scores_from_evaluation(self, summary: dict[str, float]) -> dict[str, float]:
        if not summary:
            return {}
        precision = summary.get("precision")
        recall = summary.get("recall")
        latency = summary.get("latency")
        fairness = summary.get("fairness")

        scores: dict[str, float] = {}
        if precision is not None or recall is not None:
            components = [c for c in (precision, recall) if isinstance(c, (int, float))]
            if components:
                scores["logic_reasoning"] = self._clamp(sum(components) / len(components))
                scores["reliability"] = self._clamp(
                    (components[0] if len(components) == 1 else sum(components) / len(components))
                )

        if isinstance(latency, (int, float)):
            # Smaller latency is better; normalise against expected ceiling.
            scaled = 1.0 - min(1.0, max(latency, 0.0) / self._latency_normaliser)
            scores["execution_efficiency"] = self._clamp(scaled)

        if isinstance(fairness, (int, float)):
            # fairness metric represents disparity; lower is better.
            scores["fairness_alignment"] = self._clamp(1.0 - max(fairness, 0.0))

        return scores

    def _scores_from_benchmarks(self, payload: dict[str, Any]) -> dict[str, float]:
        tests: Iterable[tuple[str, dict[str, Any]]] = []
        if "tests" in payload and isinstance(payload["tests"], dict):
            tests = payload["tests"].items()

        aggregated: dict[str, list[float]] = {}
        for test_id, metadata in tests:
            if not isinstance(metadata, dict):
                continue
            ability = self._categorise_benchmark(test_id)
            if ability is None:
                continue
            score = self._extract_benchmark_score(metadata)
            aggregated.setdefault(ability, []).append(score)

        averaged: dict[str, float] = {}
        for ability, scores in aggregated.items():
            if scores:
                averaged[ability] = self._clamp(sum(scores) / len(scores))
        return averaged

    def _categorise_benchmark(self, test_id: str) -> str | None:
        lowered = test_id.lower()
        for ability, keywords in self._BENCHMARK_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return ability
        if "benchmark" in lowered or "test" in lowered:
            return "general_problem_solving"
        return None

    def _extract_benchmark_score(self, metadata: dict[str, Any]) -> float:
        if "score" in metadata and isinstance(metadata["score"], (int, float)):
            return self._clamp(float(metadata["score"]))
        outcome = str(metadata.get("outcome", "")).lower()
        if outcome in {"passed", "success", "succeeded"}:
            return 1.0
        if outcome in {"failed", "failure"}:
            return 0.0
        # fall back to success boolean if present
        success = metadata.get("success")
        if isinstance(success, bool):
            return 1.0 if success else 0.0
        return 0.5

    def _build_report(self, history: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not history:
            return None

        latest = history[-1]
        latest_scores: dict[str, float] = latest.get("scores", {})
        weak_spots = self._identify_weak_spots(history)
        return {
            "history": history,
            "scores": latest_scores,
            "weak_abilities": [weak.__dict__ for weak in weak_spots],
        }

    def _identify_weak_spots(self, history: list[dict[str, Any]]) -> list[AbilityWeakSpot]:
        if len(history) < self._streak_length:
            return []

        weak_spots: list[AbilityWeakSpot] = []
        latest = history[-1].get("scores", {})

        abilities = {ability for entry in history for ability in entry.get("scores", {})}
        for ability in abilities:
            streak = 0
            recent_scores: list[float] = []
            for entry in reversed(history):
                score = entry.get("scores", {}).get(ability)
                if score is None:
                    break
                recent_scores.append(score)
                if score < self._low_score_threshold:
                    streak += 1
                else:
                    break
            if streak >= self._streak_length:
                trend = None
                if len(recent_scores) >= 2:
                    trend = recent_scores[0] - recent_scores[1]
                weak_spots.append(
                    AbilityWeakSpot(
                        name=ability,
                        streak=streak,
                        latest_score=latest.get(ability, 0.0),
                        trend=trend,
                    )
                )
        return weak_spots

    @staticmethod
    def _clamp(value: float) -> float:
        if not isinstance(value, (int, float)):
            return 0.0
        return max(0.0, min(1.0, float(value)))

