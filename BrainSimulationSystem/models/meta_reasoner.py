"""
Meta reasoning utilities for validating plans and decisions.

Implements reflective checks such as counterfactual reasoning, majority voting
over multiple reasoning chains, and chain-of-thought summarisation stubs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict, deque
import math
import time


@dataclass
class MetaReasonerConfig:
    """Configuration for meta reasoning checks."""

    enable_counterfactual: bool = True
    enable_consistency: bool = True
    enable_self_reflection: bool = True
    reflection_threshold: float = 0.3


class MetaReasoner:
    """Apply meta-reasoning passes to evaluate plan reliability."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = MetaReasonerConfig(**(config or {}))
        self._strategy_stats: Dict[str, Dict[str, StrategyPerformance]] = defaultdict(dict)

    def evaluate(
        self,
        plan: Dict[str, Any],
        decision: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {}

        if self.config.enable_consistency:
            analysis["consistency"] = self._evaluate_consistency(plan)

        if self.config.enable_counterfactual:
            analysis["counterfactuals"] = self._counterfactual_analysis(plan, decision, context)

        if self.config.enable_self_reflection:
            analysis["reflection"] = self._self_reflection(decision)

        analysis["reliability_score"] = self._aggregate_scores(analysis)
        return analysis

    def record_strategy_outcome(
        self,
        task_signature: str,
        strategy: str,
        *,
        reward: Optional[float] = None,
        success: Optional[bool] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log the performance of ``strategy`` for ``task_signature``."""

        stats = self._strategy_stats[task_signature].setdefault(
            strategy, StrategyPerformance()
        )
        stats.update(reward=reward, success=success, confidence=confidence, metadata=metadata)

    def recommend_strategy(
        self,
        task_signature: str,
        candidates: Iterable[str],
        default: Optional[str] = None,
    ) -> str:
        """Select the most promising strategy for ``task_signature``."""

        available = list(candidates)
        if not available:
            raise ValueError("No candidate strategies provided")

        stats_map = self._strategy_stats.get(task_signature, {})
        ranked = sorted(
            (
                (name, stats_map.get(name))
                for name in available
            ),
            key=lambda pair: pair[1].score if pair[1] is not None else float("-inf"),
            reverse=True,
        )
        if ranked and ranked[0][1] is not None and ranked[0][1].trials > 0:
            return ranked[0][0]
        return default or available[0]

    def summarise_strategies(self, task_signature: str) -> Dict[str, Any]:
        """Return performance summary for strategies associated with ``task_signature``."""

        stats_map = self._strategy_stats.get(task_signature, {})
        return {name: stats.as_dict() for name, stats in stats_map.items()}

    def _evaluate_consistency(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        evaluations = plan.get("evaluations", [])
        if not evaluations:
            return {"score": 0.5, "notes": "no evaluations supplied"}

        scores = [entry.get("score", 0.0) for entry in evaluations]
        mean_score = sum(scores) / max(len(scores), 1)
        variance = sum((s - mean_score) ** 2 for s in scores) / max(len(scores), 1)
        return {"score": float(mean_score), "variance": float(variance)}

    def _counterfactual_analysis(
        self,
        plan: Dict[str, Any],
        decision: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidates = plan.get("candidates", [])
        if len(candidates) < 2:
            return {"alternatives_considered": 0, "notes": "insufficient candidates"}

        chosen = decision.get("decision")
        alternative_scores = []
        for candidate in candidates:
            if candidate == chosen:
                continue
            justification_len = len(candidate.get("justification", []))
            alternative_scores.append(justification_len * 0.1)

        diversity = len(alternative_scores)
        avg = sum(alternative_scores) / max(diversity, 1)
        return {
            "alternatives_considered": diversity,
            "average_support": float(avg),
        }

    def _self_reflection(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        confidence = decision.get("confidence", 0.5)
        requires_review = confidence < self.config.reflection_threshold
        return {
            "confidence": float(confidence),
            "requires_review": requires_review,
        }

    def _aggregate_scores(self, analysis: Dict[str, Any]) -> float:
        components = []
        consistency = analysis.get("consistency")
        if isinstance(consistency, dict):
            components.append(consistency.get("score", 0.5))
        counter = analysis.get("counterfactuals")
        if isinstance(counter, dict):
            components.append(counter.get("average_support", 0.4))
        reflection = analysis.get("reflection")
        if isinstance(reflection, dict):
            confidence = reflection.get("confidence", 0.5)
            components.append(confidence if not reflection.get("requires_review") else confidence / 2)

        if not components:
            return 0.5
        return float(sum(components) / len(components))


@dataclass
class StrategyPerformance:
    """Track historical performance of a reasoning or execution strategy."""

    rewards: deque = field(default_factory=lambda: deque(maxlen=50))
    confidences: deque = field(default_factory=lambda: deque(maxlen=50))
    successes: int = 0
    trials: int = 0
    last_reward: Optional[float] = None
    last_confidence: Optional[float] = None
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        *,
        reward: Optional[float],
        success: Optional[bool],
        confidence: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trials += 1
        if success:
            self.successes += 1
        if reward is not None:
            value = float(reward)
            self.rewards.append(value)
            self.last_reward = value
        if confidence is not None:
            value = float(confidence)
            self.confidences.append(value)
            self.last_confidence = value
        if metadata:
            self.metadata.update(metadata)
        self.last_updated = time.time()

    @property
    def success_rate(self) -> float:
        if self.trials == 0:
            return 0.0
        return self.successes / self.trials

    @property
    def avg_reward(self) -> float:
        if not self.rewards:
            return 0.0
        return float(sum(self.rewards) / len(self.rewards))

    @property
    def avg_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return float(sum(self.confidences) / len(self.confidences))

    @property
    def score(self) -> float:
        """Overall score balancing success, reward, and confidence."""

        success_component = 0.6 * self.success_rate
        reward_component = 0.3 * math.tanh(self.avg_reward)
        confidence_component = 0.1 * self.avg_confidence
        recency_penalty = self._recency_penalty()
        return (success_component + reward_component + confidence_component) * recency_penalty

    def _recency_penalty(self) -> float:
        elapsed = max(0.0, time.time() - self.last_updated)
        # Decay score by up to 30% over six hours of inactivity.
        decay = min(0.3, (elapsed / 21600.0) * 0.3)
        return 1.0 - decay

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trials": self.trials,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "avg_confidence": self.avg_confidence,
            "last_reward": self.last_reward,
            "last_confidence": self.last_confidence,
            "score": self.score,
        }
