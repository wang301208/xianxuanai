from __future__ import annotations

"""Self-learning brain module with curiosity-driven updates."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from modules.brain.neuromorphic.spiking_network import NeuromorphicRunResult

try:  # pragma: no cover - fallback if ML dependencies are missing
    from backend.ml.experience_collector import ActiveCuriositySelector
except Exception:  # pragma: no cover - lightweight stand-in for tests
    class ActiveCuriositySelector:  # type: ignore[override]
        """Minimal curiosity selector used when full backend is unavailable."""

        def __init__(
            self, reward_threshold: float = 0.0, novelty_weight: float = 0.5
        ) -> None:
            self.reward_threshold = reward_threshold
            self.novelty_weight = novelty_weight
            self.seen_states: Set[str] = set()
            self.avg_reward = 0.0
            self.count = 0

        def consider(self, sample: Dict[str, Any]) -> bool:
            self.count += 1
            r = sample["reward"]
            self.avg_reward += (r - self.avg_reward) / self.count
            novelty = 0.0 if sample["state"] in self.seen_states else 1.0
            curiosity = self.novelty_weight * novelty + (1 - self.novelty_weight) * max(
                0.0, r - self.avg_reward
            )
            if curiosity > self.reward_threshold:
                self.seen_states.add(sample["state"])
                return True
            return False

try:  # pragma: no cover - use full implementation if available
    from backend.world_model import WorldModel
except Exception:  # pragma: no cover - simplified model for testing
    class WorldModel:  # type: ignore[override]
        """Minimal world model tracking resource usage with EWMA."""

        def __init__(self, alpha: float = 0.5) -> None:
            self.alpha = alpha
            self._predictions: Dict[str, Dict[str, float]] = {}

        def update_resources(self, agent_id: str, usage: Dict[str, float]) -> None:
            prev = self._predictions.get(agent_id)
            if prev is None:
                self._predictions[agent_id] = {
                    "cpu": usage.get("cpu", 0.0),
                    "memory": usage.get("memory", 0.0),
                }
            else:
                self._predictions[agent_id] = {
                    "cpu": self.alpha * usage.get("cpu", 0.0)
                    + (1 - self.alpha) * prev.get("cpu", 0.0),
                    "memory": self.alpha * usage.get("memory", 0.0)
                    + (1 - self.alpha) * prev.get("memory", 0.0),
                }

        def predict(self, agent_id: str) -> Dict[str, float]:
            return self._predictions.get(agent_id, {"cpu": 0.0, "memory": 0.0})


@dataclass
class SelfLearningBrain:
    """Combine curiosity-based selection with a simple world model.

    The brain keeps a lightweight memory of novel states. When an interaction
    sample is considered interesting by :class:`ActiveCuriositySelector`, the
    sample is stored and the :class:`WorldModel` is updated with the observed
    resource usage, correcting its predictions.  The updated prediction can be
    used as an improved policy for future interactions.
    """

    world_model: WorldModel = field(default_factory=WorldModel)
    selector: ActiveCuriositySelector = field(default_factory=ActiveCuriositySelector)
    memory: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rejection_threshold: int = 3
    error_threshold: float = 0.5
    error_window: int = 3
    exploration_flags: Set[str] = field(default_factory=set)
    knowledge_gap_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    knowledge_gap_flags: Set[str] = field(default_factory=set)

    def curiosity_driven_learning(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """Update world model based on curiosity and return new predictions.

        Parameters
        ----------
        sample:
            Mapping with at least ``state``, ``reward``, ``agent_id`` and
            ``usage`` (a dict of resource metrics).  Only samples deemed
            interesting by the selector are used to update the model.

        Returns
        -------
        dict
            The world model's prediction for ``agent_id`` after potential
            updates.
        """

        agent_id = sample.get("agent_id", sample["state"])
        state = sample["state"]
        usage = sample.get("usage", {})
        prediction = self.world_model.predict(agent_id)
        prediction_error = self._compute_prediction_error(prediction, usage)

        entry = self._ensure_memory_entry(state, sample)
        metadata = entry["metadata"]
        metadata["last_error"] = prediction_error
        self._update_error_history(metadata, prediction_error)

        accepted = self.selector.consider(sample)

        if accepted:
            metadata["rejections"] = 0
            if usage:
                # correct world model prediction with observed usage
                self.world_model.update_resources(agent_id, usage)
        else:
            metadata["rejections"] += 1

        self._update_exploration_priority(state, metadata)

        return self.world_model.predict(agent_id)

    def consume_exploration_candidates(self) -> Dict[str, Dict[str, Any]]:
        """Return and clear states flagged for additional exploration."""

        if not self.exploration_flags and not self.knowledge_gap_flags:
            return {}

        def _priority_for(state: str) -> float:
            entry: Dict[str, Any] | None = None
            if state in self.memory:
                entry = self.memory.get(state)
            elif state in self.knowledge_gap_registry:
                entry = self.knowledge_gap_registry.get(state)
            metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
            try:
                return float(metadata.get("priority", 1.0))
            except (TypeError, ValueError):
                return 1.0

        ordered_states = sorted(
            set(self.exploration_flags) | set(self.knowledge_gap_flags),
            key=_priority_for,
            reverse=True,
        )
        candidates: Dict[str, Dict[str, Any]] = {}
        for state in ordered_states:
            entry = self.memory.get(state)
            if entry is None and state in self.knowledge_gap_registry:
                entry = self.knowledge_gap_registry.get(state)
            if not isinstance(entry, dict):
                continue
            entry.setdefault("metadata", {})
            entry["metadata"]["flagged"] = False
            if state in self.exploration_flags:
                self.exploration_flags.discard(state)
            if state in self.knowledge_gap_flags:
                self.knowledge_gap_flags.discard(state)
            if "sample" not in entry or not isinstance(entry["sample"], dict):
                concept = entry["metadata"].get("concept")
                entry["sample"] = {
                    "state": state,
                    "goal_id": f"self-study:{self._normalise_concept(str(concept or state))}",
                    "task": f"Investigate {concept or state}",
                    "reward": 0.0,
                }
            if "goal" not in entry and entry.get("metadata", {}).get("origin") == "knowledge-gap":
                concept = entry["metadata"].get("concept") or state
                entry["goal"] = f"self-study:{self._normalise_concept(str(concept))}"
            candidates[state] = entry
        return candidates

    def _ensure_memory_entry(self, state: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        entry = self.memory.get(state)
        if entry is None:
            entry = {
                "sample": sample,
                "metadata": {
                    "rejections": 0,
                    "error_history": [],
                    "smoothed_error": 0.0,
                    "flagged": False,
                    "last_error": 0.0,
                    "priority": 1.0,
                    "attempts": 0,
                    "failures": 0,
                    "successes": 0,
                    "last_reward": 0.0,
                    "last_outcome": None,
                    "outcome_history": [],
                },
            }
            self.memory[state] = entry
        else:
            entry["sample"] = sample
            entry.setdefault("metadata", {})
            entry["metadata"].setdefault("rejections", 0)
            entry["metadata"].setdefault("error_history", [])
            entry["metadata"].setdefault("smoothed_error", 0.0)
            entry["metadata"].setdefault("flagged", False)
            entry["metadata"].setdefault("last_error", 0.0)
            entry["metadata"].setdefault("priority", 1.0)
            entry["metadata"].setdefault("attempts", 0)
            entry["metadata"].setdefault("failures", 0)
            entry["metadata"].setdefault("successes", 0)
            entry["metadata"].setdefault("last_reward", 0.0)
            entry["metadata"].setdefault("last_outcome", None)
            entry["metadata"].setdefault("outcome_history", [])
        return entry

    def _update_error_history(self, metadata: Dict[str, Any], error: float) -> None:
        history: List[float] = metadata.get("error_history", [])
        history.append(error)
        if len(history) > self.error_window:
            history.pop(0)
        metadata["error_history"] = history
        if history:
            metadata["smoothed_error"] = sum(history) / len(history)

    def _update_exploration_priority(self, state: str, metadata: Dict[str, Any]) -> None:
        should_flag = False
        if metadata["rejections"] >= self.rejection_threshold:
            should_flag = True
        elif (
            len(metadata.get("error_history", [])) >= self.error_window
            and metadata.get("smoothed_error", 0.0) > self.error_threshold
        ):
            should_flag = True

        if should_flag:
            metadata["flagged"] = True
            self.exploration_flags.add(state)

    def record_exploration_outcome(
        self,
        state: str,
        success: bool,
        metrics: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Log the outcome of an exploration attempt and adjust priority.

        Parameters
        ----------
        state:
            Identifier previously produced by :meth:`curiosity_driven_learning`.
        success:
            ``True`` when the exploration achieved its objective.
        metrics:
            Optional mapping containing feedback such as reward/success values.

        Returns
        -------
        dict
            Updated metadata dictionary for ``state``.
        """

        entry = self._ensure_memory_entry(state, {"state": state})
        metadata = entry.setdefault("metadata", {})
        metadata.setdefault("priority", 1.0)
        metadata.setdefault("attempts", 0)
        metadata.setdefault("failures", 0)
        metadata.setdefault("successes", 0)
        metadata.setdefault("outcome_history", [])
        metadata.setdefault("rejections", 0)

        reward = 0.0
        if metrics and isinstance(metrics, Mapping) and "reward" in metrics:
            try:
                reward = float(metrics["reward"])
            except (TypeError, ValueError):
                reward = 0.0

        metadata["attempts"] = int(metadata.get("attempts", 0)) + 1
        metadata["last_reward"] = reward
        metadata["last_outcome"] = bool(success)

        if success:
            metadata["successes"] = int(metadata.get("successes", 0)) + 1
            metadata["rejections"] = 0
            decay_factor = 0.5 if reward > 0 else 0.7
            metadata["priority"] = max(0.1, float(metadata.get("priority", 1.0)) * decay_factor)
            metadata["flagged"] = False
        else:
            metadata["failures"] = int(metadata.get("failures", 0)) + 1
            metadata["rejections"] = int(metadata.get("rejections", 0)) + 1
            failure_boost = 1.0 + (metadata["failures"] / max(metadata["attempts"], 1))
            if reward < 0:
                failure_boost += abs(reward)
            metadata["priority"] = float(metadata.get("priority", 1.0)) + failure_boost
            metadata["flagged"] = True
            self.exploration_flags.add(state)

        history = metadata.get("outcome_history", [])
        history.append({"success": bool(success), "reward": reward})
        if len(history) > max(1, self.error_window):
            history = history[-self.error_window :]
        metadata["outcome_history"] = history

        entry["metadata"] = metadata
        self.memory[state] = entry
        return metadata

    def register_knowledge_gap(
        self,
        concept: str,
        *,
        context: Mapping[str, Any] | None = None,
        reason: str | None = None,
        priority: float | None = None,
    ) -> str | None:
        """Flag a concept for self-directed exploration due to missing knowledge."""

        concept = (concept or "").strip()
        if not concept:
            return None
        state = f"knowledge-gap:{self._normalise_concept(concept)}"
        sample = {
            "state": state,
            "goal_id": f"self-study:{self._normalise_concept(concept)}",
            "task": f"Investigate knowledge gap about {concept}",
            "reward": 0.0,
            "context": dict(context) if isinstance(context, Mapping) else {},
        }
        entry = self._ensure_memory_entry(state, sample)
        metadata = entry.setdefault("metadata", {})
        metadata.setdefault("priority", 1.0)
        priority_increased = False
        if priority is not None:
            try:
                current_priority = float(metadata.get("priority", 1.0))
            except (TypeError, ValueError):
                current_priority = 1.0
            try:
                proposed_priority = float(priority)
            except (TypeError, ValueError):
                proposed_priority = current_priority
            if proposed_priority > current_priority:
                metadata["priority"] = proposed_priority
                priority_increased = True
            else:
                metadata["priority"] = current_priority
        metadata.setdefault("concept", concept)
        metadata.setdefault("origin", "knowledge-gap")
        if reason:
            existing_reasons = metadata.get("reasons")
            if isinstance(existing_reasons, list):
                if reason not in existing_reasons:
                    existing_reasons.append(reason)
                metadata["reasons"] = existing_reasons
            else:
                metadata["reasons"] = [reason]
        if context and isinstance(context, Mapping):
            existing_context = metadata.setdefault("context", {})
            if isinstance(existing_context, Mapping):
                existing_context = dict(existing_context)
            else:
                existing_context = {}
            merged_context = dict(existing_context)
            merged_context.update({k: v for k, v in context.items()})
            metadata["context"] = merged_context
        else:
            metadata.setdefault("context", {})
        was_flagged = bool(metadata.get("flagged", False))
        metadata["flagged"] = True
        entry["metadata"] = metadata
        self.memory[state] = entry
        self.knowledge_gap_registry[state] = entry
        if not was_flagged:
            self.knowledge_gap_flags.add(state)
        elif priority_increased:
            self.knowledge_gap_flags.add(state)
        return state

    @staticmethod
    def _compute_prediction_error(
        prediction: Mapping[str, float], actual: Mapping[str, float]
    ) -> float:
        if not actual:
            return 0.0
        keys = set(prediction.keys()) | set(actual.keys())
        return sum(abs(prediction.get(k, 0.0) - actual.get(k, 0.0)) for k in keys) / max(
            len(keys), 1
        )

    @staticmethod
    def _normalise_concept(concept: str) -> str:
        token = concept.strip().lower()
        normalised = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in token)
        normalised = normalised.strip("-_")
        return normalised or "concept"


    @staticmethod
    def build_neuromorphic_sample(
        label: str,
        run_result: "NeuromorphicRunResult",
        metrics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Create a curiosity-ready sample from neuromorphic telemetry."""

        if isinstance(metrics, Mapping):
            metrics_dict = dict(metrics)
        else:
            metrics_dict = asdict(metrics)  # type: ignore[arg-type]
        spikes = run_result.spike_counts or [0]
        state_signature = "-".join(str(v) for v in spikes) or "0"
        usage = {
            "energy": float(run_result.energy_used),
            "spikes": float(sum(spikes)),
            "idle": float(run_result.idle_skipped),
        }
        reward = 1.0 - float(metrics_dict.get("mse", 0.0)) - 0.5 * float(metrics_dict.get("avg_rate_diff", 0.0))
        reward = max(-1.0, min(1.0, reward))
        sample: Dict[str, Any] = {
            "state": f"{label}:{state_signature}",
            "agent_id": label,
            "usage": usage,
            "reward": reward,
            "metrics": metrics_dict,
        }
        if run_result.average_rate:
            sample["metrics"]["average_rate"] = list(run_result.average_rate)
        sample["metrics"]["spike_counts"] = list(spikes)
        return sample

    def ingest_neuromorphic_metrics(
        self,
        label: str,
        run_result: "NeuromorphicRunResult",
        metrics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Feed neuromorphic evaluation metrics into the curiosity loop."""

        sample = self.build_neuromorphic_sample(label, run_result, metrics)
        self.curiosity_driven_learning(sample)
        return sample


__all__ = ["SelfLearningBrain"]
