"""Runtime curriculum controller for scalable brain architectures."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from BrainSimulationSystem.config.stage_profiles import (
    StageRuntimePolicy,
    StageSpec,
    build_stage_config,
    get_stage_spec,
    list_stages,
)

if TYPE_CHECKING:  # pragma: no cover - optional dependencies
    from modules.monitoring.collector import RealTimeMetricsCollector, MetricEvent
    from modules.evolution.dynamic_architecture import DynamicArchitectureExpander
    from modules.evolution.self_evolving_ai_architecture import SelfEvolvingAIArchitecture
else:  # pragma: no cover - typing-only aliases
    RealTimeMetricsCollector = Any
    MetricEvent = Any
    DynamicArchitectureExpander = Any
    SelfEvolvingAIArchitecture = Any


@dataclass(frozen=True)
class StageTransition:
    """Description of a curriculum stage change."""

    previous: StageSpec
    current: StageSpec
    reason: str
    summary: Dict[str, float]


class CurriculumStageManager:
    """Monitor metrics, grow architecture modules, and trigger stage upgrades."""

    def __init__(
        self,
        *,
        collector: RealTimeMetricsCollector | None = None,
        expander: DynamicArchitectureExpander | None = None,
        architecture: SelfEvolvingAIArchitecture | None = None,
        stages: Sequence[str] | None = None,
        starting_stage: str | None = None,
    ) -> None:
        self._collector = collector
        self._expander = expander
        self._architecture = architecture
        self._stage_keys: List[str] = [
            get_stage_spec(key).key for key in (stages or list_stages())
        ]
        self._stage_index = self._resolve_initial_stage(starting_stage)
        self._event_cursor = 0
        self._window: Deque[Tuple[float, bool]] = deque()
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self._last_transition: StageTransition | None = None

    # ------------------------------------------------------------------ #
    @property
    def current_stage(self) -> StageSpec:
        """Stage spec currently in effect."""

        key = self._stage_keys[self._stage_index]
        return get_stage_spec(key)

    # ------------------------------------------------------------------ #
    def current_config(self) -> Dict[str, Any]:
        """Return a copy of the configuration for the current stage."""

        key = self.current_stage.key
        if key not in self._config_cache:
            self._config_cache[key] = build_stage_config(
                key, base_profile=self.current_stage.base_profile
            )
        return deepcopy(self._config_cache[key])

    # ------------------------------------------------------------------ #
    def ingest_events(
        self, events: Sequence[MetricEvent]
    ) -> StageTransition | None:
        """Manually feed metric events (useful for tests or offline runs)."""

        if not events:
            return None
        return self._process_events(events)

    # ------------------------------------------------------------------ #
    def step(self) -> StageTransition | None:
        """Poll the collector and perform a curriculum update if ready."""

        events = self._pull_events()
        if not events:
            return None
        return self._process_events(events)

    # ------------------------------------------------------------------ #
    def _process_events(
        self, events: Sequence[MetricEvent]
    ) -> StageTransition | None:
        """Update performance windows and decide on transitions."""

        policy = self.current_stage.runtime_policy
        for event in events:
            latency = float(getattr(event, "latency", 0.0))
            success = self._is_success(event)
            self._window.append((latency, success))
            if len(self._window) > policy.promotion_window:
                self._window.popleft()

        self._maybe_expand_architecture(events, policy)
        if self._should_promote(policy):
            return self._promote("performance window satisfied")
        return None

    # ------------------------------------------------------------------ #
    def _pull_events(self) -> List[MetricEvent]:
        """Fetch new events from the collector (if any)."""

        if self._collector is None:
            return []
        all_events = list(self._collector.events())
        if not all_events or self._event_cursor >= len(all_events):
            return []
        new_events = all_events[self._event_cursor :]
        self._event_cursor = len(all_events)
        return new_events

    # ------------------------------------------------------------------ #
    def _should_promote(self, policy: StageRuntimePolicy) -> bool:
        """Return ``True`` if the promotion window meets policy criteria."""

        if len(self._window) < policy.promotion_window:
            return False
        latencies = [lat for lat, _ in self._window]
        successes = sum(1 for _, ok in self._window if ok)
        success_rate = successes / len(self._window)
        latency_ok = max(latencies) <= policy.max_latency_s
        return latency_ok and success_rate >= policy.min_success_rate

    # ------------------------------------------------------------------ #
    def _maybe_expand_architecture(
        self,
        events: Sequence[MetricEvent],
        policy: StageRuntimePolicy,
    ) -> None:
        """Trigger dynamic expansion when persistent bottlenecks appear."""

        if self._expander is None or not events:
            return
        module_latencies: Dict[str, List[float]] = defaultdict(list)
        for event in events:
            module = getattr(event, "module", "unknown")
            module_latencies[module].append(float(getattr(event, "latency", 0.0)))

        if not module_latencies:
            return

        module, samples = max(
            module_latencies.items(),
            key=lambda item: sum(item[1]) / max(len(item[1]), 1),
        )
        avg_latency = sum(samples) / max(len(samples), 1)
        if avg_latency <= policy.bottleneck_latency_s:
            return

        performance_ratio = policy.max_latency_s / max(avg_latency, 1e-6)
        self._expander.auto_expand(
            performance=performance_ratio,
            env_feedback=None,
            threshold=0.95,
        )

    # ------------------------------------------------------------------ #
    def _promote(self, reason: str) -> StageTransition | None:
        """Advance to the next stage if available."""

        if self._stage_index >= len(self._stage_keys) - 1:
            return None

        previous = self.current_stage
        self._stage_index += 1
        self._window.clear()
        current = self.current_stage
        self._notify_evolving_architecture(current)

        summary = {
            "window_success_rate": self._window_success_rate(),
            "window_max_latency": self._window_peak_latency(),
        }

        transition = StageTransition(
            previous=previous,
            current=current,
            reason=reason,
            summary=summary,
        )
        self._last_transition = transition
        return transition

    # ------------------------------------------------------------------ #
    def _notify_evolving_architecture(self, spec: StageSpec) -> None:
        """Send a genome update to the self-evolving architecture."""

        if self._architecture is None:
            return
        genome = self._stage_genome(spec)
        self._architecture.update_architecture(
            genome, metrics={"stage": spec.key, "label": spec.label}
        )

    # ------------------------------------------------------------------ #
    def _stage_genome(self, spec: StageSpec) -> Dict[str, float]:
        """Encode the stage scaling factors as a genome dictionary."""

        genome = {
            "stage_index": float(self._stage_index),
            "global_volume_scale": spec.default_scaling.volume,
            "global_density_scale": spec.default_scaling.neuron_density,
        }
        for region, scaling in spec.region_overrides.items():
            genome[f"{region.value}_volume_scale"] = scaling.volume
            genome[f"{region.value}_density_scale"] = scaling.neuron_density
        return genome

    # ------------------------------------------------------------------ #
    def _window_success_rate(self) -> float:
        if not self._window:
            return 0.0
        successes = sum(1 for _, ok in self._window if ok)
        return successes / max(len(self._window), 1)

    # ------------------------------------------------------------------ #
    def _window_peak_latency(self) -> float:
        if not self._window:
            return 0.0
        return max(lat for lat, _ in self._window)

    # ------------------------------------------------------------------ #
    def _is_success(self, event: MetricEvent) -> bool:
        """Best-effort conversion of event metadata into a boolean."""

        status = getattr(event, "status", None)
        if isinstance(status, str):
            lowered = status.lower()
            if lowered in {"success", "ok", "pass", "completed"}:
                return True
            if lowered in {"failure", "fail", "error"}:
                return False
        _sentinel = object()
        prediction = getattr(event, "prediction", _sentinel)
        actual = getattr(event, "actual", None)
        if prediction is not _sentinel and actual is not None:
            return prediction == actual
        return True

    # ------------------------------------------------------------------ #
    def _resolve_initial_stage(self, starting_stage: str | None) -> int:
        """Resolve the numeric index of the starting stage."""

        if starting_stage is None:
            return 0
        target = get_stage_spec(starting_stage).key
        if target not in self._stage_keys:
            available = ", ".join(self._stage_keys)
            raise ValueError(
                f"Stage '{starting_stage}' is not part of active curriculum ({available})"
            )
        return self._stage_keys.index(target)


__all__ = ["CurriculumStageManager", "StageTransition"]
