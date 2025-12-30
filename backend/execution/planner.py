"""Simple planner to decompose high level goals into executable sub-tasks."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import re
from typing import List, Optional

from backend.creative_engine.problem_solver import DivergentConvergentSolver
from modules.evolution.evolution_engine import (
    SpecialistModuleRegistry,
    TaskContext,
)


CapabilitySource = str | Sequence[str] | Mapping[str, Iterable[str]]


def _normalise_capabilities(values: CapabilitySource | None) -> List[str]:
    """Return a flat list of normalised capability identifiers."""

    capabilities: list[str] = []
    if values is None:
        return capabilities
    if isinstance(values, str):
        tokens = re.split(r"[,;\s]+", values)
        capabilities.extend(token.strip() for token in tokens if token.strip())
        return capabilities
    if isinstance(values, Mapping):
        for token in values.values():
            capabilities.extend(_normalise_capabilities(token))
        return capabilities
    for token in values:
        if token is None:
            continue
        if isinstance(token, str):
            token = token.strip()
            if token:
                capabilities.append(token)
        else:
            capabilities.extend(_normalise_capabilities(token))
    return capabilities


_CAPABILITY_PATTERN = re.compile(
    r"\[(?:capability|capabilities):([^\]]+)\]", re.IGNORECASE
)


def _strip_capability_annotations(text: str) -> str:
    """Remove inline capability annotations from ``text``."""

    return _CAPABILITY_PATTERN.sub("", text)


def _extract_capabilities_from_text(text: str) -> List[str]:
    """Extract capability tokens embedded within ``text`` annotations."""

    matches = _CAPABILITY_PATTERN.findall(text)
    capabilities: list[str] = []
    for value in matches:
        capabilities.extend(_normalise_capabilities(value))
    return capabilities


TaskContextBuilder = Callable[[str | dict], Optional[TaskContext]]


def _default_task_context_builder(goal: str | dict) -> Optional[TaskContext]:
    """Generate a :class:`TaskContext` from ``goal`` when annotations exist."""

    metadata: Mapping[str, object] | None = None
    name = ""
    capabilities: list[str] = []

    if isinstance(goal, str):
        name = _strip_capability_annotations(goal).strip()
        capabilities.extend(_extract_capabilities_from_text(goal))
    elif isinstance(goal, Mapping):
        metadata = goal.get("metadata")  # type: ignore[assignment]
        goal_name = goal.get("goal") or goal.get("name")
        if isinstance(goal_name, str):
            name = _strip_capability_annotations(goal_name).strip()
            capabilities.extend(_extract_capabilities_from_text(goal_name))
        cap_sources: list[CapabilitySource | None] = [
            goal.get("required_capabilities"),
            goal.get("capabilities"),
        ]
        if isinstance(metadata, Mapping):
            cap_sources.extend(
                [
                    metadata.get("required_capabilities"),
                    metadata.get("capabilities"),
                ]
            )
        strategies = goal.get("strategies")
        if isinstance(strategies, Sequence):
            for strategy in strategies:
                if isinstance(strategy, Mapping):
                    cap_sources.append(strategy.get("capabilities"))
                    cap_sources.append(strategy.get("required_capabilities"))
                    title = strategy.get("name") or strategy.get("description")
                    if isinstance(title, str):
                        capabilities.extend(_extract_capabilities_from_text(title))
        for source in cap_sources:
            capabilities.extend(_normalise_capabilities(source))
    if not capabilities:
        return None
    unique_caps = tuple(dict.fromkeys(cap.lower() for cap in capabilities if cap))
    if not unique_caps:
        return None
    metadata_mapping = metadata if isinstance(metadata, Mapping) else None
    context_name = name or (
        _strip_capability_annotations(goal).strip()
        if isinstance(goal, str)
        else "goal"
    )
    return TaskContext(
        name=str(context_name),
        required_capabilities=unique_caps,
        metadata=metadata_mapping,
    )


class Planner:
    """Decompose high level goals into ordered sub-tasks."""

    def __init__(
        self,
        solver: DivergentConvergentSolver | None = None,
        registry: SpecialistModuleRegistry | None = None,
        task_context_builder: TaskContextBuilder | None = None,
    ) -> None:
        self.solver = solver
        self.registry = registry
        self._task_context_builder = task_context_builder or _default_task_context_builder

    def decompose(self, goal: str, source: str | None = None) -> List[str]:
        """Return a list of sub-tasks derived from a high level goal.

        Parameters
        ----------
        goal:
            The objective to break down into smaller tasks.
        source:
            Optional tag describing where the goal originated.  When
            provided, each resulting task is annotated to retain this
            provenance information.

        The default implementation uses a few heuristic separators to
        break the goal into manageable pieces. It can be replaced with a
        more sophisticated planner or LLM powered approach in the future.
        """
        if not goal:
            return []
        goal = _strip_capability_annotations(goal)
        # Replace common separators with newlines to unify splitting logic
        separators = ["\n", ";", " and ", " then "]
        normalized = goal
        for sep in separators[1:]:
            normalized = normalized.replace(sep, "\n")
        tasks = [task.strip() for task in normalized.splitlines() if task.strip()]
        if source:
            tasks = [f"{task} [{source}]" for task in tasks]
        return tasks

    def _build_task_context(self, goal: str | dict) -> Optional[TaskContext]:
        if not self._task_context_builder:
            return None
        return self._task_context_builder(goal)

    def _plan_with_specialist(self, context: TaskContext) -> Optional[List[str]]:
        if not self.registry:
            return None
        specialist = self.registry.select_best(context)
        if specialist is None:
            return None
        specialist_plan: list[str] = []
        try:
            result = specialist.solver({}, context)
        except Exception:
            result = None
        if isinstance(result, Mapping):
            for key, value in result.items():
                description = value if isinstance(value, str) else str(value)
                prefix = f"[{specialist.name}]"
                if key:
                    specialist_plan.append(f"{prefix} {key}: {description}")
                else:
                    specialist_plan.append(f"{prefix} {description}")
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            for entry in result:
                description = entry if isinstance(entry, str) else str(entry)
                specialist_plan.append(f"[{specialist.name}] {description}")
        elif isinstance(result, str):
            specialist_plan.append(f"[{specialist.name}] {result}")
        if not specialist_plan:
            specialist_plan.append(
                f"[{specialist.name}] Execute specialist plan for '{context.name}'"
            )
        return specialist_plan

    def solve(self, goal: str | dict) -> List[str]:
        """Return a plan for ``goal``.

        When ``goal`` specifies multiple strategies via a mapping with
        ``start``, ``goal`` and ``strategies`` keys, the
        :class:`DivergentConvergentSolver` is used to pick the best path.
        Otherwise the goal string is simply decomposed into sub-tasks.
        """

        context = self._build_task_context(goal)
        if context:
            specialist_plan = self._plan_with_specialist(context)
            if specialist_plan:
                return specialist_plan

        if isinstance(goal, dict) and goal.get("strategies"):
            if not self.solver:
                raise ValueError("No solver configured")
            start = goal.get("start", "")
            target = goal.get("goal", "")
            strategies = goal.get("strategies", [])
            best_path, _ = self.solver.solve(start, target, strategies)
            return best_path
        if isinstance(goal, str):
            fallback_goal = goal
        elif isinstance(goal, Mapping):
            fallback_goal = goal.get("goal", "")  # type: ignore[assignment]
        else:
            fallback_goal = ""
        return self.decompose(str(fallback_goal))
