"""Autonomous task generation, coordination, and memory consolidation utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass
class GeneratedTask:
    """Representation of a self-generated task."""

    description: str
    priority: float = 0.5
    reason: str = "curiosity"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutonomousTaskGenerator:
    """Produce self-directed tasks when idle or seeking novelty."""

    def __init__(self, curiosity_drive: float = 0.5) -> None:
        self.curiosity_drive = curiosity_drive
        self.generated: List[GeneratedTask] = []

    def generate(self, state: Optional[Dict[str, Any]] = None) -> List[GeneratedTask]:
        """Return a list of candidate tasks based on current state."""

        state = state or {}
        reason = "curiosity" if state.get("idle", False) else "maintenance"
        novelty = state.get("novelty_score", self.curiosity_drive)
        task = GeneratedTask(
            description=(
                "Explore new data source" if reason == "curiosity" else "Refine recent skill"
            ),
            priority=min(1.0, 0.5 + novelty * 0.5),
            reason=reason,
            metadata={"novelty": novelty},
        )
        self.generated.append(task)
        return [task]

    def generate_from_context(
        self,
        knowledge_gaps: Optional[List[str]] = None,
        unresolved_tasks: Optional[List[Dict[str, Any]]] = None,
        env_changes: Optional[List[str]] = None,
    ) -> List[GeneratedTask]:
        """Derive tasks from knowledge gaps, unresolved items, or environment changes."""

        tasks: List[GeneratedTask] = []
        for gap in knowledge_gaps or []:
            tasks.append(
                GeneratedTask(
                    description=f"Research and fill knowledge gap: {gap}",
                    priority=0.8,
                    reason="knowledge_gap",
                    metadata={"gap": gap},
                )
            )
        for item in unresolved_tasks or []:
            desc = item.get("description", "Resolve pending task")
            tasks.append(
                GeneratedTask(
                    description=desc,
                    priority=float(item.get("priority", 0.6)),
                    reason=item.get("reason", "unresolved"),
                    metadata=item,
                )
            )
        for change in env_changes or []:
            tasks.append(
                GeneratedTask(
                    description=f"Investigate environment change: {change}",
                    priority=0.7,
                    reason="environment_change",
                    metadata={"change": change},
                )
            )
        for task in tasks:
            task.priority = min(1.0, max(0.1, task.priority))
        self.generated.extend(tasks)
        return tasks


class MemoryConsolidator:
    """Periodically summarise and consolidate long-term memory."""

    def __init__(
        self,
        knowledge_importer: Any | None = None,
        vector_store: Any | None = None,
    ) -> None:
        self.knowledge_importer = knowledge_importer
        self.vector_store = vector_store
        self.consolidations: List[Dict[str, Any]] = []

    def consolidate(self, snapshots: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Perform a consolidation pass and optionally ingest into knowledge/memory stores."""

        snapshots = snapshots or []
        summary_text = self._summarise(snapshots)
        vector_id = None
        if self.vector_store is not None and summary_text:
            try:
                vector_id = self.vector_store.add_text(
                    summary_text, metadata={"source": "consolidation"}
                )
            except Exception:
                vector_id = None
        if self.knowledge_importer is not None and snapshots:
            try:
                self.knowledge_importer.ingest_facts(snapshots)
            except Exception:
                pass
        summary = {
            "items_processed": len(snapshots),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "vector_id": vector_id,
        }
        self.consolidations.append(summary)
        return summary

    def _summarise(self, snapshots: List[Dict[str, Any]]) -> str:
        """Simple heuristic summarisation of snapshot metadata."""

        lines: List[str] = []
        for snap in snapshots:
            parts = [f"{k}: {v}" for k, v in snap.items()]
            lines.append("; ".join(parts))
        return "\n".join(lines)


class _BasicReflection:
    """Minimal reflection helper used when no meta-learning module is provided."""

    def __init__(self) -> None:
        self.history: List[str] = []

    def assess(self, loss_before: float, loss_after: float) -> str:
        delta = loss_before - loss_after
        message = f"loss changed by {delta:.4f}"
        self.history.append(message)
        return message


__all__ = [
    "AgentProfile",
    "AutonomousTaskGenerator",
    "CognitiveEngineBridge",
    "GeneratedTask",
    "GoalRefinementLoop",
    "MemoryConsolidator",
    "PerceptionRouter",
    "PlannedAction",
    "RoleBasedAgentOrchestrator",
]


@dataclass
class PlannedAction:
    """Action planned by a cognitive engine for execution via a skill or executor."""

    name: str
    skill: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str | None = None


class CognitiveEngineBridge:
    """Bridge an external planner/executor with the skill registry pipeline.

    The bridge is intentionally lightweight: it accepts any callable ``planner``
    that returns an ordered list of actions (strings or mappings) and resolves
    those actions to skill invocations. This lets the system connect LLM-based
    planners, rules engines, or domain-specific heuristics directly to the
    skills/plugin layer without rewriting the orchestration logic.
    """

    def __init__(
        self,
        *,
        planner: Optional[Callable[[str, Dict[str, Any]], Sequence[Any]]] = None,
        executor: Optional[Callable[[PlannedAction, Dict[str, Any]], Any]] = None,
        skill_registry: Optional[Any] = None,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._skill_registry = skill_registry

    # ------------------------------------------------------------------
    def plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> List[PlannedAction]:
        """Generate an ordered list of actions for ``goal`` using the planner."""

        context = context or {}
        raw_plan: Sequence[Any] = []
        if self._planner is not None:
            raw_plan = list(self._planner(goal, context) or [])
        return self._normalise_actions(raw_plan)

    # ------------------------------------------------------------------
    def execute_plan(
        self, goal: str, context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Plan and execute actions, returning structured step results."""

        plan = self.plan(goal, context)
        context = context or {}
        results: List[Dict[str, Any]] = []
        for action in plan:
            result: Any = None
            error: str | None = None
            try:
                result = self._invoke_action(action, context)
                if asyncio.iscoroutine(result):
                    result = asyncio.run(result)
            except Exception as exc:  # pragma: no cover - defensive guardrail
                error = str(exc)
            results.append(
                {
                    "action": action.name,
                    "skill": action.skill,
                    "parameters": action.parameters,
                    "result": result,
                    "error": error,
                }
            )
        return results

    # ------------------------------------------------------------------
    def _normalise_actions(self, items: Iterable[Any]) -> List[PlannedAction]:
        actions: List[PlannedAction] = []
        for idx, item in enumerate(items):
            if isinstance(item, str):
                actions.append(PlannedAction(name=f"step-{idx}", skill=item))
                continue
            if isinstance(item, Mapping):
                skill = str(
                    item.get("skill")
                    or item.get("action")
                    or item.get("name")
                    or f"skill-{idx}"
                )
                actions.append(
                    PlannedAction(
                        name=str(item.get("name", f"step-{idx}")),
                        skill=skill,
                        parameters=dict(item.get("parameters") or item.get("args") or {}),
                        rationale=item.get("rationale") or item.get("why"),
                    )
                )
        return actions

    # ------------------------------------------------------------------
    def _invoke_action(self, action: PlannedAction, context: Dict[str, Any]) -> Any:
        if self._skill_registry is not None and hasattr(self._skill_registry, "invoke"):
            return self._skill_registry.invoke(action.skill, action.parameters, **context)
        if self._executor is None:
            raise RuntimeError(f"No executor or skill registry available for '{action.skill}'")
        return self._executor(action, context)


class PerceptionRouter:
    """Aggregate external/environment signals into prioritised observations."""

    def __init__(
        self,
        connectors: Optional[Sequence[Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]]]] = None,
        *,
        prioritiser: Optional[Callable[[Dict[str, Any]], float]] = None,
    ) -> None:
        self._connectors = list(connectors or [])
        self._prioritiser = prioritiser or (lambda signal: float(signal.get("priority", 0.0)))
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def register_connector(
        self, connector: Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]]
    ) -> None:
        self._connectors.append(connector)

    # ------------------------------------------------------------------
    def poll(self, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Poll all connectors and return prioritised signals."""

        context = context or {}
        signals: List[Dict[str, Any]] = []
        for connector in self._connectors:
            try:
                batch = list(connector(context) or [])
            except Exception:
                continue
            for signal in batch:
                if not isinstance(signal, Mapping):
                    continue
                signals.append(dict(signal))

        deduped: Dict[tuple[str, str | None], Dict[str, Any]] = {}
        for signal in signals:
            key = (str(signal.get("type", "unknown")), signal.get("source"))
            if key not in deduped:
                deduped[key] = signal
                continue
            if self._prioritiser(signal) > self._prioritiser(deduped[key]):
                deduped[key] = signal
        ordered = sorted(deduped.values(), key=self._prioritiser, reverse=True)
        self._history.extend(ordered)
        return ordered

    # ------------------------------------------------------------------
    def history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        records = list(self._history)
        if limit is not None:
            records = records[-limit:]
        return records


@dataclass
class AgentProfile:
    """Role-based agent description for multi-agent orchestration."""

    agent_id: str
    role: str
    specialties: List[str] = field(default_factory=list)
    inbox: List[Dict[str, Any]] = field(default_factory=list)

    def record_task(self, task: Dict[str, Any]) -> None:
        self.inbox.append(dict(task))


class RoleBasedAgentOrchestrator:
    """Coordinate a small team of role-specialised agents over the event bus."""

    def __init__(self, event_bus: Optional[Any] = None) -> None:
        self._agents: Dict[str, AgentProfile] = {}
        self._bus = event_bus

    # ------------------------------------------------------------------
    def register_agent(
        self, agent_id: str, role: str, *, specialties: Optional[Iterable[str]] = None
    ) -> None:
        self._agents[agent_id] = AgentProfile(
            agent_id=agent_id, role=role, specialties=list(specialties or [])
        )

    # ------------------------------------------------------------------
    def assign_task(self, task: Dict[str, Any]) -> Optional[AgentProfile]:
        """Assign ``task`` to the best-matching agent and emit routing metadata."""

        if not self._agents:
            return None
        desired_role = task.get("role") or task.get("stage") or "generalist"
        tags = set(task.get("tags") or [])
        best: AgentProfile | None = None
        best_score = -1
        for agent in self._agents.values():
            role_score = 2 if agent.role == desired_role else 0
            overlap = len(tags.intersection(agent.specialties))
            score = role_score + overlap
            if score > best_score:
                best_score = score
                best = agent
        if best is None:
            return None
        best.record_task(task)
        if self._bus is not None:
            try:
                publish = getattr(self._bus, "publish", None) or getattr(self._bus, "emit", None)
                if publish is not None:
                    publish("agent.task.assigned", {"agent_id": best.agent_id, "task": task})
            except Exception:
                pass
        return best


class GoalRefinementLoop:
    """Continuous goal loop that reflects on outcomes and produces follow-up tasks."""

    def __init__(
        self,
        task_generator: Optional[AutonomousTaskGenerator] = None,
        reflection: Optional[Any] = None,
    ) -> None:
        self._generator = task_generator or AutonomousTaskGenerator()
        self._reflection = reflection or _BasicReflection()
        self._backlog: List[GeneratedTask] = []

    # ------------------------------------------------------------------
    def cycle(
        self,
        completed: Optional[List[Dict[str, Any]]] = None,
        knowledge_gaps: Optional[List[str]] = None,
        environment_signals: Optional[List[str]] = None,
    ) -> List[GeneratedTask]:
        """Run a goal-refinement pass and return newly generated tasks."""

        completed = completed or []
        reflections: List[str] = []
        for item in completed:
            loss_before = float(item.get("loss_before", 1.0))
            loss_after = float(item.get("loss_after", 0.5))
            reflections.append(self._reflection.assess(loss_before, loss_after))

        generated = self._generator.generate_from_context(
            knowledge_gaps=knowledge_gaps or reflections,
            env_changes=environment_signals,
        )
        if not generated:
            generated = self._generator.generate({"idle": True, "novelty_score": 0.5})
        self._backlog.extend(generated)
        return generated

    # ------------------------------------------------------------------
    def backlog(self) -> List[GeneratedTask]:
        return list(self._backlog)
