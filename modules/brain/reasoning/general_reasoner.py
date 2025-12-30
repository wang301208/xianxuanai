"""General reasoning engine combining several simplistic techniques.

This module defines :class:`GeneralReasoner`, a light‑weight component that
combines three complementary approaches:

* a tiny **concept graph** storing explicit relations between concepts,
* the existing :class:`AnalogicalReasoner` for structural mapping, and
* a very small **few‑shot memory** used as nearest‑neighbour retrieval.

In addition to these heuristics the reasoner can optionally coordinate with the
broader planning stack and with the task memory system.  The
:func:`reason_about_unknown` method now iterates until a stopping criterion is
met, persisting every hypothesis into :class:`modules.memory.task_memory.TaskMemoryManager`
or a provided long-term vector store.  Stored traces can be recalled through
:func:`resume_reasoning`, allowing :class:`modules.brain.whole_brain.WholeBrainSimulation`
or downstream planners to continue verification across cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .analogical import AnalogicalReasoner

try:  # optional dependencies used for persistence
    from modules.memory import ExperiencePayload, TaskMemoryManager
    from modules.memory.vector_store import VectorMemoryStore, VectorRecord
except Exception:  # pragma: no cover - optional dependency unavailable during docs build
    ExperiencePayload = None  # type: ignore
    TaskMemoryManager = None  # type: ignore
    VectorMemoryStore = None  # type: ignore
    VectorRecord = None  # type: ignore

try:  # lightweight chain-of-thought planner used for subgoal decomposition
    from backend.autogpt.autogpt.core.planning.reasoner import (  # type: ignore
        ReasoningPlanner as ChainReasoningPlanner,
    )
    from backend.autogpt.autogpt.core.planning.schema import (  # type: ignore
        Task,
        TaskContext,
        TaskStatus,
        TaskType,
    )
except Exception:  # pragma: no cover - planner not available in minimal installs
    ChainReasoningPlanner = None  # type: ignore
    Task = None  # type: ignore
    TaskContext = None  # type: ignore
    TaskStatus = None  # type: ignore
    TaskType = None  # type: ignore


@dataclass
class _ReasoningStep:
    """Structured representation of a single reasoning iteration."""

    iteration: int
    hypothesis: str
    verification: str
    confidence: float
    method: str
    memory_id: Optional[str] = None

    def as_dict(self) -> Dict[str, str | float | int | None]:
        return {
            "iteration": self.iteration,
            "hypothesis": self.hypothesis,
            "verification": self.verification,
            "confidence": round(float(self.confidence), 3),
            "method": self.method,
            "memory_id": self.memory_id,
        }


class GeneralReasoner:
    """Toy general reasoner combining graph lookup, analogy and few‑shot hints."""

    def __init__(
        self,
        *,
        task_memory: "TaskMemoryManager | None" = None,
        vector_store: "VectorMemoryStore | None" = None,
        planner: object | None = None,
    ) -> None:
        # concept_graph maps a concept to a set of related concepts
        self.concept_graph: Dict[str, set[str]] = {}
        self.analogical = AnalogicalReasoner()
        # store few‑shot examples as ``(description, solution)`` tuples
        self.examples: List[Tuple[str, str]] = []
        self._task_memory = task_memory
        self._vector_store = vector_store
        if planner is not None:
            self._planner = planner
        elif ChainReasoningPlanner is not None:  # pragma: no branch - deterministic selection
            self._planner = ChainReasoningPlanner()
        else:
            self._planner = None
        self._trace_cache: Dict[str, List[_ReasoningStep]] = {}

    # ------------------------------------------------------------------
    # knowledge insertion helpers
    def add_concept_relation(self, concept: str, related: str) -> None:
        """Insert a directed relation ``concept -> related`` into the graph."""

        self.concept_graph.setdefault(concept.lower(), set()).add(related.lower())

    def add_example(self, description: str, solution: str) -> None:
        """Store a textual example and its solution for few‑shot retrieval."""

        self.examples.append((description, solution))

    # ------------------------------------------------------------------
    def _nearest_example(self, task_description: str) -> Tuple[str, str] | None:
        """Return the example with maximal word overlap to ``task_description``."""

        if not self.examples:
            return None
        td_words = set(task_description.lower().split())
        best: Tuple[str, str] | None = None
        best_score = -1
        for desc, sol in self.examples:
            score = len(td_words & set(desc.lower().split()))
            if score > best_score:
                best = (desc, sol)
                best_score = score
        return best

    # ------------------------------------------------------------------
    def reason_about_unknown(
        self,
        task_description: str,
        max_steps: int | None = None,
        *,
        task_id: str | None = None,
        confidence_threshold: float = 0.75,
        resume: bool = True,
    ) -> List[Dict[str, str | float | int | None]]:
        """Generate hypotheses for an unfamiliar ``task_description``.

        The method tries three techniques in order: concept graph lookup,
        analogical transfer and few‑shot retrieval.  When these heuristics are
        insufficient the reasoner keeps iterating, alternating planner support
        (when available) and a light-weight reflection step until either the
        requested ``max_steps`` is reached or the accumulated confidence meets
        ``confidence_threshold``.  Each produced step is a dictionary with
        ``hypothesis`` and ``verification`` fields describing the reasoning
        step and how it was (naively) validated.  At least one step is always
        returned so that callers can display some form of progress even without
        prior knowledge.
        """

        description = task_description.strip()
        if not description:
            return [
                {
                    "hypothesis": "no description provided",
                    "verification": "unable to reason without task description",
                    "confidence": 0.0,
                    "method": "validation",
                    "iteration": 1,
                    "memory_id": None,
                }
            ]

        resolved_task_id = task_id or self._default_task_id(description)
        cached_trace = []
        if resume:
            cached_trace = self._load_trace(resolved_task_id, description)

        steps: List[_ReasoningStep] = list(cached_trace)
        used_methods = {step.method for step in steps}
        used_relations = {
            tuple(h.split(" relates to "))
            for h in (step.hypothesis for step in steps if " relates to " in step.hypothesis)
        }
        used_memory_ids = {step.memory_id for step in steps if step.memory_id}

        tokens = description.lower().split()

        requested_steps = max_steps if isinstance(max_steps, int) and max_steps > 0 else None
        memory_lookup = max(3, requested_steps or 6)
        memory_records = self._retrieve_memories(description, top_k=memory_lookup * 2)

        last_confidence = steps[-1].confidence if steps else 0.0
        iteration = len(steps)
        reusable_methods = {"planner", "reflection"}

        while (requested_steps is None or iteration < requested_steps) and (
            last_confidence < confidence_threshold
        ):
            iteration += 1
            step = self._next_step(
                description,
                tokens,
                iteration,
                used_methods,
                used_relations,
                memory_records,
                used_memory_ids,
                reusable_methods=reusable_methods,
                prior_steps=steps,
            )
            if step is None:
                break
            steps.append(step)
            used_methods.add(step.method)
            if " relates to " in step.hypothesis:
                parts = step.hypothesis.split(" relates to ")
                if len(parts) == 2:
                    used_relations.add((parts[0], parts[1]))
            if step.memory_id:
                used_memory_ids.add(step.memory_id)
            last_confidence = step.confidence
            self._persist_step(resolved_task_id, step)

        if not steps:
            fallback = self._fallback_step(description, iteration=1)
            steps.append(fallback)
            self._persist_step(resolved_task_id, fallback)

        limit = requested_steps or len(steps)
        return [step.as_dict() for step in steps][:limit]

    # ------------------------------------------------------------------ public helpers
    def resume_reasoning(
        self, task_id: str, task_description: str | None = None, *, limit: int | None = None
    ) -> List[Dict[str, str | float | int | None]]:
        """Return stored reasoning trace for ``task_id``.

        When ``task_description`` is provided and no in-memory trace exists the
        method queries the configured memory backend to reconstruct the trace.
        """

        trace = list(self._trace_cache.get(task_id, []))
        if trace:
            return [step.as_dict() for step in trace[: limit or len(trace)]]

        if task_description:
            records = self._retrieve_memories(task_description, top_k=limit or 8)
            recovered: Dict[int, _ReasoningStep] = {}
            for record in records:
                metadata = getattr(record, "metadata", {}) or {}
                if metadata.get("kind") != "reasoning_step":
                    continue
                if metadata.get("task_id") != task_id:
                    continue
                iteration = int(metadata.get("iteration", 0) or 0)
                if iteration <= 0:
                    continue
                step = _ReasoningStep(
                    iteration=iteration,
                    hypothesis=str(metadata.get("hypothesis", record.text)),
                    verification=str(metadata.get("verification", "retrieved from memory")),
                    confidence=float(metadata.get("confidence", 0.0) or 0.0),
                    method=str(metadata.get("method", "memory")),
                    memory_id=getattr(record, "id", None),
                )
                recovered.setdefault(iteration, step)
            if recovered:
                ordered = [recovered[idx] for idx in sorted(recovered)]
                self._trace_cache[task_id] = ordered
                return [step.as_dict() for step in ordered[: limit or len(ordered)]]
        return []

    # ------------------------------------------------------------------ internal helpers
    def _next_step(
        self,
        task_description: str,
        tokens: Sequence[str],
        iteration: int,
        used_methods: Iterable[str],
        used_relations: Iterable[Tuple[str, str]],
        memory_records: Sequence["VectorRecord"],
        used_memory_ids: Iterable[str | None],
        *,
        reusable_methods: Iterable[str] | None = None,
        prior_steps: Sequence[_ReasoningStep] | None = None,
    ) -> _ReasoningStep | None:
        used_method_set = set(used_methods)
        used_relation_set = {tuple(r) for r in used_relations}
        used_memory_set = {m for m in used_memory_ids if m}
        reusable_set = set(reusable_methods or [])

        def method_available(name: str) -> bool:
            return name not in used_method_set or name in reusable_set

        # 1) concept graph lookup for unseen relations
        if method_available("concept_graph"):
            for tok in tokens:
                neighbours = self.concept_graph.get(tok)
                if not neighbours:
                    continue
                for target in neighbours:
                    relation = (tok, target)
                    if relation in used_relation_set:
                        continue
                    memory_id, snippet = self._next_memory(memory_records, used_memory_set)
                    verification = "relation derived from concept graph"
                    confidence = 0.45
                    if snippet:
                        verification += f" | memory: {snippet}"
                        confidence += 0.05
                    return _ReasoningStep(
                        iteration=iteration,
                        hypothesis=f"{tok} relates to {target}",
                        verification=verification,
                        confidence=min(0.95, confidence),
                        method="concept_graph",
                        memory_id=memory_id,
                    )

        # 2) analogical reasoning if knowledge exists
        if method_available("analogical"):
            target_structure = {
                "subject": tokens[0] if tokens else "",
                "object": tokens[-1] if tokens else "",
            }
            mapping = self.analogical.transfer_knowledge("default", task_description, target_structure)
            if mapping:
                memory_id, snippet = self._next_memory(memory_records, used_memory_set)
                verification = "mapped roles via analogical reasoning"
                confidence = 0.55
                if snippet:
                    verification += f" | memory: {snippet}"
                    confidence += 0.05
                return _ReasoningStep(
                    iteration=iteration,
                    hypothesis=f"analogy suggests mapping {mapping}",
                    verification=verification,
                    confidence=min(0.95, confidence),
                    method="analogical",
                    memory_id=memory_id,
                )

        # 3) few-shot retrieval for solution hints
        if method_available("few_shot"):
            ex = self._nearest_example(task_description)
            if ex:
                desc, sol = ex
                memory_id, snippet = self._next_memory(memory_records, used_memory_set)
                verification = f"candidate solution: {sol}"
                confidence = 0.5
                if snippet:
                    verification += f" | memory: {snippet}"
                    confidence += 0.05
                return _ReasoningStep(
                    iteration=iteration,
                    hypothesis=f"similar to example '{desc}'",
                    verification=verification,
                    confidence=min(0.9, confidence),
                    method="few_shot",
                    memory_id=memory_id,
                )

        # 4) planner-assisted subgoal decomposition when heuristics fail
        if method_available("planner"):
            subgoal = self._planner_subgoal(task_description, memory_records, used_memory_set)
            if subgoal:
                memory_id, snippet = self._next_memory(memory_records, used_memory_set)
                verification = "planner suggested actionable subgoal"
                confidence = 0.65
                if snippet:
                    verification += f" | memory: {snippet}"
                    confidence += 0.05
                return _ReasoningStep(
                    iteration=iteration,
                    hypothesis=subgoal,
                    verification=verification,
                    confidence=min(0.98, confidence),
                    method="planner",
                    memory_id=memory_id,
                )

        # 5) reflective iteration to refine hypotheses
        if method_available("reflection") and prior_steps:
            reflection = self._reflection_step(prior_steps, iteration, memory_records, used_memory_set)
            if reflection:
                return reflection

        # 6) fallback when nothing else available
        return self._fallback_step(task_description, iteration)

    def _reflection_step(
        self,
        prior_steps: Sequence[_ReasoningStep],
        iteration: int,
        memory_records: Sequence["VectorRecord"],
        used_memory_ids: set[str],
    ) -> _ReasoningStep | None:
        if not prior_steps:
            return None

        last_step = prior_steps[-1]
        base_confidence = float(last_step.confidence)
        increment = 0.1 if base_confidence < 0.7 else 0.05
        new_confidence = min(0.99, max(0.3, base_confidence + increment))
        hypothesis = f"refine focus based on step {last_step.iteration}: {last_step.hypothesis}"
        if last_step.method == "fallback":
            hypothesis = f"formulate experiment to address unknown from step {last_step.iteration}"

        verification = f"reassess step {last_step.iteration} for new insights"
        memory_id, snippet = self._next_memory(memory_records, used_memory_ids)
        if snippet:
            verification += f" | memory: {snippet}"
            new_confidence = min(0.99, new_confidence + 0.02)

        return _ReasoningStep(
            iteration=iteration,
            hypothesis=hypothesis,
            verification=verification,
            confidence=new_confidence,
            method="reflection",
            memory_id=memory_id,
        )

    def _planner_subgoal(
        self,
        task_description: str,
        memory_records: Sequence["VectorRecord"],
        used_memory_ids: set[str],
    ) -> str | None:
        planner = getattr(self, "_planner", None)
        if planner is None:
            return None
        memory_snippets = [
            self._format_memory(record)
            for record in memory_records
            if record.id not in used_memory_ids
        ][:3]
        if hasattr(planner, "plan_subgoals"):
            try:
                result = planner.plan_subgoals(task_description, context=memory_snippets)
            except Exception:
                return None
            if result:
                return str(result[0])
            return None
        if ChainReasoningPlanner is not None and isinstance(planner, ChainReasoningPlanner):
            if Task is None or TaskType is None or TaskContext is None or TaskStatus is None:
                return None
            ready = [snippet for snippet in memory_snippets if snippet]
            try:
                task = Task(
                    objective=task_description,
                    type=TaskType.PLAN,
                    priority=1,
                    ready_criteria=ready or ["identify key unknowns"],
                    acceptance_criteria=[],
                    context=TaskContext(status=TaskStatus.READY, memories=list(memory_snippets)),
                )
                result = planner.plan(task)
            except Exception:
                return None
            reasoning = list(getattr(result, "reasoning", []))
            for thought in reasoning:
                if thought and not str(thought).lower().startswith("objective"):
                    return str(thought)
            next_step = getattr(result, "next_step", "")
            return str(next_step) if next_step else None
        if hasattr(planner, "plan"):
            try:
                output = planner.plan(task_description)
            except Exception:
                return None
            if isinstance(output, str):
                return output
            if isinstance(output, Sequence) and output:
                return str(output[0])
            if hasattr(output, "next_step"):
                return str(output.next_step)
        return None

    def _fallback_step(self, task_description: str, iteration: int) -> _ReasoningStep:
        return _ReasoningStep(
            iteration=iteration,
            hypothesis="no prior knowledge available",
            verification="proceed with exploratory experiments",
            confidence=0.25,
            method="fallback",
            memory_id=None,
        )

    def _retrieve_memories(self, task_description: str, top_k: int = 5) -> List["VectorRecord"]:
        records: List["VectorRecord"] = []
        if self._task_memory is not None:
            try:
                records = list(self._task_memory.recall(task_description, top_k=top_k))
            except Exception:
                records = []
        elif self._vector_store is not None:
            try:
                records = list(self._vector_store.query(task_description, top_k=top_k))
            except Exception:
                records = []
        return records

    def _next_memory(
        self,
        records: Sequence["VectorRecord"],
        used_ids: set[str],
    ) -> Tuple[Optional[str], str]:
        for record in records:
            if record.id in used_ids:
                continue
            used_ids.add(record.id)
            snippet = self._format_memory(record)
            return record.id, snippet
        return None, ""

    def _format_memory(self, record: "VectorRecord") -> str:
        metadata = getattr(record, "metadata", {}) or {}
        for key in ("summary", "hypothesis", "verification"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        text = getattr(record, "text", "")
        return text.strip()

    def _persist_step(self, task_id: str, step: _ReasoningStep) -> None:
        self._trace_cache.setdefault(task_id, [])
        if step not in self._trace_cache[task_id]:
            self._trace_cache[task_id].append(step)

        metadata = {
            "kind": "reasoning_step",
            "task_id": task_id,
            "iteration": step.iteration,
            "confidence": float(step.confidence),
            "method": step.method,
            "hypothesis": step.hypothesis,
            "verification": step.verification,
        }
        if step.memory_id:
            metadata["memory_id"] = step.memory_id

        if self._task_memory is not None and ExperiencePayload is not None:
            payload = ExperiencePayload(
                task_id=task_id,
                summary=f"reasoning step {step.iteration}: {step.hypothesis}",
                messages=
                [
                    {"role": "thought", "content": step.hypothesis},
                    {"role": "validation", "content": step.verification},
                ],
                metadata=metadata,
            )
            try:
                self._task_memory.store_experience(payload)
            except Exception:
                pass
        elif self._vector_store is not None:
            text = f"[thought] {step.hypothesis}\n[validation] {step.verification}"
            try:
                self._vector_store.add_text(text, metadata=metadata)
            except Exception:
                pass

    def _load_trace(self, task_id: str, description: str) -> List[_ReasoningStep]:
        if task_id in self._trace_cache:
            return list(self._trace_cache[task_id])
        trace = self.resume_reasoning(task_id, description)
        if trace:
            steps = [
                _ReasoningStep(
                    iteration=int(item.get("iteration", idx + 1)),
                    hypothesis=str(item.get("hypothesis", "")),
                    verification=str(item.get("verification", "")),
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    method=str(item.get("method", "memory")),
                    memory_id=item.get("memory_id") if isinstance(item.get("memory_id"), str) else None,
                )
                for idx, item in enumerate(trace)
            ]
            self._trace_cache[task_id] = steps
            return steps
        return []

    def _default_task_id(self, description: str) -> str:
        digest = hashlib.sha1(description.encode("utf-8")).hexdigest()[:12]
        return f"reasoner:{digest}"


__all__ = ["GeneralReasoner"]
