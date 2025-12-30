
from __future__ import annotations

import logging
import random
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

from backend.knowledge.registry import get_graph_store_instance
from backend.autogpt.autogpt.core.knowledge_graph.ontology import EntityType

if TYPE_CHECKING:  # pragma: no cover - typing only import
    from backend.world_model import WorldModel


class SelfTeacher:
    """Generate self-test questions to reinforce knowledge retention."""

    def __init__(
        self,
        *,
        interval: float = 3600.0,
        questions_per_cycle: int = 1,
        logger: Optional[logging.Logger] = None,
        world_model: "WorldModel | None" = None,
        world_model_hook: Optional[Callable[[Callable[["WorldModel"], None]], None]] = None,
    ) -> None:
        self._interval = max(60.0, interval)
        self._questions_per_cycle = max(1, questions_per_cycle)
        self._logger = logger or logging.getLogger(__name__)
        self._last_run: float = 0.0
        self._world_model: "WorldModel | None" = None
        self._world_model_hook: Optional[
            Callable[[Callable[["WorldModel"], None]], None]
        ] = None
        self.attach_world_model(world_model=world_model, hook=world_model_hook)

    def attach_world_model(
        self,
        *,
        world_model: "WorldModel | None" = None,
        hook: Optional[Callable[[Callable[["WorldModel"], None]], None]] = None,
    ) -> None:
        """Attach a world model or delegation hook for competence updates."""

        if world_model is not None:
            self._world_model = world_model
        if hook is not None:
            self._world_model_hook = hook

    def _with_world_model(self, func: Callable[["WorldModel"], None]) -> None:
        if self._world_model is not None:
            try:
                func(self._world_model)
            except Exception:  # pragma: no cover - defensive logging
                self._logger.debug(
                    "World-model update from self-teacher failed.", exc_info=True
                )
            return
        if self._world_model_hook is not None:
            try:
                self._world_model_hook(func)
            except Exception:  # pragma: no cover - defensive logging
                self._logger.debug(
                    "World-model hook invocation from self-teacher failed.",
                    exc_info=True,
                )

    async def maybe_run(
        self,
        *,
        ability_registry,
        memory,
        knowledge_acquisition,
        cognition_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        if now - self._last_run < self._interval:
            return
        ability_specs = []
        try:
            ability_specs = ability_registry.dump_abilities()
        except Exception:
            self._logger.debug("Unable to dump ability specs for self-teacher.", exc_info=True)
        query_ability = self._find_ability(ability_specs, ("query_language_model", "language_model"))
        if query_ability is None:
            self._logger.debug("Self-teacher requires a language-model query ability; skipping cycle.")
            self._last_run = now
            return
        concepts = self._fetch_concepts()
        if not concepts:
            self._last_run = now
            return
        random.shuffle(concepts)
        for concept in concepts[: self._questions_per_cycle]:
            await self._exercise_concept(
                concept,
                ability_registry=ability_registry,
                query_ability=query_ability,
                ability_specs=ability_specs,
                memory=memory,
                knowledge_acquisition=knowledge_acquisition,
                cognition_metadata=cognition_metadata or {},
            )
        self._last_run = now

    async def _exercise_concept(
        self,
        concept: Dict[str, Any],
        *,
        ability_registry,
        query_ability: str,
        ability_specs: Sequence[Any],
        memory,
        knowledge_acquisition,
        cognition_metadata: Dict[str, Any],
    ) -> None:
        label = concept.get("label") or concept.get("id") or "concept"
        description = concept.get("description", "")
        question_prompt = (
            f'Generate a concise self-test question that checks understanding of the concept "{label}". '
            "Provide the question only."
        )
        question_result = await ability_registry.perform(query_ability, query=question_prompt)
        if not question_result.success or not question_result.message:
            self._logger.debug("Self-teacher failed to generate question for %s", label)
            return
        question = question_result.message.strip()
        answer_prompt = (
            "You are reviewing the concept {label}. Use the knowledge below if available.\n"
            "Concept description: {description}\n"
            "Question: {question}\n"
            "Answer concisely. If the answer is unknown, respond with UNKNOWN."
        ).format(label=label, description=description or "(no description available)", question=question)
        answer_result = await ability_registry.perform(query_ability, query=answer_prompt)
        answer_text = (answer_result.message or "").strip()
        success = answer_result.success and answer_text and "unknown" not in answer_text.lower()
        memory.add(
            f"Self-test::Concept={label}::Question={question}::Answer={answer_text or 'NO ANSWER'}::Success={success}"
        )
        if success:
            return
        failure_metadata = dict(cognition_metadata)
        failure_metadata.update(
            {
                "concept": label,
                "description": description or "",
                "question": question,
                "answer": answer_text or "",
                "reason": "Self-test identified a knowledge gap.",
                "source": "self_teacher",
            }
        )
        self._with_world_model(
            lambda model: self._record_learning_gap(
                model,
                label,
                failure_metadata,
            )
        )
        # Trigger knowledge acquisition to seek additional information
        metadata = dict(failure_metadata)
        metadata.update({"needs_knowledge": True, "knowledge_query": question})
        if knowledge_acquisition is None:
            return
        override = knowledge_acquisition.maybe_acquire(
            metadata=metadata,
            ability_specs=ability_specs,
            task=SimpleNamespace(description=question, objective=label),
            current_selection=None,
        )
        if not override:
            return
        session_id = override.get("knowledge_session_id")
        if session_id:
            knowledge_acquisition.mark_session_started(session_id)
        try:
            acquisition_result = await ability_registry.perform(
                override["next_ability"], **override.get("ability_arguments", {})
            )
            knowledge_summary = acquisition_result.message or ""
            memory.add(
                f"Knowledge-acquisition::Concept={label}::Ability={override['next_ability']}::Message={knowledge_summary}"
            )
            if session_id:
                knowledge_acquisition.complete_session(
                    session_id,
                    acquisition_result,
                    metadata=metadata,
                    memory=memory,
                )
            if acquisition_result.success:
                success_metadata = dict(metadata)
                success_metadata.update(
                    {
                        "knowledge_summary": knowledge_summary,
                        "reason": "Self-teacher knowledge acquisition succeeded.",
                        "source": "self_teacher",
                    }
                )
                self._with_world_model(
                    lambda model: self._record_learning_success(
                        model,
                        label,
                        success_metadata,
                    )
                )
        except Exception:
            self._logger.debug("Knowledge acquisition ability failed for %s", label, exc_info=True)

    def _fetch_concepts(self) -> List[Dict[str, Any]]:
        try:
            store = get_graph_store_instance()
            result = store.query(entity_type=EntityType.CONCEPT)
        except Exception:
            self._logger.debug("Unable to query concepts for self-teacher.", exc_info=True)
            return []
        concepts: List[Dict[str, Any]] = []
        for node in result.get("nodes", []):
            node_id = getattr(node, "id", None) or node.get("id") if isinstance(node, dict) else None
            node_type = getattr(node, "type", None) or (node.get("type") if isinstance(node, dict) else None)
            if node_type and isinstance(node_type, EntityType):
                if node_type != EntityType.CONCEPT:
                    continue
            elif node_type and str(node_type).lower() != "concept":
                continue
            props = {}
            if hasattr(node, "properties"):
                props = getattr(node, "properties") or {}
            elif isinstance(node, dict):
                props = node.get("properties", {})
            label = props.get("label") or props.get("name") or node_id
            description = props.get("description", "")
            concepts.append(
                {
                    "id": node_id or label or "concept",
                    "label": label or node_id or "concept",
                    "description": description or "",
                }
            )
        return concepts

    def _find_ability(self, ability_specs: Sequence[Any], keywords: Sequence[str]) -> Optional[str]:
        for spec in ability_specs:
            name = getattr(spec, "name", None)
            if isinstance(name, str) and any(keyword in name.lower() for keyword in keywords):
                return name
            tags = getattr(spec, "tags", None)
            if tags and any(keyword in str(tag).lower() for tag in tags for keyword in keywords):
                return name if isinstance(name, str) else None
        return None

    @staticmethod
    def _record_learning_gap(
        model: "WorldModel",
        concept: str,
        metadata: Dict[str, Any],
    ) -> None:
        model.update_competence(
            concept,
            0.0,
            source="self_teacher",
            metadata=dict(metadata),
        )
        model.record_opportunity(
            f"Deepen understanding of {concept}",
            weight=0.4,
            metadata=dict(metadata),
        )

    @staticmethod
    def _record_learning_success(
        model: "WorldModel",
        concept: str,
        metadata: Dict[str, Any],
    ) -> None:
        model.update_competence(
            concept,
            0.8,
            source="self_teacher",
            metadata=dict(metadata),
        )
