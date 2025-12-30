"""
Internal action planner used to suggest follow-up actions after comprehension.

The planner applies lightweight rules based on the recognised intent, semantic
relations and affect cues. This replaces prior LLM-derived action inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

try:  # pragma: no cover - optional commonsense integration
    from modules.brain.reasoning.commonsense import CommonSenseReasoner
except Exception:  # pragma: no cover - optional dependency missing
    CommonSenseReasoner = None  # type: ignore[assignment]

try:  # pragma: no cover - optional multi-strategy reasoner
    from modules.brain.reasoning.general_reasoner import GeneralReasoner
except Exception:  # pragma: no cover - optional dependency missing
    GeneralReasoner = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from BrainSimulationSystem.models.knowledge_graph import KnowledgeGraph


@dataclass
class ActionPlan:
    actions: List[str]
    confidence: float
    rationale: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    commonsense: List[Dict[str, Any]] = field(default_factory=list)


class ActionPlanner:
    """
    Rule-driven action planner for post-comprehension decisions.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self._logger = logging.getLogger(self.__class__.__name__)
        self.intent_map: Dict[str, Sequence[str]] = {
            "question": self.params.get("action_question", ["answer_question"]),
            "command": self.params.get("action_command", ["execute_request"]),
            "statement": self.params.get("action_statement", ["acknowledge"]),
            "greeting": self.params.get("action_greeting", ["greet_back"]),
            "feedback": self.params.get("action_feedback", ["acknowledge"]),
        }
        self.base_confidence = float(self.params.get("base_confidence", 0.55))
        self.max_actions = int(self.params.get("max_actions", 4))
        self.goal_depth = int(self.params.get("goal_depth", 3))
        self.max_subtasks = int(self.params.get("max_subtasks", 5))
        self._max_plan_branches = int(self.params.get("max_plan_branches", 4))
        predicates = self.params.get(
            "goal_predicates",
            ["requires", "needs", "depends_on", "enables", "subtask_of", "part_of", "step", "precedes"],
        )
        self._hierarchical_predicates: Sequence[str] = tuple(predicates)
        commonsense_cfg = self.params.get("commonsense", {})
        self._commonsense_limit = int(commonsense_cfg.get("limit", 3))
        self._commonsense_reasoner: Optional["CommonSenseReasoner"]
        if commonsense_cfg.get("enabled") and CommonSenseReasoner is not None:
            try:
                self._commonsense_reasoner = CommonSenseReasoner(
                    enabled=True,
                    endpoint=commonsense_cfg.get("endpoint", "https://api.conceptnet.io"),
                )
            except Exception as exc:  # pragma: no cover - optional dependency runtime failure
                self._logger.debug("Failed to initialize CommonSenseReasoner: %s", exc)
                self._commonsense_reasoner = None
        else:
            self._commonsense_reasoner = None

        self._general_reasoner_steps = int(self.params.get("general_reasoner_steps", 3))
        self._general_reasoner: Optional["GeneralReasoner"]
        use_reasoner = bool(self.params.get("enable_general_reasoner"))
        if use_reasoner and GeneralReasoner is not None:
            try:
                self._general_reasoner = GeneralReasoner()
            except Exception as exc:  # pragma: no cover - optional dependency runtime failure
                self._logger.debug("Failed to initialize GeneralReasoner: %s", exc)
                self._general_reasoner = None
        else:
            self._general_reasoner = None

    def plan(
        self,
        intent: str,
        suggested_actions: Sequence[str],
        semantic_relations: Sequence[Dict[str, str]],
        affect_tone: Sequence[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> ActionPlan:
        """
        Produce an ordered list of follow-up actions.

        Parameters
        ----------
        intent:
            Label returned by the intent recogniser.
        suggested_actions:
            Suggestions derived from semantic analysis (e.g. verb-object pairs).
        semantic_relations:
            Relations extracted from the semantic analyser to inform prioritisation.
        affect_tone:
            Tone tags detected by the affect analyser (to detect urgency/politeness).
        context:
            Optional language context dictionary (memory, goals, etc.).
        """

        structured_context = context if isinstance(context, dict) else {}
        intent_actions = list(self.intent_map.get(intent, []))
        combined_actions: List[str] = []

        for action in intent_actions:
            if action not in combined_actions:
                combined_actions.append(action)

        for suggestion in suggested_actions:
            normalized = suggestion.replace(" ", "_")
            if normalized not in combined_actions:
                combined_actions.append(normalized)

        for relation in semantic_relations:
            head = relation.get("head")
            dep = relation.get("dependent")
            rel_type = relation.get("relation")
            if rel_type == "object" and head:
                candidate = f"utilise_{head}"
                if candidate not in combined_actions:
                    combined_actions.append(candidate)
            if rel_type == "subject" and dep:
                candidate = f"confirm_{dep}"
                if candidate not in combined_actions:
                    combined_actions.append(candidate)

        goal_tree = self._build_goal_tree(structured_context)
        for entry in goal_tree:
            action = entry.get("action")
            if action and action not in combined_actions:
                combined_actions.append(action)

        commonsense_hints = self._commonsense_support(structured_context)

        if "urgent" in affect_tone:
            combined_actions.insert(0, "prioritise_request")
        elif "polite" in affect_tone and "acknowledge" not in combined_actions:
            combined_actions.insert(0, "acknowledge")

        combined_actions = combined_actions[: self.max_actions]

        confidence = self.base_confidence
        if intent == "command":
            confidence += 0.15
        if "urgent" in affect_tone:
            confidence += 0.1
        confidence = float(min(0.95, confidence))

        rationale = {
            "intent": intent,
            "suggested_actions": list(suggested_actions),
            "affect_tone": list(affect_tone),
        }
        if structured_context:
            rationale["context_keys"] = list(structured_context.keys())
        if goal_tree:
            rationale["goal_tree_steps"] = len(goal_tree)
        if commonsense_hints:
            rationale["commonsense"] = commonsense_hints

        return ActionPlan(
            actions=combined_actions,
            confidence=confidence,
            rationale=rationale,
            steps=goal_tree,
            commonsense=commonsense_hints,
        )

    # ------------------------------------------------------------------ #
    # Goal decomposition helpers
    # ------------------------------------------------------------------ #
    def _build_goal_tree(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        goals = self._normalize_goals(context)
        if not goals:
            return []

        plan_tree: List[Dict[str, Any]] = []
        graph: Optional["KnowledgeGraph"] = context.get("knowledge_graph") if isinstance(context, dict) else None
        if graph is not None:
            plan_tree.extend(self._graph_subtasks(graph, goals))

        reasoner = context.get("general_reasoner") if isinstance(context, dict) else None
        if reasoner is None:
            reasoner = self._general_reasoner
        if reasoner is not None and len(plan_tree) < self._max_plan_branches:
            plan_tree.extend(
                self._reasoner_subtasks(reasoner, goals, max_branches=self._max_plan_branches - len(plan_tree))
            )

        return plan_tree[: self._max_plan_branches]

    def _graph_subtasks(self, graph: "KnowledgeGraph", goals: Sequence[str]) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        for goal in goals:
            path_entries: List[Dict[str, Any]] = []
            if hasattr(graph, "find_paths"):
                try:
                    path_entries = graph.find_paths(
                        goal,
                        predicates=self._hierarchical_predicates,
                        max_depth=self.goal_depth,
                        max_paths=self.max_subtasks,
                    )
                except Exception as exc:  # pragma: no cover - defensive fallback
                    self._logger.debug("KnowledgeGraph.find_paths failed for %s: %s", goal, exc)
                    path_entries = []
            if not path_entries:
                path_entries = self._legacy_graph_paths(graph, goal)

            for bundle in path_entries:
                step = self._bundle_to_step(goal, bundle)
                if step:
                    steps.append(step)
                    if len(steps) >= self._max_plan_branches:
                        return steps
        return steps

    def _legacy_graph_paths(self, graph: "KnowledgeGraph", goal: str) -> List[Dict[str, Any]]:
        path_entries: List[Dict[str, Any]] = []
        predicates = tuple(self._hierarchical_predicates)
        for predicate in predicates:
            # prerequisites where goal is object
            for subject, _, _ in graph.query(predicate=predicate, obj=goal):
                path_entries.append(
                    {
                        "nodes": [goal, subject],
                        "triples": [{"triple": (subject, predicate, goal), "direction": "backward"}],
                        "target": subject,
                        "source": "knowledge_graph",
                    }
                )
            # sequencing where goal leads to another action
            for _, _, obj in graph.query(subject=goal, predicate=predicate):
                path_entries.append(
                    {
                        "nodes": [goal, obj],
                        "triples": [{"triple": (goal, predicate, obj), "direction": "forward"}],
                        "target": obj,
                        "source": "knowledge_graph",
                    }
                )
        return path_entries[: self.max_subtasks]

    def _bundle_to_step(self, goal: str, bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        triples_info: List[Any] = bundle.get("triples") or []
        if not triples_info:
            return None
        target = bundle.get("target")
        nodes = bundle.get("nodes")
        if not target and nodes:
            target = nodes[-1]
        if not target or target == goal:
            return None

        subtasks: List[Dict[str, Any]] = []
        for idx, info in enumerate(triples_info, start=1):
            if isinstance(info, dict):
                triple = info.get("triple")
                direction = info.get("direction", "forward")
            else:
                triple = info
                direction = "forward"
            if not triple:
                continue
            subtasks.append(
                {
                    "step": idx,
                    "from": triple[0],
                    "relation": triple[1],
                    "to": triple[2],
                    "direction": direction,
                }
            )

        if not subtasks:
            return None

        return {
            "goal": goal,
            "action": target,
            "subtasks": subtasks,
            "source": bundle.get("source", "knowledge_graph"),
            "path_nodes": bundle.get("nodes", []),
        }

    def _reasoner_subtasks(
        self,
        reasoner: Any,
        goals: Sequence[str],
        *,
        max_branches: int,
    ) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        for goal in goals:
            if len(steps) >= max_branches:
                break
            try:
                trace = reasoner.reason_about_unknown(
                    goal,
                    max_steps=self._general_reasoner_steps,
                    confidence_threshold=0.5,
                )
            except Exception as exc:  # pragma: no cover - optional dependency failure
                self._logger.debug("GeneralReasoner failed for %s: %s", goal, exc)
                continue
            subtasks = []
            for idx, entry in enumerate(trace, start=1):
                hypothesis = entry.get("hypothesis") if isinstance(entry, dict) else entry
                verification = entry.get("verification") if isinstance(entry, dict) else ""
                if not hypothesis:
                    continue
                subtasks.append(
                    {
                        "step": idx,
                        "from": goal if idx == 1 else subtasks[-1]["to"],
                        "relation": entry.get("method", "reasoning") if isinstance(entry, dict) else "reasoning",
                        "to": str(hypothesis),
                        "verification": verification,
                    }
                )
            if not subtasks:
                continue
            steps.append(
                {
                    "goal": goal,
                    "action": subtasks[-1]["to"],
                    "subtasks": subtasks,
                    "source": "general_reasoner",
                }
            )
        return steps

    def _normalize_goals(self, context: Dict[str, Any]) -> List[str]:
        raw_goals = context.get("goals") if isinstance(context, dict) else None
        if raw_goals is None:
            return []
        if isinstance(raw_goals, (str, bytes)):
            raw_items = [raw_goals]
        elif isinstance(raw_goals, (list, tuple, set)):
            raw_items = list(raw_goals)
        else:
            raw_items = [raw_goals]

        normalized: List[str] = []
        for goal in raw_items:
            if isinstance(goal, str):
                candidate = goal.strip()
            elif isinstance(goal, dict):
                candidate = (
                    goal.get("name")
                    or goal.get("goal")
                    or goal.get("target")
                    or goal.get("description")
                    or ""
                )
                candidate = str(candidate).strip()
            else:
                candidate = str(goal).strip()
            if candidate:
                normalized.append(candidate)
        return normalized

    def _commonsense_support(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        reasoner = None
        if isinstance(context, dict):
            reasoner = context.get("commonsense_reasoner")
        if reasoner is None:
            reasoner = self._commonsense_reasoner
        if reasoner is None:
            return []

        hints: List[Dict[str, Any]] = []
        goals = self._normalize_goals(context if isinstance(context, dict) else {})
        for goal in goals:
            try:
                conclusions = reasoner.infer(goal, limit=self._commonsense_limit)
            except Exception as exc:  # pragma: no cover - optional dependency failure
                self._logger.debug("Commonsense inference failed for %s: %s", goal, exc)
                continue
            if conclusions:
                hints.append({"goal": goal, "conclusions": conclusions})
        return hints
