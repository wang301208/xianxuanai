"""
Language production module (Broca analogue).

Generates natural language replies using internal planning, templates and syntax
validation instead of external LLM calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from BrainSimulationSystem.models.action_planner import ActionPlan


class LanguageProduction:
    """Pipeline wrapper for language generation."""

    def __init__(
        self,
        language_generator,
        syntax_processor,
        action_planner,
        cognitive_controller,
        working_memory,
        semantic_network,
        usage_stats: Dict[str, int],
        grammar_inducer=None,
        sequence_model=None,
    ) -> None:
        self.language_generator = language_generator
        self.syntax_processor = syntax_processor
        self.action_planner = action_planner
        self.cognitive_controller = cognitive_controller
        self.working_memory = working_memory
        self.semantic_network = semantic_network
        self.usage_stats = usage_stats
        self.grammar_inducer = grammar_inducer
        self.sequence_model = sequence_model

    def generate(
        self,
        goal: Any,
        inputs: Dict[str, Any],
        comprehension_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = inputs.get("language_context") or {}
        generation_goal = goal or self._derive_goal_from_comprehension(comprehension_snapshot)

        semantic_obj = comprehension_snapshot.get("_semantic_obj")
        affect_obj = comprehension_snapshot.get("_affect_obj") or comprehension_snapshot.get("affect", {})
        action_plan: ActionPlan = comprehension_snapshot.get("_action_plan_obj") or self.action_planner.plan(
            generation_goal.get("intent", comprehension_snapshot.get("intent", "inform")),
            getattr(semantic_obj, "suggested_actions", []) if semantic_obj else [],
            getattr(semantic_obj, "relations", []) if semantic_obj else [],
            self._extract_affect_tone(affect_obj),
            context,
        )

        controller_flags = comprehension_snapshot.get("controller") or {}
        template_hint = controller_flags.get("template_hint")

        response_payload = self.language_generator.generate_response(
            generation_goal,
            comprehension_snapshot,
            action_plan,
            affect_obj,
            self.semantic_network,
            template_hint=template_hint,
        )

        reply_text = response_payload["text"]
        try:
            syntax_tree = self.syntax_processor.parse_sentence(reply_text.split())
        except RecursionError:
            syntax_tree = {"type": "S", "children": []}

        if self.grammar_inducer:
            self.grammar_inducer.observe_sentence(reply_text.split())
        elif self.sequence_model:
            self.sequence_model.observe_sequence(reply_text.split(), syntax_tree.get("pos", []))

        record_terms = self._split_terms(response_payload["values"].get("key_terms_phrase"))
        self.working_memory.add_record(
            {
                "type": "output",
                "text": reply_text,
                "summary": response_payload["values"].get("summary_sentence") or reply_text,
                "intent": generation_goal.get("intent", comprehension_snapshot.get("intent")),
                "key_terms": record_terms,
                "tone": response_payload.get("tone", []),
                "polarity": response_payload.get("polarity"),
                "actions": list(action_plan.actions),
                "status": "resolved" if action_plan.actions else "informative",
                "semantic_layers": comprehension_snapshot.get("semantic_layers"),
            }
        )
        self.usage_stats["requests"] += 1

        return {
            "goal": generation_goal,
            "reply": reply_text,
            "syntax": syntax_tree,
            "generator": response_payload,
            "action_plan": {
                "actions": action_plan.actions,
                "confidence": action_plan.confidence,
                "rationale": action_plan.rationale,
            },
            "controller": controller_flags,
        }

    @staticmethod
    def _split_terms(phrase: Optional[str]) -> List[str]:
        if not phrase:
            return []
        return [term.strip() for term in phrase.split(",") if term.strip()]

    @staticmethod
    def _extract_affect_tone(affect: Any) -> Sequence[str]:
        if affect is None:
            return []
        if isinstance(affect, dict):
            return affect.get("tone") or []
        return getattr(affect, "tone", []) or []

    @staticmethod
    def _derive_goal_from_comprehension(comprehension: Dict[str, Any]) -> Dict[str, Any]:
        if not comprehension:
            return {"intent": "inform", "summary": "No prior language input"}
        intent = comprehension.get("intent", "inform")
        if intent == "question":
            response_intent = "answer"
        elif intent == "command":
            response_intent = "confirm"
        else:
            response_intent = "inform"
        return {
            "intent": response_intent,
            "reference": comprehension.get("summary"),
            "key_terms": comprehension.get("key_terms", []),
        }
