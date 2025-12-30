"""
Language comprehension module (Wernicke analogue).

This class encapsulates the internal pipeline for understanding an input
utterance using the project's native components instead of external LLM calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from BrainSimulationSystem.models.intent_recognizer import IntentResult
from BrainSimulationSystem.models.semantic_analyser import SemanticAnalysisResult
from BrainSimulationSystem.models.affect_analyser import AffectAnalysisResult
from BrainSimulationSystem.models.action_planner import ActionPlan
from BrainSimulationSystem.models.semantic_parser import SemanticFallbackParser


class LanguageComprehension:
    """Pipeline wrapper for language understanding."""

    def __init__(
        self,
        phoneme_processor,
        word_recognizer,
        syntax_processor,
        semantic_network,
        semantic_analyser,
        intent_recognizer,
        affect_analyser,
        working_memory,
        attention_diffuser,
        cognitive_controller,
        action_planner,
        semantic_layers,
        usage_stats: Dict[str, int],
        lexicon_entries: Optional[set] = None,
        language_cortex=None,
        grammar_inducer=None,
        sequence_model=None,
        *,
        semantic_fallback: Optional[SemanticFallbackParser] = None,
        semantic_fallback_threshold: float = 0.55,
    ) -> None:
        self.phoneme_processor = phoneme_processor
        self.word_recognizer = word_recognizer
        self.syntax_processor = syntax_processor
        self.semantic_network = semantic_network
        self.semantic_analyser = semantic_analyser
        self.intent_recognizer = intent_recognizer
        self.affect_analyser = affect_analyser
        self.working_memory = working_memory
        self.attention_diffuser = attention_diffuser
        self.cognitive_controller = cognitive_controller
        self.action_planner = action_planner
        self.semantic_layers = semantic_layers
        self.usage_stats = usage_stats
        self.lexicon_entries = lexicon_entries if lexicon_entries is not None else set()
        self.language_cortex = language_cortex
        self.grammar_inducer = grammar_inducer
        self.sequence_model = sequence_model
        self.semantic_fallback = semantic_fallback
        self.semantic_fallback_threshold = float(semantic_fallback_threshold)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def process(
        self,
        text: str,
        inputs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], SemanticAnalysisResult, AffectAnalysisResult, ActionPlan, Dict[str, Any]]:
        context = inputs.get("language_context") or {}
        memory_state = self.working_memory.get_state()
        self.semantic_layers.start_turn({"text": text, "intent_hint": memory_state.get("recent_intent")})

        phonemes = [self.phoneme_processor.process_phoneme(p) for p in text.lower()]
        raw_tokens = text.split()
        clean_tokens: List[str] = []
        normalized_sequence: List[str] = []
        normalized_tokens: List[str] = []

        for token in raw_tokens:
            cleaned = token.strip(".,!?\"'")
            clean_tokens.append(cleaned)
            normalized = cleaned.lower()
            normalized_sequence.append(normalized)
            if not normalized:
                continue
            normalized_tokens.append(normalized)
            if normalized not in self.lexicon_entries:
                self.word_recognizer.add_word(normalized, list(normalized), concept=normalized)
                self.lexicon_entries.add(normalized)
            self.word_recognizer.activate_word(normalized, amount=0.05)

        cortex_context = None
        if self.language_cortex is not None:
            cortex_context = self.language_cortex.encode_tokens(normalized_tokens)

        attention_directives = context.get("attention_directives") if isinstance(context, dict) else None
        focus_terms = None
        if attention_directives:
            focus_terms = attention_directives.get("semantic_focus")
        self.attention_diffuser.apply(
            self.semantic_network,
            memory_state,
            normalized_tokens,
            focus_terms=focus_terms,
            directives=attention_directives,
        )

        syntax_structure = self.syntax_processor.parse_sentence(clean_tokens or raw_tokens)
        semantic_info = self.semantic_analyser.analyse(
            text,
            clean_tokens or raw_tokens,
            normalized_sequence,
            syntax_structure,
        )
        multimodal_context = self._ingest_multimodal_context(inputs, context, semantic_info)
        if self.grammar_inducer:
            self.grammar_inducer.observe_sentence(clean_tokens or raw_tokens)
        elif self.sequence_model:
            self.sequence_model.observe_sequence(clean_tokens or raw_tokens, syntax_structure.get("pos", []))
        self.semantic_layers.ingest_semantic_info(semantic_info)

        intent_result = self.intent_recognizer.classify(
            text,
            normalized_tokens,
            self.semantic_network,
            self.syntax_processor,
            cortex_vector=cortex_context.tolist() if cortex_context is not None else None,
        )
        affect_info = self.affect_analyser.analyse(
            text,
            clean_tokens or raw_tokens,
            normalized_sequence,
            intent_result.label,
        )

        controller_flags = self.cognitive_controller.decide(
            intent_result.label,
            {"actions": [], "confidence": semantic_info.confidence},
            affect_info,
            memory_state,
            semantic_info,
        )
        self.attention_diffuser.refine_focus(
            self.semantic_network,
            controller_flags.get("focus_terms", []),
        )
        self.semantic_layers.add_focus_terms(controller_flags.get("focus_terms", []))

        action_plan = self.action_planner.plan(
            intent_result.label,
            semantic_info.suggested_actions,
            semantic_info.relations,
            affect_info.tone,
            context,
        )

        fallback_semantics = None
        if (
            self.semantic_fallback is not None
            and semantic_info.confidence < self.semantic_fallback_threshold
        ):
            fallback_semantics = self.semantic_fallback.parse(
                text,
                context=self._prepare_fallback_context(context, memory_state),
                existing=semantic_info,
            )
            if fallback_semantics:
                self._merge_fallback_semantics(semantic_info, fallback_semantics)

        structured = self._compose_result(
            text,
            phonemes,
            clean_tokens or raw_tokens,
            normalized_tokens,
            semantic_info,
            syntax_structure,
            intent_result,
            affect_info,
            action_plan,
            controller_flags,
            context,
            memory_state,
            cortex_context,
            attention_directives,
        )
        if multimodal_context:
            structured["multimodal"] = multimodal_context

        self.working_memory.add_record(
            {
                "type": "input",
                "text": text,
                "summary": structured.get("summary"),
                "intent": intent_result.label,
                "key_terms": list(structured.get("key_terms", [])),
                "tone": affect_info.tone,
                "polarity": affect_info.polarity,
                "actions": list(action_plan.actions),
                "status": "pending" if action_plan.actions else "not_applicable",
                "semantic_layers": structured.get("semantic_layers"),
            }
        )
        self.usage_stats["requests"] += 1

        structured["_semantic_obj"] = semantic_info
        structured["_affect_obj"] = affect_info
        structured["_action_plan_obj"] = action_plan
        structured["controller"] = controller_flags
        if fallback_semantics:
            structured["semantic_fallback"] = fallback_semantics

        return structured, semantic_info, affect_info, action_plan, controller_flags

    def _ingest_multimodal_context(
        self,
        inputs: Dict[str, Any],
        context: Any,
        semantic_info: Optional[SemanticAnalysisResult] = None,
    ) -> Dict[str, Any]:
        """Best-effort bridge for non-text modalities into the semantic network.

        This enables simple "visual-language" grounding when upstream perception
        modules provide symbolic labels (e.g. detected objects) via context.
        """

        concepts: List[str] = []

        def _add(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _add(item)
                return
            text = str(value).strip()
            if not text:
                return
            concepts.append(text)

        sources = []
        if isinstance(inputs, dict):
            sources.append(inputs)
        if isinstance(context, dict):
            sources.append(context)

        for src in sources:
            for key in (
                "visual_concepts",
                "vision_concepts",
                "visual_labels",
                "vision_labels",
                "image_concepts",
            ):
                if key in src:
                    _add(src.get(key))

            perception = src.get("perception")
            if isinstance(perception, dict):
                vision = perception.get("vision") or perception.get("visual")
                if isinstance(vision, dict):
                    _add(vision.get("concepts") or vision.get("labels"))

        deduped: List[str] = []
        seen = set()
        for raw in concepts:
            key = raw.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)

        if not deduped:
            return {}

        scene_node = "visual_scene"
        self.semantic_network.add_node(scene_node, {"source": "multimodal"})

        for concept in deduped[:16]:
            self.semantic_network.add_node(concept, {"source": "multimodal"})
            self.semantic_network.activate_concept(concept, amount=0.15)
            self.semantic_network.add_relation(scene_node, concept, "perceives", strength=0.35)
            self.semantic_network.add_relation(concept, scene_node, "in_scene", strength=0.2)
            if semantic_info is not None:
                if concept not in semantic_info.key_terms:
                    semantic_info.key_terms.append(concept)

        return {"visual_concepts": deduped[:16]}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _compose_result(
        self,
        text: str,
        phonemes: Sequence[str],
        tokens: Sequence[str],
        normalized_tokens: Sequence[str],
        semantic_info: SemanticAnalysisResult,
        syntax_structure: Dict[str, Any],
        intent_result: IntentResult,
        affect_info: AffectAnalysisResult,
        action_plan: ActionPlan,
        controller_flags: Dict[str, Any],
        context: Dict[str, Any],
        memory_state: Dict[str, Any],
        cortex_context: Optional[Any],
        attention_directives: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary = self._generate_summary(text, semantic_info)
        action_items = list(action_plan.actions) or semantic_info.suggested_actions
        if intent_result.label == "command" and not action_items:
            action_items.append("execute_request")

        combined_confidence = min(
            0.99,
            0.5 * semantic_info.confidence
            + 0.3 * affect_info.confidence
            + 0.2 * action_plan.confidence,
        )

        structured = {
            "intent": intent_result.label,
            "intent_confidence": intent_result.confidence,
            "intent_source": intent_result.source,
            "intent_scores": intent_result.scores,
            "intent_details": intent_result.details,
            "summary": summary,
            "key_terms": semantic_info.key_terms,
            "entities": semantic_info.entities,
            "relations": semantic_info.relations,
            "activation_map": semantic_info.activation_map,
            "action_items": action_items,
            "emotions": affect_info.emotions,
            "tone": affect_info.tone,
            "sentiment": affect_info.polarity,
            "affect_score": affect_info.score,
            "confidence": combined_confidence,
            "syntax": syntax_structure,
            "context_snapshot": context,
            "semantic_layers": {},
            "llm": {},
            "llm_raw": {},
            "input": text,
            "phonemes": list(phonemes),
            "tokens": list(tokens),
            "normalized_tokens": list(normalized_tokens),
            "context": context,
            "memory_state": memory_state,
            "semantic": {
                "concept_stats": semantic_info.concept_stats,
                "relations": semantic_info.relations,
                "entities": semantic_info.entities,
                "suggested_actions": semantic_info.suggested_actions,
            },
            "affect": {
                "polarity": affect_info.polarity,
                "tone": affect_info.tone,
                "score": affect_info.score,
                "confidence": affect_info.confidence,
                "evidence": affect_info.evidence,
            },
        }
        if attention_directives:
            structured["attention_directives"] = attention_directives
        structured["pending_actions"] = list(action_plan.actions)

        consolidation = self.semantic_layers.commit_if_needed(
            semantic_info.confidence,
            controller_flags.get("focus_terms", []),
        )
        structured["semantic_layers"] = {
            "local_summary": self.semantic_layers.get_local_summary(),
            "consolidation": consolidation,
        }
        if cortex_context is not None:
            intent_vector = self.language_cortex.predict_intent_vector(cortex_context)
            structured["cortex_context"] = cortex_context.tolist()
            structured["intent_vector"] = intent_vector.tolist()
        else:
            structured["cortex_context"] = []
            structured["intent_vector"] = []

        self.semantic_network.activate_concept(
            intent_result.label,
            amount=min(0.45, 0.15 + intent_result.confidence * 0.5),
        )
        return structured

    def _merge_fallback_semantics(
        self,
        semantic_info: SemanticAnalysisResult,
        fallback: Dict[str, Any],
    ) -> None:
        key_terms = fallback.get("key_terms") or []
        for term in key_terms:
            if term and term not in semantic_info.key_terms:
                semantic_info.key_terms.append(term)

        fallback_relations = fallback.get("relations") or []
        for relation in fallback_relations:
            if relation and relation not in semantic_info.relations:
                semantic_info.relations.append(relation)

        fallback_entities = fallback.get("entities") or []
        for entity in fallback_entities:
            if entity and entity not in semantic_info.entities:
                semantic_info.entities.append(entity)

        fallback_actions = fallback.get("suggested_actions") or fallback.get("actions") or []
        for action in fallback_actions:
            if action and action not in semantic_info.suggested_actions:
                semantic_info.suggested_actions.append(action)

        fallback_confidence = fallback.get("confidence")
        if isinstance(fallback_confidence, (int, float)):
            semantic_info.confidence = max(float(fallback_confidence), semantic_info.confidence)

    @staticmethod
    def _prepare_fallback_context(
        context: Dict[str, Any],
        memory_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        snapshot = {
            "dialogue_state": context.get("dialogue_state"),
            "topic": context.get("topic"),
            "memory_highlights": memory_state.get("recent") if isinstance(memory_state, dict) else None,
        }
        return {key: value for key, value in snapshot.items() if value is not None}

    @staticmethod
    def _generate_summary(text: str, semantic_info: SemanticAnalysisResult) -> str:
        clean = text.strip()
        if not clean:
            return ""
        if len(clean) <= 160:
            return clean
        key_terms = semantic_info.key_terms[:3]
        if key_terms:
            return f"Focus on {', '.join(key_terms)}."
        return clean[:160] + "..."
