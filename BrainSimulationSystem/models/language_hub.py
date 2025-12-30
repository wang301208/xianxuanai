"""
语言中枢模块

将传统的语言处理子系统（音系、语义、句法）与大型语言模型（LLM）结合，
# Broca-style production                                                  #
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from BrainSimulationSystem.models.cognitive_base import CognitiveProcess
from BrainSimulationSystem.models.language_processing import (
    AuditoryLexiconLearner,
    GrammarInducer,
    LanguageGenerator,
    PhonemeProcessor,
    SemanticNetwork,
    SyntaxProcessor,
    WordRecognizer,
)
from BrainSimulationSystem.models.intent_recognizer import IntentRecognizer, IntentResult
from BrainSimulationSystem.models.semantic_analyser import SemanticAnalyser, SemanticAnalysisResult
from BrainSimulationSystem.models.affect_analyser import AffectAnalyser, AffectAnalysisResult
from BrainSimulationSystem.models.action_planner import ActionPlanner, ActionPlan
from BrainSimulationSystem.models.working_memory import WorkingMemory
from BrainSimulationSystem.models.attention_diffuser import AttentionDiffuser
from BrainSimulationSystem.models.cognitive_controller import CognitiveController
from BrainSimulationSystem.models.semantic_layers import SemanticLayerManager
from BrainSimulationSystem.models.language_cortex import LanguageCortex
from BrainSimulationSystem.models.language_comprehension import LanguageComprehension
from BrainSimulationSystem.models.language_production import LanguageProduction
from BrainSimulationSystem.models.dialogue_state import DialogueStateTracker
from BrainSimulationSystem.models.semantic_parser import SemanticFallbackParser
from BrainSimulationSystem.learning.feedback_store import FeedbackLogger
from BrainSimulationSystem.models.auditory_frontend import AuditoryFrontend
from BrainSimulationSystem.models.sequence_model import LightweightSequenceModel

if TYPE_CHECKING:
    from BrainSimulationSystem.integration.llm_service import LLMService

logger = logging.getLogger(__name__)


class LanguageHub(CognitiveProcess):
# Broca-style production                                                  #

    def __init__(
        self,
        network: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        llm_service: Optional["LLMService"] = None,
    ) -> None:
        super().__init__(network, params or {}, name="language_hub")
        self.params.setdefault("comprehension_prompt", DEFAULT_COMPREHENSION_PROMPT)
        self.params.setdefault("generation_prompt", DEFAULT_GENERATION_PROMPT)
        self.params.setdefault("llm", {})
        self.params.setdefault("intent", {})
        self.params.setdefault("affect", {})
        self.params.setdefault("actions", {})
        self.params.setdefault("memory", {})
        self.params.setdefault("attention", {})
        self.params.setdefault("controller", {})
        self.params.setdefault("semantic_layers", {})
        self.params.setdefault("feedback", {})
        self.params.setdefault("dialogue_state", {})
        self.params.setdefault("semantic_fallback", {})
        self.llm = llm_service
        self.llm_available = bool(
            self.llm and getattr(self.llm, "is_available", lambda: True)()
        )
        self.phoneme_processor = PhonemeProcessor(self.params.get("phoneme", {}))
        self.semantic_network = SemanticNetwork(self.params.get("semantic", {}))
        self.word_recognizer = WordRecognizer(self.params.get("lexicon", {}), self.semantic_network)
        self.sequence_model = LightweightSequenceModel(self.params.get("sequence_model", {}))
        self.syntax_processor = SyntaxProcessor(self.params.get("syntax", {}))
        self.syntax_processor.set_sequence_model(self.sequence_model)
        self.word_recognizer.set_syntax_processor(self.syntax_processor)
        self.grammar_inducer = GrammarInducer(
            self.syntax_processor,
            self.params.get("grammar_induction", {}),
            sequence_model=self.sequence_model,
        )
        self.semantic_analyser = SemanticAnalyser(self.semantic_network, self.params.get("semantics", {}))
        self.affect_analyser = AffectAnalyser(self.params.get("affect", {}))
        self.action_planner = ActionPlanner(self.params.get("actions", {}))
        self.working_memory = WorkingMemory(self.params.get("memory", {}))
        self.attention_diffuser = AttentionDiffuser(self.params.get("attention", {}))
        self.cognitive_controller = CognitiveController(self.params.get("controller", {}))
        self.semantic_layers = SemanticLayerManager(self.semantic_network, self.params.get("semantic_layers", {}))
        self.language_cortex = LanguageCortex(self.params.get("cortex", {}))
        self.language_generator = LanguageGenerator(self.params.get("generation", {}))
        if hasattr(self.language_generator, "set_sequence_model"):
            try:
                self.language_generator.set_sequence_model(self.sequence_model)
            except Exception:
                pass
        self.intent_recognizer = IntentRecognizer(self.params.get("intent", {}))
        self.dialogue_state = DialogueStateTracker(self.params.get("dialogue_state", {}))
        fallback_cfg = self.params.get("semantic_fallback", {})
        fallback_enabled = fallback_cfg.get("enabled", True)
        self.semantic_fallback_threshold = float(fallback_cfg.get("confidence_threshold", 0.58))
        self.semantic_fallback = (
            SemanticFallbackParser(self.llm, fallback_cfg) if fallback_enabled else None
        )
        self._lexicon_entries: set[str] = set()
        self.usage_stats = {"requests": 0, "fallback": 0}
        self.internal_review_threshold = float(self.params.get("internal_review_threshold", 0.45))
        feedback_params = self.params.get("feedback", {})
        feedback_path = feedback_params.get("path")
        self.feedback_logger = FeedbackLogger(
            feedback_path,
            enabled=feedback_params.get("enabled", True),
            max_cache=int(feedback_params.get("max_cache", 256)),
        )
        self.feedback_threshold = float(feedback_params.get("low_confidence_threshold", 0.5))
        self.high_confidence_threshold = float(feedback_params.get("high_confidence_threshold", 0.75))
        self.learning_cache_limit = int(feedback_params.get("learning_cache_limit", 128))
        self.learning_cache: List[Dict[str, Any]] = []
        self.migration_ratio = float(self.params.get("migration_ratio", 0.3))
        self.migration_increment = float(self.params.get("migration_increment", 0.02))
        self.migration_decay = float(self.params.get("migration_decay", 0.005))
        self.comprehension = LanguageComprehension(
            self.phoneme_processor,
            self.word_recognizer,
            self.syntax_processor,
            self.semantic_network,
            self.semantic_analyser,
            self.intent_recognizer,
            self.affect_analyser,
            self.working_memory,
            self.attention_diffuser,
            self.cognitive_controller,
            self.action_planner,
            self.semantic_layers,
            self.usage_stats,
            self._lexicon_entries,
            self.language_cortex,
            grammar_inducer=self.grammar_inducer,
            sequence_model=self.sequence_model,
            semantic_fallback=self.semantic_fallback,
            semantic_fallback_threshold=self.semantic_fallback_threshold,
        )
        self.production = LanguageProduction(
            self.language_generator,
            self.syntax_processor,
            self.action_planner,
            self.cognitive_controller,
            self.working_memory,
            self.semantic_network,
            self.usage_stats,
            grammar_inducer=self.grammar_inducer,
            sequence_model=self.sequence_model,
        )
        self.auditory_learner = AuditoryLexiconLearner(
            self.word_recognizer,
            self.semantic_network,
            self.params.get("phoneme_discovery", {}),
        )
        self.auditory_frontend = AuditoryFrontend(
            self.word_recognizer,
            self.semantic_network,
            self.auditory_learner,
            self.params.get("auditory_frontend", {}),
        )
        self.last_comprehension: Dict[str, Any] = {}
        self.last_generation: Dict[str, Any] = {}
        self._ensure_intent_nodes()

    # ------------------------------------------------------------------ #
    # 核心对外接口                                                         #
    # ------------------------------------------------------------------ #
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一处理入口。

        支持的输入字段：
            - language_input: 待理解的原始文本
            - language_context: 上下文（记忆、情境等）
            - language_generation: 生成指令或目标
            - generate_language_response: 是否强制生成回答
        """
        results: Dict[str, Any] = {
            "status": "idle",
            "llm_available": self.llm_available,
            "migration_ratio": round(self.migration_ratio, 3),
        }

        text_input: Optional[str] = inputs.get("language_input")
        tokens_from_text = text_input.split() if text_input else []

        proto_ids = self._ingest_audio_frames(inputs, tokens_from_text)
        if proto_ids:
            results["proto_phonemes"] = proto_ids

        dialogue_snapshot = self.dialogue_state.snapshot()
        results["dialogue_state"] = dialogue_snapshot

        base_context = dict(inputs.get("language_context") or {})
        base_context.setdefault("dialogue_state", dialogue_snapshot)
        if "attention_directives" in inputs and "attention_directives" not in base_context:
            base_context["attention_directives"] = inputs["attention_directives"]
        comprehension_inputs = dict(inputs)
        comprehension_inputs["language_context"] = base_context

        generation_goal: Optional[Any] = inputs.get("language_generation")
        if text_input:
            comprehension = self._comprehend(text_input, comprehension_inputs)
            results["comprehension"] = comprehension
            results["status"] = "processing"
            updated_snapshot = self.dialogue_state.update(
                comprehension,
                speaker=inputs.get("speaker", "user"),
            )
            self.dialogue_state.ingest_actions(comprehension.get("action_items", []))
            comprehension["dialogue_state"] = updated_snapshot
            results["dialogue_state"] = updated_snapshot
        if generation_goal or inputs.get("generate_language_response"):
            generation_context = dict(comprehension_inputs.get("language_context", {}))
            generation_context["dialogue_state"] = results.get(
                "dialogue_state", self.dialogue_state.snapshot()
            )
            production_inputs = dict(inputs)
            production_inputs["language_context"] = generation_context
            production = self._produce(generation_goal, production_inputs)
            results["generation"] = production
            results["status"] = "responding"

        results["usage_stats"] = self.usage_stats.copy()
        return results

    # ------------------------------------------------------------------ #
    # 语言理解（Wernicke 区）                                               #
    # ------------------------------------------------------------------ #
    def _comprehend(self, text: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate to LanguageComprehension pipeline."""
        structured, semantic_info, affect_info, action_plan, controller_flags = self.comprehension.process(text, inputs)
        self.last_comprehension = structured
        self.last_comprehension["_semantic_obj"] = semantic_info
        self.last_comprehension["_affect_obj"] = affect_info
        self.last_comprehension["_action_plan_obj"] = action_plan
        self.last_comprehension["controller"] = controller_flags
        self._reinforce_if_unconfident(self.last_comprehension)
        self._apply_pronoun_resolution(self.last_comprehension)
        self._log_feedback(self.last_comprehension)
        self._update_learning_schedule(self.last_comprehension)
        return self.last_comprehension

    def _ingest_audio_frames(self, inputs: Dict[str, Any], context_tokens: Sequence[str]) -> List[int]:
        frames = None
        for key in ("audio_frames", "audio_features", "language_audio", "auditory_features"):
            candidate = inputs.get(key)
            if candidate is not None:
                frames = candidate
                break

        if frames is None or self.auditory_frontend is None:
            return []

        try:
            return self.auditory_frontend.stream_frames(frames, context_tokens=context_tokens)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Auditory frontend ingestion failed: %s", exc)
            return []

    def _build_comprehension_messages(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        system_prompt = self.params.get("comprehension_prompt", DEFAULT_COMPREHENSION_PROMPT)
        context_snippet = json.dumps(context, ensure_ascii=False) if context else "{}"
        user_prompt = (
            "Raw input: \"{text}\"\n"
            "Context snapshot: {context_snippet}\n"
            "Analyse the language and return intent, key facts, entities, emotional cues and potential follow-up actions."
        ).format(text=text, context_snippet=context_snippet)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _produce(self, goal: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate to LanguageProduction pipeline."""
        production = self.production.generate(goal, inputs, self.last_comprehension or {})
        self.last_generation = production
        return production

    def _ensure_intent_nodes(self) -> None:
        """Ensure semantic intent nodes exist for activation updates."""
        for concept in ("question", "command", "statement", "greeting"):
            if concept not in self.semantic_network.nodes:
                self.semantic_network.add_node(concept)

    def _reinforce_if_unconfident(self, comprehension: Dict[str, Any]) -> None:
        confidence = float(comprehension.get("confidence", 0.0) or 0.0)
        if confidence >= self.internal_review_threshold:
            return

        controller_flags = comprehension.setdefault("controller", {})
        controller_flags["internal_review"] = True

        focus_terms: List[str] = list(comprehension.get("key_terms", []))
        if not focus_terms:
            focus_terms = [token for token in comprehension.get("normalized_tokens", [])[:4] if token]
        if focus_terms:
            self.attention_diffuser.refine_focus(self.semantic_network, focus_terms)
            for concept in focus_terms:
                self.semantic_network.activate_concept(concept, amount=0.05)
            controller_flags["focus_terms"] = list(dict.fromkeys(focus_terms))

        normalized_tokens: Sequence[str] = comprehension.get("normalized_tokens", [])
        if normalized_tokens:
            second_pass = self.intent_recognizer.classify(
                comprehension.get("input", ""),
                normalized_tokens,
                self.semantic_network,
                self.syntax_processor,
                cortex_vector=comprehension.get("cortex_context"),
            )
            comprehension.setdefault("internal_review", {})["intent_second_pass"] = {
                "label": second_pass.label,
                "confidence": second_pass.confidence,
                "source": second_pass.source,
                "scores": second_pass.scores,
            }
            if second_pass.confidence > comprehension.get("intent_confidence", 0.0):
                comprehension["intent"] = second_pass.label
                comprehension["intent_confidence"] = second_pass.confidence

            if self.language_cortex is not None:
                target_vector = self.language_cortex.predict_intent_vector(
                    self.language_cortex.encode_tokens(normalized_tokens)
                )
                self.language_cortex.update_from_example(normalized_tokens, target_vector)
                comprehension.setdefault("internal_review", {})["intent_vector_reinforced"] = target_vector.tolist()

        comprehension["confidence"] = max(confidence, min(0.55, confidence + 0.08))

    def _apply_pronoun_resolution(self, comprehension: Dict[str, Any]) -> None:
        tokens = comprehension.get("normalized_tokens") or []
        if not tokens:
            return
        resolved_items: List[Dict[str, Any]] = []
        for index, token in enumerate(tokens):
            if token.lower() not in _PRONOUN_TOKENS:
                continue
            resolved = self.working_memory.resolve_pronoun(token)
            if not resolved:
                continue
            resolved_items.append({"pronoun": token, "resolved": resolved, "index": index})
            self.semantic_network.add_relation(resolved.lower(), token, "pronoun_reference", strength=0.2)
        if resolved_items:
            comprehension["resolved_references"] = resolved_items
            self.working_memory.update_last_record({"resolved_references": resolved_items})

    def _log_feedback(self, comprehension: Dict[str, Any]) -> None:
        confidence = float(comprehension.get("confidence", 0.0))
        entry = {
            "text": comprehension.get("input"),
            "intent": comprehension.get("intent"),
            "intent_confidence": comprehension.get("intent_confidence"),
            "confidence": confidence,
            "key_terms": comprehension.get("key_terms"),
            "action_items": comprehension.get("action_items"),
            "resolved_references": comprehension.get("resolved_references", []),
            "turn": self.usage_stats["requests"],
        }
        if confidence < self.feedback_threshold:
            entry["reason"] = "low_confidence"
            self.feedback_logger.record(entry)
        elif confidence >= self.high_confidence_threshold:
            self._cache_learning_example(entry)

    def _cache_learning_example(self, entry: Dict[str, Any]) -> None:
        self.learning_cache.append(entry)
        if len(self.learning_cache) > self.learning_cache_limit:
            self.learning_cache.pop(0)

    def _update_learning_schedule(self, comprehension: Dict[str, Any]) -> None:
        confidence = float(comprehension.get("confidence", 0.0))
        if confidence >= self.high_confidence_threshold:
            self.migration_ratio = min(1.0, self.migration_ratio + self.migration_increment)
        else:
            self.migration_ratio = max(0.0, self.migration_ratio - self.migration_decay)
        comprehension["migration_ratio"] = self.migration_ratio

    # ------------------------------------------------------------------ #
    # 辅助方法                                                              #
    # ------------------------------------------------------------------ #
    def get_last_comprehension(self) -> Dict[str, Any]:
        return self.last_comprehension

    def get_last_generation(self) -> Dict[str, Any]:
        return self.last_generation

    def drain_learning_cache(self) -> List[Dict[str, Any]]:
        cache = self.learning_cache[:]
        self.learning_cache.clear()
        return cache


_PRONOUN_TOKENS = {
    "he",
    "she",
    "they",
    "them",
    "him",
    "her",
    "it",
    "its",
    "himself",
    "herself",
    "themselves",
    "their",
    "theirs",
    "itself",
}


DEFAULT_COMPREHENSION_PROMPT = (
    'You are the language comprehension hub of a cortical simulation (analogous to the Wernicke area). '
    'Read the user input and return a JSON object capturing intent, key_terms, summary, action_items, and emotions.'
)

DEFAULT_GENERATION_PROMPT = (
# Broca-style production                                                  #
    'Given a communication goal, craft a concise, coherent and empathetic reply on behalf of the agent.'
)


__all__ = ["LanguageHub"]
