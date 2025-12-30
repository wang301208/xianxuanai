"""
LLM service module.

Previously this layer proxied requests to external GPT/BERT style providers.
It now wraps the in-house language comprehension/production pipeline so the
rest of the system can keep the same interface while remaining self-contained.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from BrainSimulationSystem.models.language_processing import (
    AuditoryLexiconLearner,
    GrammarInducer,
    LanguageGenerator,
    PhonemeProcessor,
    SemanticNetwork,
    SyntaxProcessor,
    WordRecognizer,
)
from BrainSimulationSystem.models.language_comprehension import LanguageComprehension
from BrainSimulationSystem.models.language_production import LanguageProduction
from BrainSimulationSystem.models.intent_recognizer import IntentRecognizer
from BrainSimulationSystem.models.semantic_analyser import SemanticAnalyser
from BrainSimulationSystem.models.affect_analyser import AffectAnalyser
from BrainSimulationSystem.models.action_planner import ActionPlanner
from BrainSimulationSystem.models.working_memory import WorkingMemory
from BrainSimulationSystem.models.attention_diffuser import AttentionDiffuser
from BrainSimulationSystem.models.cognitive_controller import CognitiveController
from BrainSimulationSystem.models.semantic_layers import SemanticLayerManager
from BrainSimulationSystem.models.language_cortex import LanguageCortex
from BrainSimulationSystem.models.auditory_frontend import AuditoryFrontend
from BrainSimulationSystem.models.sequence_model import LightweightSequenceModel

logger = logging.getLogger(__name__)


def _default_logger() -> logging.Logger:
    _log = logging.getLogger("LLMService")
    if not _log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        _log.addHandler(handler)
        _log.setLevel(logging.INFO)
    return _log


@dataclass
class LLMResponse:
    """Standard response envelope returned by the compatibility layer."""

    text: str
    provider: str
    raw: Any
    latency: float
    meta: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "latency": self.latency,
            "meta": self.meta,
        }


class LLMService:
    """
    Compatibility wrapper for legacy LLM integrations.

    The interface remains close to the previous OpenAI adaptor, but requests are
    executed by the internal language pipeline (LanguageComprehension +
    LanguageProduction).
    """

    def __init__(
        self,
        provider: str = "internal",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 60,
        logger: Optional[logging.Logger] = None,
        pipeline: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        self.logger = logger or _default_logger()
        self.provider = provider or "internal"
        self.model = model or "language-cortex"
        self.timeout = timeout
        self.endpoint = endpoint
        self.mode = extra.get("mode", "default")
        self.pipeline_config = pipeline or extra.get("pipeline") or {}

        self.usage_stats: Dict[str, int] = {"requests": 0, "fallback": 0}
        self._lexicon_entries: set[str] = set()
        self._pipeline_ready = False
        self._last_comprehension: Dict[str, Any] = {}
        self._last_generation: Dict[str, Any] = {}

        self._init_pipeline()

    # ------------------------------------------------------------------#
    # Public helpers                                                    #
    # ------------------------------------------------------------------#
    def is_available(self) -> bool:
        return self._pipeline_ready

    def chat(
        self,
        messages: Iterable[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        messages = list(messages)
        start = time.perf_counter()

        if not messages:
            latency = time.perf_counter() - start
            return LLMResponse(
                text="",
                provider="internal_language_pipeline",
                raw={"comprehension": {}, "generation": None},
                latency=latency,
                meta={"empty": True},
            )

        try:
            text_input, context, audio_frames = self._prepare_inputs(messages)
            if not text_input:
                raise ValueError("No user utterance provided to LLMService.chat")

            proto_ids: List[int] = []
            if audio_frames:
                proto_ids = self.auditory_frontend.stream_frames(
                    audio_frames, context_tokens=text_input.split()
                )
                if proto_ids:
                    context = dict(context)
                    context["proto_phonemes"] = proto_ids

            structured, semantic_info, affect_info, action_plan, controller = self.comprehension.process(
                text_input,
                {"language_context": context},
            )
            structured["_semantic_obj"] = semantic_info
            structured["_affect_obj"] = affect_info
            structured["_action_plan_obj"] = action_plan
            structured["controller"] = controller
            self._last_comprehension = structured

            sanitized = self._sanitize_structured(structured)
            raw_payload: Dict[str, Any] = {"comprehension": sanitized}
            meta: Dict[str, Any] = {
                "mode": "internal_pipeline",
                "intent": sanitized.get("intent"),
                "confidence": sanitized.get("confidence"),
            }

            if response_format == "json_object":
                payload_text = json.dumps(sanitized, ensure_ascii=False)
                generation_payload = None
            else:
                generation_payload = self.production.generate(
                    controller.get("generation_goal") if controller else None,
                    {"language_context": context},
                    structured,
                )
                self._last_generation = generation_payload
                payload_text = generation_payload.get("reply", "")
                raw_payload["generation"] = self._sanitize_structured(generation_payload)
                meta["generation_intent"] = generation_payload.get("goal", {}).get("intent")

            latency = time.perf_counter() - start
            return LLMResponse(
                text=payload_text,
                provider="internal_language_pipeline",
                raw=raw_payload,
                latency=latency,
                meta=meta,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            self.logger.exception("Internal language pipeline failed; using heuristic fallback", exc_info=exc)
            self.usage_stats["fallback"] += 1
            fallback_text = self._heuristic_reply(messages, response_format=response_format)
            latency = time.perf_counter() - start
            return LLMResponse(
                text=fallback_text,
                provider="fallback",
                raw={"messages": messages, "error": str(exc)},
                latency=latency,
                meta={"fallback": True},
            )

    def structured_chat(
        self,
        instructions: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt = instructions.get("system", "")
        user_prompt = inputs.get("user") or ""
        history = inputs.get("history") or []

        prompt_messages: List[Dict[str, str]] = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.extend(history)
        prompt_messages.append({"role": "user", "content": user_prompt})

        response = self.chat(prompt_messages, response_format="json_object")
        parsed = self._try_parse_json(response.text)
        if parsed is None:
            parsed = {"text": response.text, "provider": response.provider}
        parsed["_meta"] = {"provider": response.provider, "latency": response.latency}
        return parsed

    def query(
        self,
        prompt: str,
        *,
        response_format: str = "json_object",
    ) -> Any:
        response = self.chat(
            [{"role": "user", "content": prompt}],
            response_format=response_format,
        )
        if response_format == "json_object":
            return self._try_parse_json(response.text) or {}
        return response.text

    # ------------------------------------------------------------------#
    # Internal helpers                                                  #
    # ------------------------------------------------------------------#
    def _init_pipeline(self) -> None:
        cfg = self.pipeline_config
        self.phoneme_processor = PhonemeProcessor(cfg.get("phoneme", {}))
        self.semantic_network = SemanticNetwork(cfg.get("semantic", {}))
        self.word_recognizer = WordRecognizer(cfg.get("lexicon", {}), self.semantic_network)
        self.sequence_model = LightweightSequenceModel(cfg.get("sequence_model", {}))
        self.syntax_processor = SyntaxProcessor(cfg.get("syntax", {}))
        self.syntax_processor.set_sequence_model(self.sequence_model)
        self.word_recognizer.set_syntax_processor(self.syntax_processor)
        self.grammar_inducer = GrammarInducer(
            self.syntax_processor,
            cfg.get("grammar_induction", {}),
            sequence_model=self.sequence_model,
        )
        self.semantic_analyser = SemanticAnalyser(self.semantic_network, cfg.get("semantics", {}))
        self.affect_analyser = AffectAnalyser(cfg.get("affect", {}))
        self.action_planner = ActionPlanner(cfg.get("actions", {}))
        self.working_memory = WorkingMemory(cfg.get("memory", {}))
        self.attention_diffuser = AttentionDiffuser(cfg.get("attention", {}))
        self.cognitive_controller = CognitiveController(cfg.get("controller", {}))
        self.semantic_layers = SemanticLayerManager(self.semantic_network, cfg.get("semantic_layers", {}))
        self.language_cortex = LanguageCortex(cfg.get("cortex", {}))
        self.language_generator = LanguageGenerator(cfg.get("generation", {}))
        self.intent_recognizer = IntentRecognizer(cfg.get("intent", {}))

        self.auditory_learner = AuditoryLexiconLearner(
            self.word_recognizer,
            self.semantic_network,
            cfg.get("phoneme_discovery", {}),
        )
        self.auditory_frontend = AuditoryFrontend(
            self.word_recognizer,
            self.semantic_network,
            self.auditory_learner,
            cfg.get("auditory_frontend", {}),
        )

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
        self._pipeline_ready = True

    def _prepare_inputs(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], Optional[Sequence[Sequence[float]]]]:
        history: List[Dict[str, str]] = []
        latest_user = ""
        latest_audio: Optional[Sequence[Sequence[float]]] = None
        for msg in messages:
            role = (msg.get("role") or "user").lower()
            content = msg.get("content", "")
            history.append({"role": role, "content": content})
            if role == "user":
                latest_user = content
                for key in ("audio_features", "audio_frames", "auditory_features"):
                    if key in msg and msg[key] is not None:
                        latest_audio = msg[key]
                        break

        context: Dict[str, Any] = {}
        if history:
            context["history"] = history[:-1] if len(history) > 1 else []
            context["last_role"] = history[-1]["role"]
        if latest_user:
            context["last_user_message"] = latest_user
        return latest_user, context, latest_audio

    def _sanitize_structured(self, data: Dict[str, Any]) -> Dict[str, Any]:
        filtered = {k: v for k, v in data.items() if not k.startswith("_")}
        return self._make_serializable(filtered)

    def _make_serializable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._make_serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._make_serializable(v) for v in value]
        if isinstance(value, tuple):
            return [self._make_serializable(v) for v in value]
        if isinstance(value, set):
            return [self._make_serializable(v) for v in sorted(value)]
        if hasattr(value, "to_dict"):
            return self._make_serializable(value.to_dict())
        if hasattr(value, "model_dump"):
            return self._make_serializable(value.model_dump())
        if hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool, type(None))):
            public = {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
            return self._make_serializable(public)
        return value

    @staticmethod
    def _try_parse_json(payload: str) -> Optional[Dict[str, Any]]:
        payload = (payload or "").strip()
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            start = payload.find("{")
            end = payload.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(payload[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _heuristic_reply(
        messages: Sequence[Dict[str, str]],
        *,
        response_format: Optional[str],
    ) -> str:
        if not messages:
            return ""
        last = messages[-1].get("content", "")
        if response_format == "json_object":
            intent = "question" if "?" in last else "statement"
            raw_terms = [
                token.strip(".,!?")
                for token in last.split()
                if len(token.strip(".,!?")) > 3
            ]
            stopwords = {"key_terms", "summary", "intent", "action_items", "context", "snapshot"}
            key_terms: List[str] = []
            for term in raw_terms:
                lower = term.lower()
                if lower in stopwords:
                    continue
                if lower not in key_terms:
                    key_terms.append(lower)
            summary = " ".join(last.split()[:20])
            data = {
                "intent": intent,
                "key_terms": key_terms[:8],
                "summary": summary,
                "confidence": round(random.uniform(0.45, 0.75), 3),
                "fallback": True,
            }
            return json.dumps(data, ensure_ascii=False)
        return f"(fallback response) {last[:200]}"

    @classmethod
    def from_config(cls, params: Optional[Dict[str, Any]] = None) -> "LLMService":
        params = params or {}
        pipeline_cfg = params.get("pipeline") or {}
        return cls(
            provider=params.get("provider", "internal"),
            model=params.get("model"),
            api_key=params.get("api_key"),
            endpoint=params.get("endpoint"),
            timeout=params.get("timeout", 60),
            pipeline=pipeline_cfg,
        )


__all__ = ["LLMService", "LLMResponse"]
