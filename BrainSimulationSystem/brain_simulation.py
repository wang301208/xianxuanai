"""Brain simulation system module.

大脑模拟系统模块。

Provides integrated orchestration of perception, attention, memory, decision, and learning subsystems.
提供感知、注意、记忆、决策与学习子系统的整体编排。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Sequence
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import numpy as np
import time
import threading
import json
import os
import logging

from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.core.backends import SimulationBackend, get_backend
from BrainSimulationSystem.core.learning import create_learning_rule
from BrainSimulationSystem.models.persistent_memory import PersistentMemoryManager
from BrainSimulationSystem.models.knowledge_graph import KnowledgeGraph, KnowledgeConstraint, Triple
from BrainSimulationSystem.models.symbolic_reasoner import SymbolicReasoner, Rule
from BrainSimulationSystem.models.hybrid_planner import HybridPlanner
from BrainSimulationSystem.models.meta_reasoner import MetaReasoner
from BrainSimulationSystem.models.attention_manager import AttentionManager
from BrainSimulationSystem.models.attention_controller import GoalDrivenAttentionController
from BrainSimulationSystem.models.memory_retrieval import MemoryRetrievalAdvisor
from BrainSimulationSystem.models.self_model import SelfAwarenessModule
from BrainSimulationSystem.models.motor_control import MotorControlSystem, MotorControlUnavailable
from BrainSimulationSystem.models.limbic_system import LimbicSystem
from BrainSimulationSystem.models.global_workspace import MetacognitiveController
from BrainSimulationSystem.knowledge.source_loader import load_external_sources
from BrainSimulationSystem.learning.self_supervised import SelfSupervisedConfig, SelfSupervisedPredictor
from BrainSimulationSystem.module_registry import ModuleFactory
from BrainSimulationSystem.knowledge.ingestion import KnowledgeIngestionManager
try:
    from BrainSimulationSystem.models.emotion_processing import EmotionSystem
except Exception:  # pragma: no cover - optional dependency fallback
    EmotionSystem = None

try:
    from BrainSimulationSystem.models.emotion_motivation import EmotionMotivationSystem
except Exception:  # pragma: no cover - optional dependency fallback
    EmotionMotivationSystem = None

try:
    from BrainSimulationSystem.motivation.curiosity import SocialCuriosityEngine
except Exception:  # pragma: no cover - optional dependency fallback
    SocialCuriosityEngine = None

try:
    from BrainSimulationSystem.personality.temporal_evolution import PersonalityEvolver
except Exception:  # pragma: no cover - optional dependency fallback
    PersonalityEvolver = None

try:
    from BrainSimulationSystem.personality.dynamics import PersonalityDynamics
except Exception:  # pragma: no cover - optional dependency fallback
    PersonalityDynamics = None

from BrainSimulationSystem.config.default_config import get_config
from BrainSimulationSystem.config.stage_profiles import build_stage_config


class BrainSimulation:
    """Brain simulation system.

    大脑模拟系统。

    Integrates neural networks, synaptic plasticity, learning rules, and cognitive processes behind a unified interface.
    整合神经元网络、突触可塑性、学习规则与认知过程，提供统一接口。
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        profile: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        """Initialise the brain simulation system.

        初始化大脑模拟系统。

        Args:
            config: Configuration dictionary; defaults are used when omitted.
            config: 配置字典；若缺省则采用默认配置。
            profile: Optional profile name equivalent to setting ``profile`` in the config.
            profile: 可选配置档案名称，效果等同于在配置中设置 ``profile``。
            stage: Optional curriculum stage that loads infant/adult brain presets.
            stage: 可选课程阶段标识，用于加载婴儿/成长脑预设。
        """
        # 加载配置（支持按档案或阶段选择规模）
        overrides = dict(config) if isinstance(config, dict) else {}
        requested_profile = profile or overrides.pop("profile", None)
        stage_name = stage or overrides.pop("stage", None)

        if stage_name:
            if requested_profile:
                base_payload = build_stage_config(
                    stage_name, base_profile=requested_profile
                )
            else:
                base_payload = build_stage_config(stage_name)
        else:
            base_payload = get_config(profile=requested_profile or "prototype")

        base_config = self._merge_config(base_payload, overrides)
        self.config = self._apply_defaults(base_config)

        # 初始化仿真后端
        simulation_cfg = self.config.setdefault("simulation", {})
        backend_name = simulation_cfg.get("backend", "native")
        self.backend: SimulationBackend = get_backend(backend_name)
        self.network = self.backend.build_network(self.config)

        # 创建学习规则
        self.learning_rules = []
        self.logger = logging.getLogger(self.__class__.__name__)
        learning_rules = self.config.get("network", {}).get("learning_rules", {})
        for rule_name, rule_params in learning_rules.items():
            if rule_params.get("enabled", False):
                rule = create_learning_rule(rule_name, self.network, rule_params)
                self.learning_rules.append(rule)

        # 创建认知过程
        modules_cfg = self.config.get("modules", {})
        registry_overrides = modules_cfg.get("registry") if isinstance(modules_cfg, dict) else None
        if not isinstance(registry_overrides, dict):
            registry_overrides = None
        self._module_factory = ModuleFactory(
            self.network,
            registry=registry_overrides,
            logger=self.logger,
        )
        self._register_module_builders()
        self._module_container = None
        self._dynamic_module_configs: Dict[str, Dict[str, Any]] = self._load_dynamic_module_configs(modules_cfg)
        self._initialize_core_modules()
        self._language_semantic_attached = False
        self.self_supervised_enabled = False
        self.self_supervised: Optional[SelfSupervisedPredictor] = None
        self.self_supervised_summary: Dict[str, Any] = {}
        self._initialize_self_supervised_predictor()

        # 模拟状态
        self.is_running = False
        self.simulation_thread = None
        self.current_time = 0.0
        self.simulation_results = {
            "times": [],
            "spikes": [],
            "voltages": [],
            "weights": [],
            "cognitive_states": [],
            "predictive_error": [],
            "reconstruction_error": [],
            "self_supervised": [],
        }

        # 事件回调
        self.event_callbacks = {}
        self._register_builtin_event_handlers()
        self.attention_manager_enabled = False
        self.attention_manager: Optional[AttentionManager] = None
        self.attention_focus_state: Dict[str, Any] = {}
        self.self_model_enabled = False
        self.self_model: Optional[SelfAwarenessModule] = None
        self.self_model_state: Dict[str, Any] = {}
        self._initialize_high_level_systems()
        self.attention_controller = GoalDrivenAttentionController(self.config.get("attention_controller", {}))
        self.last_attention_directives: Dict[str, Any] = {}
        self.memory_retrieval_advisor = MemoryRetrievalAdvisor(self.config.get("memory_retrieval", {}))
        self._last_plan_result: Optional[Dict[str, Any]] = None
        self.pending_research_actions: List[str] = []
        self._active_plan_goals: List[str] = []
        self._active_plan_sequence: List[str] = []
        self._active_plan_cursor: int = 0
        self._last_predicted_reward: float = 0.0
        self._pending_experience: Optional[Dict[str, Any]] = None
        self._experience_step: int = 0

    @staticmethod
    def _load_dynamic_module_configs(modules_cfg: Any) -> Dict[str, Dict[str, Any]]:
        """Extract optional dynamic module configs from ``config['modules']``."""

        if not isinstance(modules_cfg, dict):
            return {}

        raw = (
            modules_cfg.get("components")
            or modules_cfg.get("dynamic_components")
            or modules_cfg.get("dynamic")
            or {}
        )
        if not isinstance(raw, dict):
            return {}

        configs: Dict[str, Dict[str, Any]] = {}
        for name, cfg in raw.items():
            if not name:
                continue
            if cfg is None:
                continue
            if isinstance(cfg, str):
                configs[str(name)] = {"class": cfg}
            elif isinstance(cfg, dict):
                configs[str(name)] = dict(cfg)
        return configs

    def _register_module_builders(self) -> None:
        """Register best-effort builders for optional modules."""

        factory = getattr(self, "_module_factory", None)
        if factory is None:
            return
        try:
            factory.register("language_hub", self._build_language_hub_module)
        except Exception:
            return

    def _build_language_hub_module(self, network: Any, config: Dict[str, Any]) -> Any:
        """Instantiate LanguageHub with an optional LLMService helper."""

        cfg = dict(config or {})
        llm_service_cfg = cfg.pop("llm_service", None)
        llm_service = None
        if isinstance(llm_service_cfg, dict):
            enabled = bool(llm_service_cfg.pop("enabled", True))
            if enabled:
                try:
                    from BrainSimulationSystem.integration.llm_service import LLMService

                    llm_service_cfg.setdefault("provider", "internal_pipeline")
                    llm_service_cfg.setdefault("logger", getattr(self, "logger", None))
                    llm_service = LLMService(**llm_service_cfg)
                except Exception as exc:  # pragma: no cover - optional integration
                    if getattr(self, "logger", None) is not None:
                        self.logger.debug("Failed to initialize LLMService: %s", exc)
                    llm_service = None

        from BrainSimulationSystem.models.language_hub import LanguageHub

        return LanguageHub(network, cfg, llm_service=llm_service)

    def _get_language_module(self) -> Optional[Any]:
        for key in ("language", "language_hub", "language_module"):
            module = self.dynamic_modules.get(key)
            if module is not None:
                return module
        return None

    @staticmethod
    def _resolve_language_input(inputs: Dict[str, Any]) -> Optional[str]:
        """Best-effort extraction of a language utterance from heterogeneous inputs."""

        direct = inputs.get("language_input")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

        for key in ("text", "utterance", "user_text", "prompt"):
            candidate = inputs.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        perception = inputs.get("perception")
        if isinstance(perception, dict):
            candidate = perception.get("text")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        return None

    def _maybe_attach_language_semantics(self, language_module: Any) -> None:
        """Attach LanguageHub SemanticNetwork into the memory system (best effort)."""

        if self._language_semantic_attached:
            return

        semantic_network = getattr(language_module, "semantic_network", None)
        if semantic_network is None:
            return

        memory_process = getattr(self, "memory", None)
        memory_system = getattr(memory_process, "memory_system", None) if memory_process is not None else None
        attach = getattr(memory_system, "attach_semantic_network", None) if memory_system is not None else None
        if callable(attach):
            try:
                attach(semantic_network)
                self._language_semantic_attached = True
            except Exception:
                return

    def _ingest_language_relations(self, comprehension: Dict[str, Any], *, timestamp: float) -> int:
        """Mirror language semantic relations into the global knowledge graph."""

        if not isinstance(comprehension, dict):
            return 0
        relations = comprehension.get("relations") or []
        if not isinstance(relations, list) or not relations:
            return 0
        if self.knowledge_graph is None:
            return 0

        triples: List[Triple] = []
        for entry in relations[:64]:
            if not isinstance(entry, dict):
                continue
            head = str(entry.get("head", "")).strip()
            dependent = str(entry.get("dependent", "")).strip()
            relation = str(entry.get("relation", "")).strip()
            if not head or not dependent or not relation:
                continue
            if head.upper() == "ROOT":
                continue
            triples.append((head, relation, dependent))

        if not triples:
            return 0

        try:
            return self.knowledge_graph.upsert_triples(
                triples,
                default_metadata={"source": "language", "timestamp": float(timestamp)},
            )
        except Exception:
            return 0

    def _build_language_memory_store(
        self,
        comprehension: Dict[str, Any],
        *,
        base_context: Dict[str, Any],
        timestamp: float,
        speaker: Optional[str] = None,
    ) -> Dict[str, Any]:
        key_terms = comprehension.get("key_terms") or []
        if isinstance(key_terms, str):
            key_terms = [key_terms]
        if not isinstance(key_terms, list):
            key_terms = []

        entities = comprehension.get("entities") or []
        if isinstance(entities, str):
            entities = [entities]
        if not isinstance(entities, list):
            entities = []

        intent = comprehension.get("intent")
        concept = None
        for candidate in list(key_terms) + list(entities):
            text = str(candidate).strip().lower()
            if text:
                concept = text
                break
        if concept is None:
            concept = str(intent or "utterance").strip().lower() or "utterance"

        content: Dict[str, Any] = {
            "concept": concept,
            "event": "language_turn",
            "text": comprehension.get("input"),
            "summary": comprehension.get("summary"),
            "intent": intent,
            "key_terms": list(key_terms)[:12],
            "entities": list(entities)[:12],
            "relations": comprehension.get("relations"),
            "speaker": speaker,
            "timestamp": float(timestamp),
        }

        store_context = dict(base_context or {})
        store_context.setdefault("source", "language")
        store_context.setdefault("timestamp", float(timestamp))
        if speaker:
            store_context.setdefault("speaker", speaker)

        return {
            "memory_type": "EPISODIC",
            "content": content,
            "context": store_context,
            "emotional_tags": self._extract_emotional_tags(base_context),
        }

    def _experience_config(self) -> Dict[str, Any]:
        memory_cfg = self.config.get("memory", {})
        if not isinstance(memory_cfg, dict):
            return {}
        cfg = memory_cfg.get("experience", {})
        if isinstance(cfg, dict):
            return cfg
        if cfg is True:
            return {"enabled": True}
        return {}

    def _experience_enabled(self) -> bool:
        cfg = self._experience_config()
        enabled = cfg.get("enabled")
        return bool(enabled) if enabled is not None else False

    def _build_experience_memory_store(
        self,
        pending: Dict[str, Any],
        *,
        base_context: Dict[str, Any],
        timestamp: float,
        reward: Any,
        inputs: Dict[str, Any],
        next_perception_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        action = pending.get("decision", pending.get("action"))
        confidence = pending.get("confidence")
        predicted_reward = pending.get("predicted_reward")
        start_time = pending.get("timestamp")
        start_stage = pending.get("stage")

        outcome_keys = (
            "reward",
            "success",
            "done",
            "terminated",
            "truncated",
            "error",
            "status",
        )
        outcome_payload = {key: inputs.get(key) for key in outcome_keys if key in inputs}
        if reward is not None and "reward" not in outcome_payload:
            outcome_payload["reward"] = reward

        content: Dict[str, Any] = {
            "event": "experience",
            "action": action,
            "confidence": confidence,
            "predicted_reward": predicted_reward,
            "reward": reward,
            "outcome": self._sanitize_payload(outcome_payload) if outcome_payload else {},
            "perception": pending.get("perception_context") or {},
            "next_perception": next_perception_context or {},
            "start_time": float(start_time) if isinstance(start_time, (int, float)) else None,
            "timestamp": float(timestamp),
            "stage": (self.config.get("metadata", {}) or {}).get("stage"),
            "stage_at_action": start_stage,
        }

        concept = pending.get("concept")
        if concept:
            content["concept"] = concept
        elif action is not None:
            content["concept"] = str(action).strip().lower() or "experience"
        else:
            content["concept"] = "experience"

        store_context: Dict[str, Any] = {
            "source": "experience",
            "timestamp": float(timestamp),
        }
        goals = inputs.get("goals")
        if goals:
            store_context["goals"] = self._sanitize_payload(goals)

        return {
            "memory_type": str(self._experience_config().get("memory_type", "EPISODIC")),
            "content": content,
            "context": store_context,
            "emotional_tags": self._extract_emotional_tags(base_context),
        }

    def _extract_experience_triples(self, content: Dict[str, Any], *, max_triples: int = 32) -> List[Triple]:
        if max_triples <= 0:
            return []

        triples: List[Triple] = []

        raw_triples = content.get("triples")
        if isinstance(raw_triples, list):
            for entry in raw_triples:
                if len(triples) >= max_triples:
                    break
                if isinstance(entry, (list, tuple)) and len(entry) == 3:
                    triples.append((str(entry[0]), str(entry[1]), str(entry[2])))

        agent = str(content.get("agent") or "agent")
        action = content.get("action")
        concept = content.get("concept")
        stage = content.get("stage")

        if action is not None and len(triples) < max_triples:
            triples.append((agent, "performed", str(action)))
        if concept and action is not None and len(triples) < max_triples:
            triples.append((str(action), "about", str(concept)))
        if stage and len(triples) < max_triples:
            triples.append((agent, "in_stage", str(stage)))

        return triples

    @staticmethod
    def _extract_emotional_tags(context: Any) -> Dict[str, float]:
        """Extract a minimal (valence, arousal) tag payload from a context snapshot."""

        if not isinstance(context, dict):
            return {}
        emotion_state = context.get("emotion_state")
        if not isinstance(emotion_state, dict):
            return {}

        tags: Dict[str, float] = {}
        for key in ("valence", "arousal"):
            value = emotion_state.get(key)
            if isinstance(value, (int, float)):
                tags[key] = float(value)
        return tags

    def _initialize_core_modules(self) -> None:
        """Instantiate core cognitive subsystems using the module factory."""

        component_configs = {
            "perception": self.config.get("perception", {}),
            "attention": self.config.get("attention", {}),
            "memory": self.config.get("memory", {}),
            "decision": self.config.get("decision", {}),
        }
        component_configs.update(self._dynamic_module_configs)
        container = self._module_factory.build_many(component_configs)
        self._module_container = container
        self.perception = container["perception"]
        self.attention = container["attention"]
        self.memory = container["memory"]
        self.decision = container["decision"]
        self.dynamic_modules = {
            name: module
            for name, module in container.components.items()
            if name not in {"perception", "attention", "memory", "decision"}
        }

    def _rebuild_modules(self, targets: Optional[set[str]] = None) -> None:
        """Rebuild modules for the requested targets using the current registry."""

        component_configs = {
            "perception": self.config.get("perception", {}),
            "attention": self.config.get("attention", {}),
            "memory": self.config.get("memory", {}),
            "decision": self.config.get("decision", {}),
        }
        component_configs.update(self._dynamic_module_configs)

        if self._module_container is None or not targets:
            container = self._module_factory.build_many(component_configs)
            self._module_container = container
        else:
            container = self._module_container
            for name, cfg in component_configs.items():
                if targets is None or name in targets:
                    container.components[name] = self._module_factory.build(name, cfg)

        self.perception = container.components.get("perception", getattr(self, "perception", None))
        self.attention = container.components.get("attention", getattr(self, "attention", None))
        self.memory = container.components.get("memory", getattr(self, "memory", None))
        self.decision = container.components.get("decision", getattr(self, "decision", None))
        self.dynamic_modules = {
            name: module
            for name, module in container.components.items()
            if name not in {"perception", "attention", "memory", "decision"}
        }

    def _ingest_structured_perception(self, structured_payload: Optional[Dict[str, Any]]) -> None:
        """Bridge structured perception results into the knowledge graph."""

        if not structured_payload:
            return
        triples = structured_payload.get("triples")
        if not triples or self.knowledge_graph is None:
            return

        normalized_triples: List[Triple] = []
        for triple in triples:
            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                normalized_triples.append(
                    (str(triple[0]), str(triple[1]), str(triple[2]))
                )
        if not normalized_triples:
            return

        metadata = structured_payload.get("metadata")
        default_metadata: Dict[str, Any] = {}
        if isinstance(metadata, dict):
            for key in ("rows", "columns", "timestamp"):
                if metadata.get(key) is not None:
                    default_metadata[key] = metadata[key]
            if metadata.get("source"):
                default_metadata.setdefault("provenance", metadata["source"])
        provenance = structured_payload.get("provenance")
        if provenance:
            default_metadata.setdefault("provenance", provenance)
        embeddings = structured_payload.get("embeddings")
        if isinstance(embeddings, dict):
            summary = embeddings.get("summary") or structured_payload.get("summary_embedding")
            if summary is not None:
                default_metadata["embedding"] = summary

        inserted = self.knowledge_graph.upsert_triples(
            normalized_triples,
            default_metadata=default_metadata or None,
        )
        structured_payload["ingested_triples"] = inserted
        self.last_structured_ingest = {
            "rows": (metadata or {}).get("rows") if isinstance(metadata, dict) else None,
            "columns": (metadata or {}).get("columns") if isinstance(metadata, dict) else None,
            "total_triples": len(normalized_triples),
            "ingested": inserted,
            "timestamp": self.current_time,
        }

    def _maybe_queue_curiosity_topic(
        self,
        curiosity_result: Dict[str, Any],
        attention_directives: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> None:
        manager = getattr(self, "knowledge_ingestion", None)
        if manager is None:
            return
        drive = curiosity_result.get("drive")
        if drive is None or drive < getattr(self, "curiosity_ingest_threshold", 0.65):
            return

        topic = None
        stimulus = curiosity_result.get("stimulus", {})
        candidate = None
        if isinstance(stimulus, dict):
            candidate = (
                stimulus.get("topic")
                or stimulus.get("focus")
                or stimulus.get("novelty_topic")
            )
        if not candidate:
            snapshot = attention_directives.get("goal_snapshot") if attention_directives else None
            if isinstance(snapshot, list) and snapshot:
                candidate = snapshot[0]
            elif isinstance(snapshot, str):
                candidate = snapshot
        if not candidate:
            goals = inputs.get("goals")
            if isinstance(goals, list) and goals:
                candidate = goals[0]
            elif isinstance(goals, str):
                candidate = goals
        if not candidate:
            return
        metadata = {
            "reason": "curiosity",
            "summary": f"Auto exploration of {candidate}",
            "tags": ["curiosity", "exploration"],
        }
        research_action = f"research_topic:{candidate}"
        if research_action not in self.pending_research_actions:
            self.pending_research_actions.append(research_action)
        manager.request_topic(candidate, metadata=metadata)

    def _compute_attention_directives(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        controller = getattr(self, "attention_controller", None)
        if controller is None:
            return {}
        language_context = inputs.get("language_context")
        dialogue_state = None
        if isinstance(language_context, dict):
            dialogue_state = language_context.get("dialogue_state")
        working_memory_state = inputs.get("memory_context")
        motivation = inputs.get("motivation")
        return controller.compute(
            goals=inputs.get("goals"),
            planner=self._last_plan_result,
            dialogue_state=dialogue_state,
            working_memory=working_memory_state if isinstance(working_memory_state, dict) else None,
            motivation=motivation if isinstance(motivation, dict) else None,
        )

    def _build_auto_memory_query(self, inputs: Dict[str, Any], directives: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        advisor = getattr(self, "memory_retrieval_advisor", None)
        if advisor is None:
            return None
        language_context = inputs.get("language_context")
        dialogue_state = None
        summary = None
        if isinstance(language_context, dict):
            dialogue_state = language_context.get("dialogue_state")
            summary = language_context.get("last_summary") or language_context.get("summary")
        if not dialogue_state and directives:
            dialogue_state = {"topics": directives.get("semantic_focus")}
        return advisor.build_query(
            goals=inputs.get("goals"),
            planner=self._last_plan_result,
            dialogue_state=dialogue_state,
            summary=summary,
        )

    def get_capabilities_report(self) -> Dict[str, Any]:
        """Return the backend network's capabilities report if available."""

        reporter = getattr(self.network, "get_capabilities_report", None)
        if callable(reporter):
            return reporter()
        return {}

    def report_status(self) -> Dict[str, Any]:
        """Return a lightweight introspection snapshot for debugging/UI use."""

        last_cognitive = {}
        try:
            states = self.simulation_results.get("cognitive_states", [])
            if states:
                last = states[-1]
                if isinstance(last, dict):
                    last_cognitive = last
        except Exception:
            last_cognitive = {}

        decision = last_cognitive.get("decision") if isinstance(last_cognitive.get("decision"), dict) else {}
        emotion = last_cognitive.get("emotion") if isinstance(last_cognitive.get("emotion"), dict) else {}
        self_model = last_cognitive.get("self_model") if isinstance(last_cognitive.get("self_model"), dict) else {}
        metacognition = (
            last_cognitive.get("metacognition") if isinstance(last_cognitive.get("metacognition"), dict) else {}
        )

        if not metacognition and isinstance(getattr(self, "metacognition_state", None), dict):
            metacognition = dict(self.metacognition_state)

        emotion_summary: Dict[str, Any] = {}
        for key in ("valence", "arousal", "dominant_emotion"):
            if key in emotion:
                emotion_summary[key] = emotion.get(key)
        if isinstance(emotion.get("neuromodulators"), dict):
            emotion_summary["neuromodulators"] = dict(emotion.get("neuromodulators"))
        if isinstance(emotion.get("limbic_circuits"), dict):
            limbic = emotion.get("limbic_circuits", {})
            amygdala = limbic.get("amygdala") if isinstance(limbic, dict) else None
            if isinstance(amygdala, dict) and "threat_level" in amygdala:
                emotion_summary["threat_level"] = amygdala.get("threat_level")

        meta_summary: Dict[str, Any] = {}
        if isinstance(metacognition.get("commands"), dict):
            commands = metacognition["commands"]
            meta_summary["request_more_information"] = bool(commands.get("request_more_information"))
            meta_summary["learning_rate_scale"] = commands.get("learning_rate_scale")
            meta_summary["notes"] = commands.get("notes")
        workspace = metacognition.get("workspace")
        if isinstance(workspace, dict):
            meta_summary["workspace_dominant"] = workspace.get("dominant")
            meta_summary["workspace_sequence"] = workspace.get("sequence")

        beliefs = self_model.get("beliefs") if isinstance(self_model.get("beliefs"), dict) else {}
        return {
            "time": float(self.current_time),
            "running": bool(self.is_running),
            "backend": getattr(self.backend, "name", None),
            "decision": {
                "decision": decision.get("decision"),
                "confidence": decision.get("confidence"),
                "predicted_reward": decision.get("predicted_reward"),
            },
            "emotion": emotion_summary,
            "self_model": {
                "current_goal": beliefs.get("current_goal"),
                "confidence": beliefs.get("confidence"),
                "alerts": list(self_model.get("alerts", [])) if isinstance(self_model.get("alerts"), list) else [],
            },
            "metacognition": meta_summary,
            "attention_directives": dict(self.last_attention_directives) if isinstance(self.last_attention_directives, dict) else {},
            "pending_research_actions": list(self.pending_research_actions)[:16],
        }

    @staticmethod
    def _merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge overrides into base."""

        merged = dict(base)
        for key, value in overrides.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = BrainSimulation._merge_config(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the configuration exposes the keys expected by the runtime."""

        cfg = dict(config)
        network_cfg = cfg.setdefault("network", {})
        network_cfg.setdefault("learning_rules", {})
        cfg.setdefault("perception", {})
        cfg.setdefault("attention", {})
        cfg.setdefault("memory", {})
        cfg.setdefault("decision", {})
        cfg.setdefault("emotion", {})
        cfg.setdefault("curiosity", {})
        cfg.setdefault("personality", {})
        cfg.setdefault("self_supervised", {})
        cfg.setdefault("modules", {})
        cfg.setdefault("meta_learning", {})
        cfg.setdefault("attention_manager", {})
        cfg.setdefault("self_model", {})
        cfg.setdefault("metacognition", {})
        cfg.setdefault("hebbian", {})
        cfg.setdefault("motor", {})
        cfg.setdefault("simulation", {})
        return cfg

    def _initialize_self_supervised_predictor(self) -> None:
        """Instantiate the self-supervised predictor if enabled in config."""
        ss_cfg = self.config.get("self_supervised", {})
        if not isinstance(ss_cfg, dict):
            ss_cfg = {}

        enabled = bool(ss_cfg.get("enabled", ss_cfg != {}))
        self.self_supervised_enabled = enabled
        self.self_supervised_summary = {}
        if not enabled:
            self.self_supervised = None
            return

        predictor_cfg = ss_cfg.get("predictor")
        try:
            if isinstance(predictor_cfg, SelfSupervisedConfig):
                config_obj = predictor_cfg
            else:
                predictor_params: Dict[str, Any] = {}
                sources = [ss_cfg]
                if isinstance(predictor_cfg, dict):
                    sources.append(predictor_cfg)
                for source in sources:
                    for key, value in source.items():
                        if key in SelfSupervisedConfig.__dataclass_fields__:
                            predictor_params[key] = value
                config_obj = SelfSupervisedConfig(**predictor_params)
            self.self_supervised = SelfSupervisedPredictor(config_obj)
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Failed to initialize self-supervised predictor: %s", exc)
            self.self_supervised_enabled = False
            self.self_supervised = None

    def _initialize_high_level_systems(self) -> None:
        """Instantiate affective, motivational and personality subsystems."""
        emotion_cfg = self.config.get("emotion", {})
        try:
            self.emotion_system = EmotionSystem(emotion_cfg)
        except Exception as exc:  # pragma: no cover - runtime fallback
            self.logger.warning("Failed to initialize EmotionSystem: %s", exc)
            self.emotion_system = None

        self.emotion_state = {}
        if self.emotion_system is not None:
            try:
                self.emotion_state = self.emotion_system.current_state()
            except Exception as exc:  # pragma: no cover - optional path
                self.logger.debug("EmotionSystem current_state failed: %s", exc)
                self.emotion_state = {}

        limbic_cfg = self.config.get("limbic", {})
        try:
            self.limbic_system: Optional[LimbicSystem] = LimbicSystem(limbic_cfg if isinstance(limbic_cfg, dict) else {})
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Failed to initialize LimbicSystem: %s", exc)
            self.limbic_system = None

        curiosity_cfg = self.config.get("curiosity", {})
        try:
            self.curiosity_engine = SocialCuriosityEngine()
            overrides = curiosity_cfg.get("overrides")
            if isinstance(overrides, dict):
                base_curiosity = overrides.get("base_curiosity")
                if isinstance(base_curiosity, dict):
                    target = getattr(self.curiosity_engine, "base_curiosity", None)
                    if isinstance(target, dict):
                        target.update(base_curiosity)
                social_components = overrides.get("social_components")
                if isinstance(social_components, dict):
                    target = getattr(self.curiosity_engine, "social_components", None)
                    if isinstance(target, dict):
                        for key, value in social_components.items():
                            if isinstance(value, dict):
                                target.setdefault(key, {}).update(value)
                neuro_modulation = overrides.get("neuro_modulation")
                if isinstance(neuro_modulation, dict):
                    target = getattr(self.curiosity_engine, "neuro_modulation", None)
                    if isinstance(target, dict):
                        target.update(neuro_modulation)
        except Exception as exc:  # pragma: no cover - runtime fallback
            self.logger.warning("Failed to initialize SocialCuriosityEngine: %s", exc)
            self.curiosity_engine = None

        self.last_curiosity_drive = 0.0

        try:
            self.personality_evolver = PersonalityEvolver()
        except Exception as exc:  # pragma: no cover - runtime fallback
            self.logger.warning("Failed to initialize PersonalityEvolver: %s", exc)
            self.personality_evolver = None

        try:
            self.personality_dynamics = PersonalityDynamics()
        except Exception as exc:  # pragma: no cover - runtime fallback
            self.logger.warning("Failed to initialize PersonalityDynamics: %s", exc)
            self.personality_dynamics = None

        baseline_traits = {}
        if self.personality_evolver is not None:
            base_traits = getattr(self.personality_evolver, "base_traits", {})
            if isinstance(base_traits, dict):
                baseline_traits = dict(base_traits)

        dynamic_traits = {}
        if self.personality_dynamics is not None:
            base_traits = getattr(self.personality_dynamics, "base_traits", {})
            if isinstance(base_traits, dict):
                dynamic_traits = dict(base_traits)

        self.personality_state = {
            "baseline": baseline_traits,
            "dynamic": dynamic_traits,
            "environment": {},
        }

        knowledge_cfg = self.config.get("knowledge", {})
        initial_triples = knowledge_cfg.get("triples", [])
        self.knowledge_graph = KnowledgeGraph()
        normalized_triples = []
        for triple in initial_triples:
            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                normalized_triples.append(tuple(str(value) for value in triple))
        self.knowledge_graph.add_many(normalized_triples)
        self.last_structured_ingest: Optional[Dict[str, Any]] = None

        external_sources = knowledge_cfg.get("sources", [])
        if external_sources:
            try:
                added = load_external_sources(self.knowledge_graph, external_sources, logger=self.logger)
                if added:
                    self.logger.info("Loaded %d triples from external knowledge sources.", added)
            except Exception as exc:
                self.logger.warning("Failed to ingest external knowledge sources: %s", exc)

        self.knowledge_constraints: List[KnowledgeConstraint] = []
        for constraint_cfg in knowledge_cfg.get("constraints", []):
            try:
                constraint = KnowledgeConstraint(
                    description=str(constraint_cfg.get("description", "constraint")),
                    required=[tuple(pred) for pred in constraint_cfg.get("required", [])],
                    forbidden=[tuple(pred) for pred in constraint_cfg.get("forbidden", [])],
                )
                self.knowledge_constraints.append(constraint)
            except Exception as exc:
                self.logger.debug("Failed to add constraint %s: %s", constraint_cfg, exc)

        self.symbolic_reasoner = SymbolicReasoner(self.knowledge_graph)
        for rule_cfg in knowledge_cfg.get("rules", []):
            try:
                rule = Rule(
                    name=str(rule_cfg.get("name", "rule")),
                    antecedents=[tuple(pred) for pred in rule_cfg.get("antecedents", [])],
                    consequent=tuple(rule_cfg.get("consequent", ("", "", ""))),
                )
                self.symbolic_reasoner.add_rule(rule)
            except Exception as exc:
                self.logger.debug("Failed to add rule %s: %s", rule_cfg, exc)

        ingestion_cfg = self.config.get("knowledge_ingestion", {})
        self.knowledge_ingestion = KnowledgeIngestionManager(ingestion_cfg, logger=self.logger)
        self.curiosity_ingest_threshold = float(ingestion_cfg.get("curiosity_threshold", 0.65))
        self.last_knowledge_ingestion: Dict[str, Any] = {}

        planner_cfg = self.config.get("planner", {})
        self.planner_enabled = bool(planner_cfg.get("enabled", False))
        self.planner: Optional[HybridPlanner]
        if self.planner_enabled:
            try:
                controller_cfg = planner_cfg.get("controller")
                if controller_cfg is None:
                    controller_cfg = {k: v for k, v in planner_cfg.items() if k != "enabled"}
                self.planner = HybridPlanner(self.knowledge_graph, self.symbolic_reasoner, controller_cfg)
            except Exception as exc:
                self.logger.warning("Planner initialization failed: %s", exc)
                self.planner = None
                self.planner_enabled = False
        else:
            self.planner = None

        meta_cfg = self.config.get("meta_reasoning", {})
        self.meta_reasoner_enabled = bool(meta_cfg.get("enabled", False))
        self.meta_reasoner: Optional[MetaReasoner]
        if self.meta_reasoner_enabled:
            try:
                controller_cfg = meta_cfg.get("controller") or {k: v for k, v in meta_cfg.items() if k != "enabled"}
                self.meta_reasoner = MetaReasoner(controller_cfg)
            except Exception as exc:
                self.logger.warning("Meta reasoner initialization failed: %s", exc)
                self.meta_reasoner = None
                self.meta_reasoner_enabled = False
        else:
            self.meta_reasoner = None

        memory_cfg = self.config.get("memory", {})
        if memory_cfg.get("enabled", True):
            persistence_cfg = memory_cfg.get("persistence", {})
            try:
                self.persistent_memory = PersistentMemoryManager(persistence_cfg)
            except Exception as exc:
                self.logger.warning("Persistent memory initialization failed: %s", exc)
                self.persistent_memory = None
        else:
            self.persistent_memory = None

        attention_manager_cfg = self.config.get("attention_manager", {})
        self.attention_manager_enabled = bool(attention_manager_cfg.get("enabled", False))
        if self.attention_manager_enabled:
            try:
                controller_cfg = attention_manager_cfg.get("controller")
                if controller_cfg is None:
                    controller_cfg = {k: v for k, v in attention_manager_cfg.items() if k != "enabled"}
                self.attention_manager = AttentionManager(controller_cfg)
            except Exception as exc:
                self.logger.warning("Attention manager initialization failed: %s", exc)
                self.attention_manager = None
                self.attention_manager_enabled = False
        else:
            self.attention_manager = None
        self.attention_focus_state = {}

        self_model_cfg = self.config.get("self_model", {})
        self.self_model_enabled = bool(self_model_cfg.get("enabled", False))
        if self.self_model_enabled:
            try:
                module_cfg = self_model_cfg.get("module")
                if module_cfg is None:
                    module_cfg = {k: v for k, v in self_model_cfg.items() if k != "enabled"}
                self.self_model = SelfAwarenessModule(module_cfg)
            except Exception as exc:
                self.logger.warning("Self-awareness module initialization failed: %s", exc)
                self.self_model = None
                self.self_model_enabled = False
        else:
            self.self_model = None
        self.self_model_state = {}

        metacog_cfg = self.config.get("metacognition", {})
        self.metacognition_enabled = bool(metacog_cfg.get("enabled", False))
        if self.metacognition_enabled:
            try:
                controller_cfg = metacog_cfg.get("controller")
                if controller_cfg is None:
                    controller_cfg = {k: v for k, v in metacog_cfg.items() if k != "enabled"}
                self.metacognition: Optional[MetacognitiveController] = MetacognitiveController(
                    controller_cfg if isinstance(controller_cfg, dict) else {}
                )
            except Exception as exc:
                self.logger.warning("Metacognition initialization failed: %s", exc)
                self.metacognition = None
                self.metacognition_enabled = False
        else:
            self.metacognition = None
        self.metacognition_state: Dict[str, Any] = {}

        # Wire metacognition into decision module (best effort) so it can adjust
        # exploration/learning parameters via the existing meta hooks.
        if self.metacognition is not None:
            setter = getattr(self.decision, "set_meta_adjustment_provider", None)
            if callable(setter):
                try:
                    setter(self.metacognition)
                except Exception:
                    pass

        self._initialize_hebbian_learning()

        motivation_cfg = self.config.get("motivation", {})
        self.motivation_enabled = bool(motivation_cfg.get("enabled", False))
        if self.motivation_enabled:
            try:
                controller_cfg = motivation_cfg.get("controller") or {k: v for k, v in motivation_cfg.items() if k != "enabled"}
                self.emotion_motivation = EmotionMotivationSystem(controller_cfg)
            except Exception as exc:
                self.logger.warning("Emotion motivation initialization failed: %s", exc)
                self.emotion_motivation = None
                self.motivation_enabled = False
        else:
            self.emotion_motivation = None

        motor_cfg = self.config.get("motor", {})
        self.motor_enabled = bool(motor_cfg.get("enabled", False))
        self.motor_system: Optional[MotorControlSystem]
        self.motor_status: Dict[str, Any] = {}
        self.motor_last_command: Optional[Any] = None

        if self.motor_enabled:
            try:
                controller_cfg = motor_cfg.get("controller")
                if controller_cfg is None:
                    controller_cfg = {k: v for k, v in motor_cfg.items() if k != "enabled"}
                self.motor_system = MotorControlSystem(controller_cfg)
                self.motor_status["backend"] = self.motor_system.backend
            except MotorControlUnavailable as exc:
                self.logger.warning("Failed to initialize MotorControlSystem: %s", exc)
                self.motor_status["error"] = str(exc)
                self.motor_enabled = False
                self.motor_system = None
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.exception("Motor system initialization failed: %s", exc)
                self.motor_status["error"] = str(exc)
                self.motor_enabled = False
                self.motor_system = None
        else:
            self.motor_system = None
            self.motor_status = {}

    def _initialize_hebbian_learning(self) -> None:
        """Configure Hebbian learning parameters and bookkeeping."""

        hebbian_cfg = self.config.get("hebbian", {})
        if not isinstance(hebbian_cfg, dict):
            hebbian_cfg = {}

        self.hebbian_enabled = bool(hebbian_cfg.get("enabled", False))
        self.hebbian_params = {
            "potentiation": float(hebbian_cfg.get("potentiation", 0.005)),
            "decay": float(hebbian_cfg.get("decay", 0.001)),
            "max_weight": float(hebbian_cfg.get("max_weight", 5.0)),
            "min_weight": float(hebbian_cfg.get("min_weight", -5.0)),
        }
        self.hebbian_metrics: Dict[str, Any] = {}

    def reset(self) -> None:
        """重置模拟状态"""
        # 重置网络
        self.network.reset()
        
        # 重置模拟状态
        self.current_time = 0.0
        self.simulation_results = {
            "times": [],
            "spikes": [],
            "voltages": [],
            "weights": [],
            "cognitive_states": [],
            "predictive_error": [],
            "reconstruction_error": [],
            "self_supervised": [],
        }
        self.self_supervised_summary = {}
        if self.self_supervised is not None:
            try:
                self.self_supervised.reset_state()
            except Exception:  # pragma: no cover - defensive fallback
                pass
        self._pending_experience = None
        self._experience_step = 0
        self._initialize_high_level_systems()
    
    def step(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        执行一步模拟

        Args:
            inputs: 输入数据字典
            dt: 时间步长

        Returns:
            包含模拟结果的字典
        """
        # 更新时间
        self.current_time += dt

        # 更新情绪状态
        emotion_inputs = inputs.get("emotion", {})
        if not isinstance(emotion_inputs, dict):
            emotion_inputs = {}

        reward_value = inputs.get("reward")
        limbic_threat_signal = None
        curiosity_for_threat = inputs.get("curiosity")
        if isinstance(curiosity_for_threat, dict):
            limbic_threat_signal = curiosity_for_threat.get("threat")
            if limbic_threat_signal is None:
                limbic_threat_signal = curiosity_for_threat.get("danger")

        reward_prediction_error: Optional[float] = None
        emotion_result: Dict[str, Any] = {}
        extra_stimulus = inputs.get("emotion_stimulus")
        if self.emotion_system is not None:
            stimuli_payloads: List[Dict[str, Any]] = []
            raw_stimuli = emotion_inputs.get("stimuli")
            if isinstance(raw_stimuli, list):
                stimuli_payloads.extend([stim for stim in raw_stimuli if isinstance(stim, dict)])
            elif emotion_inputs:
                stimuli_payloads.append(emotion_inputs)

            extra_stimulus = inputs.get("emotion_stimulus")
            if isinstance(extra_stimulus, dict):
                stimuli_payloads.append(extra_stimulus)

            explicit_types: set[str] = set()
            for stimulus in stimuli_payloads:
                stim_type = stimulus.get("type") or stimulus.get("stimulus_type")
                if not stim_type:
                    continue
                explicit_types.add(str(stim_type).lower())
                intensity = stimulus.get("intensity", stimulus.get("value", 0.0))
                try:
                    self.emotion_system.receive_stimulus(stim_type, float(intensity))
                except Exception as exc:  # pragma: no cover - optional path
                    self.logger.debug("Emotion stimulus processing failed: %s", exc)

            # Derived limbic stimuli: threat drive and reward prediction error (RPE).
            if limbic_threat_signal is not None and "threat" not in explicit_types and "danger" not in explicit_types:
                try:
                    threat_value = float(np.clip(float(limbic_threat_signal), 0.0, 1.0))
                except Exception:
                    threat_value = 0.0
                if threat_value > 0.0:
                    try:
                        self.emotion_system.receive_stimulus("threat", threat_value)
                    except Exception as exc:  # pragma: no cover - optional path
                        self.logger.debug("Derived threat stimulus failed: %s", exc)

            if reward_value is not None and "reward" not in explicit_types and "loss" not in explicit_types:
                try:
                    reward_float = float(reward_value)
                except Exception:
                    reward_float = None
                if reward_float is not None:
                    try:
                        rpe = reward_float - float(getattr(self, "_last_predicted_reward", 0.0))
                    except Exception:
                        rpe = reward_float
                    reward_prediction_error = float(rpe)
                    magnitude = float(np.clip(abs(rpe), 0.0, 1.0))
                    stim_kind = "reward" if rpe >= 0 else "loss"
                    if magnitude > 0.0:
                        try:
                            self.emotion_system.receive_stimulus(stim_kind, magnitude)
                        except Exception as exc:  # pragma: no cover - optional path
                            self.logger.debug("Derived reward stimulus failed: %s", exc)

            regulation = emotion_inputs.get("regulation") or emotion_inputs.get("regulation_type")
            if regulation:
                try:
                    self.emotion_system.regulate_emotions(regulation)
                except Exception as exc:  # pragma: no cover - optional path
                    self.logger.debug("Emotion regulation failed: %s", exc)

            try:
                self.emotion_system.update_emotions()
            except Exception as exc:  # pragma: no cover - optional path
                self.logger.debug("Emotion update failed: %s", exc)

            try:
                self.emotion_state = self.emotion_system.current_state()
            except Exception as exc:  # pragma: no cover - optional path
                self.logger.debug("Emotion snapshot failed: %s", exc)
                self.emotion_state = {}

            emotion_result = dict(self.emotion_state) if isinstance(self.emotion_state, dict) else {}
            if reward_prediction_error is not None:
                emotion_result.setdefault("reward_prediction_error", reward_prediction_error)
        else:
            emotion_result = {}

        motivation_snapshot = None
        if getattr(self, 'emotion_motivation', None) is not None:
            stimuli_for_motivation = {"goals": inputs.get("goals", [])}
            stimuli_for_motivation.update(emotion_inputs.get("signals", {}))
            if isinstance(extra_stimulus, dict):
                stimuli_for_motivation.update(extra_stimulus)
            if limbic_threat_signal is not None and "threat" not in stimuli_for_motivation:
                stimuli_for_motivation["threat"] = limbic_threat_signal
            empathy_input = inputs.get("empathy_input")
            if not isinstance(empathy_input, dict):
                empathy_input = None
            update = self.emotion_motivation.update(
                reward=inputs.get("reward"),
                stimuli=stimuli_for_motivation,
                empathy_input=empathy_input,
            )
            motivation_snapshot = update
            emotion_result.setdefault("motivation", update.get("motivation"))
            emotion_result.setdefault("limbic", update)

        # 更新好奇心水平
        curiosity_payload = inputs.get("curiosity", {})
        if not isinstance(curiosity_payload, dict):
            curiosity_payload = {}
        curiosity_stimulus = curiosity_payload.get("stimulus", {})
        if not isinstance(curiosity_stimulus, dict):
            extra_stimulus = inputs.get("curiosity_stimulus")
            if isinstance(extra_stimulus, dict):
                curiosity_stimulus = extra_stimulus
            else:
                curiosity_stimulus = {
                    key: curiosity_payload[key]
                    for key in ("novelty", "complexity", "social_cues", "social_context")
                    if key in curiosity_payload
                }

        curiosity_result: Dict[str, Any] = {"stimulus": curiosity_stimulus}
        if self.curiosity_engine is not None:
            try:
                curiosity_drive = self.curiosity_engine.compute_integrated_curiosity(curiosity_stimulus or {})
            except Exception as exc:  # pragma: no cover - optional path
                self.logger.debug("Curiosity computation failed: %s", exc)
                curiosity_drive = self.last_curiosity_drive
            self.last_curiosity_drive = curiosity_drive
            curiosity_result["drive"] = curiosity_drive

            feedback = curiosity_payload.get("feedback", inputs.get("curiosity_feedback"))
            if isinstance(feedback, dict):
                reward = feedback.get("reward")
                social_type = feedback.get("type") or feedback.get("social_type")
                if reward is not None and social_type:
                    try:
                        self.curiosity_engine.update_social_parameters(float(reward), social_type)
                    except Exception as exc:  # pragma: no cover - optional path
                        self.logger.debug("Curiosity feedback failed: %s", exc)
        else:
            curiosity_result["drive"] = self.last_curiosity_drive

        limbic_circuits: Optional[Dict[str, Any]] = None
        if getattr(self, "limbic_system", None) is not None:
            try:
                novelty_value = curiosity_result.get("drive")
                limbic_circuits = self.limbic_system.update(
                    reward=reward_value if reward_value is not None else None,
                    predicted_reward=getattr(self, "_last_predicted_reward", 0.0),
                    threat=limbic_threat_signal,
                    novelty=novelty_value,
                    dt=float(dt),
                )
            except Exception as exc:  # pragma: no cover - optional path
                self.logger.debug("Limbic system update failed: %s", exc)
                limbic_circuits = None

        if isinstance(limbic_circuits, dict) and limbic_circuits:
            emotion_result["limbic_circuits"] = limbic_circuits
            mods = limbic_circuits.get("neuromodulators")
            if isinstance(mods, dict) and mods:
                bucket = emotion_result.get("neuromodulators")
                if isinstance(bucket, dict):
                    for key, value in mods.items():
                        try:
                            bucket[str(key)] = float(value)
                        except Exception:
                            continue
                else:
                    cleaned: Dict[str, float] = {}
                    for key, value in mods.items():
                        try:
                            cleaned[str(key)] = float(value)
                        except Exception:
                            continue
                    if cleaned:
                        emotion_result["neuromodulators"] = cleaned

                if (
                    self.emotion_system is not None
                    and getattr(self, "limbic_system", None) is not None
                    and getattr(self.limbic_system, "apply_to_emotion_system", False)
                ):
                    if mods.get("dopamine") is not None:
                        try:
                            self.emotion_system.dopamine_level = float(mods["dopamine"])
                        except Exception:
                            pass
                    if mods.get("serotonin") is not None:
                        try:
                            self.emotion_system.serotonin_level = float(mods["serotonin"])
                        except Exception:
                            pass
                    if mods.get("norepinephrine") is not None:
                        try:
                            self.emotion_system.norepinephrine_level = float(mods["norepinephrine"])
                        except Exception:
                            pass

        # 更新人格状态
        personality_payload = inputs.get("personality", {})
        env_factors: Dict[str, Any] = {}
        if isinstance(personality_payload, dict):
            env_factors = personality_payload.get("environment", {}) or {}
            if not isinstance(env_factors, dict):
                env_factors = {}

        dt_hours = max(float(dt), 0.0) / 3_600_000.0
        if dt_hours <= 0:
            dt_hours = 1e-6

        baseline_traits = self.personality_state.get("baseline", {})
        if self.personality_evolver is not None:
            try:
                baseline_traits = self.personality_evolver.update(timestep=dt_hours)
            except Exception as exc:  # pragma: no cover - optional path
                self.logger.debug("Personality baseline update failed: %s", exc)

        dynamic_traits = self.personality_state.get("dynamic", {})
        if self.personality_dynamics is not None:
            try:
                dynamic_traits = self.personality_dynamics.update(dt_hours, env_factors)
            except Exception as exc:  # pragma: no cover - optional path
                self.logger.debug("Personality dynamics update failed: %s", exc)

        self.personality_state = {
            "baseline": dict(baseline_traits) if isinstance(baseline_traits, dict) else baseline_traits,
            "dynamic": dict(dynamic_traits) if isinstance(dynamic_traits, dict) else dynamic_traits,
            "environment": env_factors,
        }

        # 处理感知输入
        # ������֪����
        attention_directives: Dict[str, Any] = {}
        if getattr(self, "attention_controller", None) is not None:
            try:
                attention_directives = self._compute_attention_directives(inputs)
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.debug("Attention directive computation failed: %s", exc)
                attention_directives = {}

        external_directives = inputs.get("attention_directives")
        if isinstance(external_directives, dict) and external_directives:
            if attention_directives:
                attention_directives = self._merge_config(attention_directives, dict(external_directives))
            else:
                attention_directives = dict(external_directives)

        # Apply any metacognitive attention overrides scheduled from the prior step.
        metacog = getattr(self, "metacognition", None)
        if getattr(self, "metacognition_enabled", False) and metacog is not None:
            overrides = getattr(metacog, "pending_attention_overrides", None)
            steps = getattr(metacog, "pending_attention_steps", 0)
            if isinstance(overrides, dict) and overrides and isinstance(steps, int) and steps > 0:
                try:
                    attention_directives = self._merge_config(attention_directives or {}, dict(overrides))
                except Exception:  # pragma: no cover - defensive
                    pass
                try:
                    metacog.pending_attention_steps = int(steps) - 1
                    if metacog.pending_attention_steps <= 0:
                        metacog.pending_attention_overrides = {}
                        metacog.pending_attention_steps = 0
                except Exception:  # pragma: no cover - defensive
                    pass
        self.last_attention_directives = attention_directives or {}

        if isinstance(limbic_circuits, dict):
            amygdala_state = limbic_circuits.get("amygdala")
            if isinstance(amygdala_state, dict) and amygdala_state.get("threat_level") is not None:
                try:
                    threat_level = float(amygdala_state.get("threat_level", 0.0))
                except Exception:
                    threat_level = 0.0
                attention_directives.setdefault("threat_level", threat_level)
                weights = attention_directives.setdefault("modality_weights", {})
                if isinstance(weights, dict):
                    try:
                        weights.setdefault("somatosensory", float(np.clip(0.6 + 0.4 * threat_level, 0.0, 1.0)))
                    except Exception:
                        pass

        perception_inputs = dict(inputs)
        if attention_directives:
            perception_inputs["attention_directives"] = attention_directives
        perception_result = self.perception.process(perception_inputs)
        self._ingest_structured_perception(perception_result.get("structured"))
        self._maybe_queue_curiosity_topic(curiosity_result, attention_directives, inputs)

        predictive_summary: Dict[str, Any] = {}
        if self.self_supervised_enabled and self.self_supervised is not None:
            try:
                predictive_summary = self.self_supervised.observe(
                    perception_result,
                    metadata={
                        "timestamp": self.current_time,
                        "reward": inputs.get("reward"),
                    },
                )
                if predictive_summary and self.attention_manager_enabled:
                    self._add_blackboard_item(
                        "self_supervised.predictor",
                        payload={
                            "prediction_error": predictive_summary.get("prediction_error"),
                            "reconstruction_loss": predictive_summary.get("reconstruction_loss"),
                            "latent_norm": predictive_summary.get("latent_norm"),
                            "timestamp": self.current_time,
                        },
                    )
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.debug("Self-supervised predictor failed: %s", exc)
                predictive_summary = {"error": str(exc)}
        self.self_supervised_summary = predictive_summary

        # 处理注意力
        attention_inputs = {
            "perception_output": perception_result.get("perception_output", []),
            "focus_position": inputs.get("focus_position", 0.5),
        }
        if attention_directives:
            attention_inputs["workspace_attention"] = attention_directives.get("workspace_attention")
            attention_inputs["workspace_focus"] = attention_directives.get("workspace_focus")
            attention_inputs["task_goal"] = attention_directives.get("goal_snapshot", inputs.get("goals"))
        attention_result = self.attention.process(attention_inputs)

        # 处理记忆
        memory_context = inputs.get("memory_context")
        if isinstance(memory_context, dict):
            memory_context = dict(memory_context)
        else:
            memory_context = {}

        # Bridge perception summaries into memory context so episodic encoding and
        # downstream language modules can reuse grounded embeddings.
        try:
            perception_context = self._build_perception_embedding_context(
                perception_result=perception_result,
                attention_directives=attention_directives,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.debug("Perception embedding context build failed: %s", exc)
            perception_context = {}
        if perception_context:
            memory_context.setdefault("perception", perception_context)
            if isinstance(attention_directives, dict) and attention_directives:
                memory_context.setdefault("attention_directives", dict(attention_directives))
        if self.self_supervised_summary:
            predictive_context = {
                "prediction_error": self.self_supervised_summary.get("prediction_error"),
                "reconstruction_loss": self.self_supervised_summary.get("reconstruction_loss"),
                "latent_preview": self.self_supervised_summary.get("latent_preview"),
                "predicted_preview": self.self_supervised_summary.get("predicted_preview"),
                "timestamp": self.self_supervised_summary.get("timestamp"),
                "previous_action": self.self_supervised_summary.get("previous_action"),
                "action_confidence": self.self_supervised_summary.get("action_confidence"),
            }
            memory_context.setdefault("self_supervised", predictive_context)
        if emotion_result:
            memory_context.setdefault("emotion_state", emotion_result)

        language_module = self._get_language_module()
        language_text = self._resolve_language_input(inputs)
        pending_language_goal = inputs.get("language_generation")
        requested_language_generation = bool(inputs.get("generate_language_response") or pending_language_goal is not None)
        language_snapshot: Optional[Dict[str, Any]] = None
        language_comprehension: Optional[Dict[str, Any]] = None
        auto_query_inputs: Dict[str, Any] = inputs

        if language_module is not None:
            self._maybe_attach_language_semantics(language_module)

        if language_module is not None and language_text:
            base_language_context = inputs.get("language_context")
            language_context = dict(base_language_context) if isinstance(base_language_context, dict) else {}
            if attention_directives:
                language_context.setdefault("attention_directives", dict(attention_directives))
            if memory_context:
                language_context.setdefault("memory_context", dict(memory_context))
            if isinstance(perception_result, dict) and perception_result:
                language_context.setdefault("perception", perception_result)
            language_context.setdefault("timestamp", float(self.current_time))

            language_inputs = dict(inputs)
            language_inputs["language_input"] = language_text
            language_inputs["language_context"] = language_context
            language_inputs.pop("language_generation", None)
            language_inputs.pop("generate_language_response", None)
            try:
                language_snapshot = language_module.process(language_inputs)
            except Exception as exc:  # pragma: no cover - optional module failure
                language_snapshot = {"error": str(exc)}

            if isinstance(language_snapshot, dict):
                candidate = language_snapshot.get("comprehension")
                if isinstance(candidate, dict):
                    language_comprehension = candidate
                    try:
                        self._ingest_language_relations(candidate, timestamp=self.current_time)
                    except Exception:
                        pass
                    updated_context = dict(language_context)
                    summary = candidate.get("summary")
                    if isinstance(summary, str) and summary.strip():
                        updated_context.setdefault("last_summary", summary.strip())
                    dialogue_state = language_snapshot.get("dialogue_state") or candidate.get("dialogue_state")
                    if isinstance(dialogue_state, dict) and dialogue_state:
                        updated_context["dialogue_state"] = dialogue_state
                    auto_query_inputs = dict(inputs)
                    auto_query_inputs["language_context"] = updated_context

        memory_inputs = {
            "store": inputs.get("memory_store", None),
            "retrieve": inputs.get("memory_retrieve", None),
        }
        if memory_context:
            memory_inputs["context"] = memory_context

        auto_store_payloads: List[Dict[str, Any]] = []

        if self._experience_enabled() and isinstance(getattr(self, "_pending_experience", None), dict):
            pending_experience = self._pending_experience
            try:
                experience_store = self._build_experience_memory_store(
                    pending_experience,
                    base_context=memory_context,
                    timestamp=self.current_time,
                    reward=reward_value,
                    inputs=inputs,
                    next_perception_context=perception_context if isinstance(perception_context, dict) else None,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.debug("Experience store payload build failed: %s", exc)
            else:
                kg_cfg = self._experience_config().get("knowledge_graph", {})
                if (
                    isinstance(kg_cfg, dict)
                    and bool(kg_cfg.get("enabled", False))
                    and self.knowledge_graph is not None
                    and isinstance(experience_store.get("content"), dict)
                ):
                    try:
                        max_triples = int(kg_cfg.get("max_triples", 32))
                    except Exception:
                        max_triples = 32
                    triples = self._extract_experience_triples(
                        experience_store["content"],
                        max_triples=max(0, min(max_triples, 256)),
                    )
                    if triples:
                        try:
                            inserted = self.knowledge_graph.upsert_triples(
                                triples,
                                default_metadata={
                                    "source": "experience",
                                    "timestamp": float(self.current_time),
                                },
                            )
                        except Exception:
                            inserted = 0
                        else:
                            experience_store["content"]["ingested_triples"] = inserted

                if memory_inputs["store"] is None:
                    auto_store_payloads.append(experience_store)
                else:
                    existing = memory_inputs["store"]
                    if isinstance(existing, (list, tuple)):
                        memory_inputs["store"] = list(existing) + [experience_store]
                    else:
                        memory_inputs["store"] = [existing, experience_store]
            finally:
                self._pending_experience = None

        if (
            self.self_supervised_enabled
            and self.self_supervised is not None
            and self.self_supervised.config.auto_store_enabled
        ):
            error_value = self.self_supervised_summary.get("prediction_error")
            if (
                error_value is not None
                and error_value >= self.self_supervised.config.auto_store_error_threshold
                and memory_inputs["store"] is None
            ):
                auto_store_payloads.append({
                    "memory_type": "SENSORY",
                    "content": {
                        "prediction_error": error_value,
                        "prediction": self.self_supervised_summary.get("predicted_preview"),
                        "observation": self.self_supervised_summary.get("reconstruction_preview"),
                        "latent": self.self_supervised_summary.get("latent_preview"),
                    },
                    "context": {
                        "timestamp": self.current_time,
                        "source": "self_supervised",
                    },
                })

        if language_comprehension is not None and memory_inputs["store"] is None:
            try:
                speaker = inputs.get("speaker")
                speaker_value = str(speaker) if isinstance(speaker, str) and speaker.strip() else None
                auto_store_payloads.append(
                    self._build_language_memory_store(
                        language_comprehension,
                        base_context=memory_context,
                        timestamp=self.current_time,
                        speaker=speaker_value,
                    )
                )
            except Exception:
                pass

        if auto_store_payloads and memory_inputs["store"] is None:
            memory_inputs["store"] = auto_store_payloads[0] if len(auto_store_payloads) == 1 else auto_store_payloads

        if memory_inputs.get("retrieve") is None:
            try:
                auto_query = self._build_auto_memory_query(auto_query_inputs, attention_directives)
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.debug("Auto memory query build failed: %s", exc)
                auto_query = None
            if auto_query:
                memory_inputs["retrieve"] = auto_query
        memory_result = self.memory.process(memory_inputs)
        transient_retrievals = list(memory_result.get("retrieved", []))

        persistent_snapshot = None
        if getattr(self, 'persistent_memory', None) is not None:
            encode_items = inputs.get("memory_encode")
            if encode_items is not None:
                if isinstance(encode_items, str):
                    encode_items = [encode_items]
                if isinstance(encode_items, (list, tuple)):
                    for item in encode_items:
                        if isinstance(item, str):
                            self.persistent_memory.add_memory(item, {"time": self.current_time})
                        elif isinstance(item, dict):
                            content = item.get("content")
                            meta = {k: v for k, v in item.items() if k != "content"}
                            if content:
                                self.persistent_memory.add_memory(str(content), meta)

            query = inputs.get("memory_query")
            top_k = int(inputs.get("memory_query_top_k", 5))
            retrieved = []
            if query is not None:
                retrieved = self.persistent_memory.search(str(query), top_k=top_k)
            persistent_snapshot = {
                "working": self.persistent_memory.working_items(),
                "retrieved": retrieved,
            }

        if persistent_snapshot is not None:
            memory_result.setdefault("persistent", persistent_snapshot)

        knowledge_updates = None
        if getattr(self, "knowledge_ingestion", None) is not None:
            try:
                knowledge_updates = self.knowledge_ingestion.tick(
                    self.current_time,
                    self.knowledge_graph,
                    self.knowledge_constraints,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.debug("Knowledge ingestion tick failed: %s", exc)
                knowledge_updates = None
            else:
                self.last_knowledge_ingestion = knowledge_updates or {}
                topics = (knowledge_updates or {}).get("topics") or []
                for topic in topics:
                    action = f"research_topic:{topic}"
                    if action in self.pending_research_actions:
                        self.pending_research_actions.remove(action)

        # 处理决策
        decision_context = dict(inputs.get("decision_context", {}))
        if attention_directives:
            decision_context.setdefault("attention_directives", attention_directives)
        if self.pending_research_actions:
            decision_context.setdefault("knowledge_goals", list(self.pending_research_actions))
        if self.self_supervised_summary:
            decision_context.setdefault("self_supervised", self.self_supervised_summary)
        if emotion_result:
            decision_context.setdefault("emotion_state", emotion_result)
            limbic_circuits = emotion_result.get("limbic_circuits") if isinstance(emotion_result, dict) else None
            if isinstance(limbic_circuits, dict) and limbic_circuits:
                decision_context.setdefault("limbic", limbic_circuits)
        if motivation_snapshot is not None:
            decision_context.setdefault("motivation", motivation_snapshot.get("motivation"))
        if persistent_snapshot is not None:
            decision_context.setdefault("memory_retrieved", persistent_snapshot.get("retrieved", []))
        if transient_retrievals:
            bucket = decision_context.setdefault("memory_retrieved", [])
            if isinstance(bucket, list):
                bucket.extend(transient_retrievals)
            else:
                decision_context["memory_retrieved"] = list(transient_retrievals)
        if language_comprehension is not None:
            decision_context.setdefault("language_intent", language_comprehension.get("intent"))
            decision_context.setdefault("language_summary", language_comprehension.get("summary"))
            decision_context.setdefault("language_key_terms", language_comprehension.get("key_terms"))
            decision_context.setdefault("language_action_items", language_comprehension.get("action_items"))
        if curiosity_result.get("drive") is not None:
            decision_context.setdefault("curiosity_drive", curiosity_result.get("drive"))
        if self.personality_state.get("dynamic"):
            decision_context.setdefault("personality_traits", self.personality_state.get("dynamic"))
        if knowledge_updates and knowledge_updates.get("added"):
            decision_context.setdefault("knowledge_updates", knowledge_updates)
        if self.self_model_state:
            decision_context.setdefault("self_beliefs", self.self_model_state.get("beliefs"))
            decision_context.setdefault("self_alerts", self.self_model_state.get("alerts"))

        attention_focus_bundle = self._update_attention_blackboard(
            perception_result=perception_result,
            attention_result=attention_result,
            memory_result=memory_result,
            persistent_snapshot=persistent_snapshot,
            motivation_snapshot=motivation_snapshot,
            curiosity_result=curiosity_result,
            decision_context=decision_context,
            inputs=inputs,
        )
        if attention_focus_bundle is not None:
            decision_context.setdefault("attention_focus", attention_focus_bundle.get("focus", []))
            decision_context.setdefault("attention_scores", attention_focus_bundle.get("scores", []))
            memory_result.setdefault("attention_focus", attention_focus_bundle.get("focus", []))
            attention_result.setdefault("managed_focus", attention_focus_bundle)
            self.attention_focus_state = attention_focus_bundle
        else:
            self.attention_focus_state = {}

        decision_options = list(inputs.get("decision_options", []))
        if language_comprehension is not None:
            action_items = language_comprehension.get("action_items")
            candidates: List[Any] = []
            if isinstance(action_items, list):
                candidates = action_items
            elif isinstance(action_items, str):
                candidates = [action_items]
            for item in candidates:
                text = str(item).strip()
                if text and text not in decision_options:
                    decision_options.append(text)
        for action in self.pending_research_actions:
            if action not in decision_options:
                decision_options.append(action)

        plan_result = None
        if getattr(self, "planner_enabled", False) and self.planner is not None:
            try:
                plan_result = self.planner.generate_plan(
                    decision_context,
                    inputs.get("goals", []),
                    decision_options,
                    self.knowledge_constraints,
                )
                decision_context.setdefault("planner", plan_result)
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.exception("Planner execution failed: %s", exc)
                plan_result = {"error": str(exc)}
        if isinstance(plan_result, dict):
            self._last_plan_result = plan_result

        # Track an optional multi-step plan sequence so decision modules can
        # follow a stable action order across simulation steps.
        plan_sequence: List[str] = []
        if isinstance(plan_result, dict):
            seq = plan_result.get("sequence")
            if isinstance(seq, list):
                for item in seq:
                    if item is None:
                        continue
                    text = str(item).strip()
                    if text and text not in plan_sequence:
                        plan_sequence.append(text)

        raw_goals = inputs.get("goals", [])
        if isinstance(raw_goals, (list, tuple, set)):
            goal_list = [str(g).strip() for g in raw_goals if str(g).strip()]
        elif raw_goals:
            goal_list = [str(raw_goals).strip()]
        else:
            goal_list = []

        if plan_sequence:
            if goal_list != self._active_plan_goals or plan_sequence != self._active_plan_sequence:
                self._active_plan_goals = list(goal_list)
                self._active_plan_sequence = list(plan_sequence)
                self._active_plan_cursor = 0
        elif goal_list and goal_list != self._active_plan_goals:
            self._active_plan_goals = list(goal_list)
            self._active_plan_sequence = []
            self._active_plan_cursor = 0

        next_action: Optional[str] = None
        if self._active_plan_sequence and 0 <= self._active_plan_cursor < len(self._active_plan_sequence):
            next_action = self._active_plan_sequence[self._active_plan_cursor]

        if self._active_plan_sequence:
            decision_context.setdefault("plan_sequence", list(self._active_plan_sequence))
        if next_action is not None:
            decision_context.setdefault("plan_next_action", next_action)
            if next_action not in decision_options:
                decision_options.append(next_action)

        constraint_evaluations: Dict[str, Any] = {}
        options = decision_options
        if self.knowledge_constraints:
            evaluated_actions: List[Any] = []
            if isinstance(options, (list, tuple)):
                evaluated_actions.extend(options)
            if isinstance(plan_result, dict):
                for candidate in plan_result.get("candidates", []):
                    action = candidate.get("action") if isinstance(candidate, dict) else candidate
                    if action is not None:
                        evaluated_actions.append(action)
            for action in evaluated_actions:
                key = str(action)
                if key not in constraint_evaluations:
                    constraint_evaluations[key] = self.knowledge_graph.evaluate_action_constraints(
                        action,
                        self.knowledge_constraints,
                    )
        if constraint_evaluations:
            decision_context.setdefault("constraint_evaluations", constraint_evaluations)
            if isinstance(plan_result, dict):
                summary: Dict[str, Any] = {}
                for action_key, evaluation in constraint_evaluations.items():
                    summary[action_key] = evaluation
                plan_result.setdefault("constraint_summary", summary)

        decision_result = self.decision.process({
            "options": decision_options,
            "context": decision_context,
            "reward": inputs.get("reward", None)
        })

        if constraint_evaluations:
            decision_result = self._apply_constraint_arbitration(
                decision_result,
                constraint_evaluations,
                decision_options,
                plan_result,
            )

        if isinstance(decision_result, dict):
            predicted = decision_result.get("predicted_reward")
            if predicted is not None:
                try:
                    self._last_predicted_reward = float(predicted)
                except Exception:
                    pass

        # Advance the active plan cursor if the selected action matches (or is part of) the plan.
        if isinstance(decision_result, dict) and self._active_plan_sequence:
            try:
                chosen = decision_result.get("decision")
                chosen_key = str(chosen).strip() if chosen is not None else ""
            except Exception:  # pragma: no cover - defensive fallback
                chosen_key = ""

            if chosen_key:
                try:
                    if 0 <= self._active_plan_cursor < len(self._active_plan_sequence):
                        expected = self._active_plan_sequence[self._active_plan_cursor]
                        if chosen_key == expected:
                            self._active_plan_cursor += 1
                        else:
                            try:
                                idx = self._active_plan_sequence.index(chosen_key)
                            except ValueError:
                                idx = -1
                            if idx >= self._active_plan_cursor:
                                self._active_plan_cursor = idx + 1

                    if self._active_plan_cursor >= len(self._active_plan_sequence):
                        self._active_plan_sequence = []
                        self._active_plan_cursor = 0
                except Exception:  # pragma: no cover - defensive fallback
                    pass

        if requested_language_generation and language_module is not None:
            base_snapshot = language_snapshot if isinstance(language_snapshot, dict) else {}
            generation_goal = pending_language_goal
            if not isinstance(generation_goal, dict):
                intent = "inform"
                if isinstance(language_comprehension, dict):
                    intent_map = {"question": "answer", "command": "confirm"}
                    intent = intent_map.get(str(language_comprehension.get("intent", "")).lower(), "inform")
                decision_text = None
                if isinstance(decision_result, dict):
                    decision_value = decision_result.get("decision")
                    decision_text = str(decision_value).strip() if decision_value is not None else None
                reference = f"Selected action: {decision_text}" if decision_text else "Decision selected."
                key_terms = []
                if isinstance(language_comprehension, dict):
                    key_terms = list(language_comprehension.get("key_terms") or [])[:4]
                generation_goal = {"intent": intent, "reference": reference, "key_terms": key_terms}

            raw_language_context = auto_query_inputs.get("language_context")
            generation_context = dict(raw_language_context) if isinstance(raw_language_context, dict) else {}
            generation_context.setdefault("timestamp", float(self.current_time))
            if isinstance(decision_result, dict):
                generation_context.setdefault("decision", decision_result)
            if transient_retrievals:
                generation_context.setdefault("memory_retrieved", transient_retrievals)

            post_inputs = dict(inputs)
            post_inputs["language_context"] = generation_context
            post_inputs["language_generation"] = generation_goal
            post_inputs["generate_language_response"] = True
            post_inputs.pop("language_input", None)
            try:
                post_snapshot = language_module.process(post_inputs)
            except Exception as exc:  # pragma: no cover - optional module failure
                post_snapshot = {"error": str(exc)}

            if isinstance(post_snapshot, dict):
                merged = dict(base_snapshot)
                if "generation" in post_snapshot:
                    merged["generation"] = post_snapshot.get("generation")
                if "status" in post_snapshot:
                    merged["status"] = post_snapshot.get("status")
                if "dialogue_state" in post_snapshot:
                    merged["dialogue_state"] = post_snapshot.get("dialogue_state")
                if "usage_stats" in post_snapshot:
                    merged["usage_stats"] = post_snapshot.get("usage_stats")
                language_snapshot = merged

        if self.self_supervised_enabled and self.self_supervised is not None:
            try:
                self.self_supervised.record_action(
                    decision_result.get("decision"),
                    metadata={
                        "confidence": decision_result.get("confidence"),
                        "score": decision_result.get("score"),
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.debug("Self-supervised action logging failed: %s", exc)

        self_model_report = None
        if self.self_model_enabled and self.self_model is not None:
            try:
                attention_focus = None
                attention_scores = None
                if isinstance(attention_focus_bundle, dict):
                    attention_focus = attention_focus_bundle.get("focus")
                    attention_scores = attention_focus_bundle.get("scores")
                elif isinstance(attention_result, dict):
                    attention_focus = attention_result.get("attention_focus")
                motivation_vector = None
                if isinstance(motivation_snapshot, dict):
                    motivation_vector = motivation_snapshot.get("motivation")
                working_memory_stats = memory_result.get("statistics") if isinstance(memory_result, dict) else None
                self_model_report = self.self_model.observe(
                    time_point=self.current_time,
                    goals=inputs.get("goals", []),
                    decision_result=decision_result if isinstance(decision_result, dict) else None,
                    attention_focus=attention_focus,
                    attention_scores=attention_scores,
                    motivation=motivation_vector if isinstance(motivation_vector, dict) else None,
                    emotion_state=emotion_result if isinstance(emotion_result, dict) else None,
                    memory_result=memory_result if isinstance(memory_result, dict) else None,
                    meta_analysis=None,
                    reward=inputs.get("reward"),
                    plan_result=plan_result if isinstance(plan_result, dict) else None,
                    working_memory=working_memory_stats if isinstance(working_memory_stats, dict) else None,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.debug("Self-awareness observation failed: %s", exc)
                self_model_report = None

        if isinstance(self_model_report, dict):
            self.self_model_state = self_model_report
            decision_context["self_beliefs"] = self_model_report.get("beliefs", {})
            decision_context["self_alerts"] = self_model_report.get("alerts", [])

        motor_result = self._process_motor(decision_result, inputs)

        meta_analysis = None
        if getattr(self, "meta_reasoner_enabled", False) and self.meta_reasoner is not None:
            try:
                meta_analysis = self.meta_reasoner.evaluate(
                    plan_result or {},
                    decision_result,
                    decision_context,
                )
            except Exception as exc:  # pragma: no cover
                self.logger.exception("Meta reasoning failed: %s", exc)
                meta_analysis = {"error": str(exc)}

        if meta_analysis is not None and isinstance(self.self_model_state, dict):
            self.self_model_state.setdefault("meta", meta_analysis)

        metacognition_snapshot = None
        meta_learning_rate_scale = 1.0
        if getattr(self, "metacognition_enabled", False) and getattr(self, "metacognition", None) is not None:
            try:
                language_context_for_meta = None
                if isinstance(auto_query_inputs, dict):
                    candidate = auto_query_inputs.get("language_context")
                    if isinstance(candidate, dict):
                        language_context_for_meta = candidate
                if language_context_for_meta is None:
                    candidate = inputs.get("language_context")
                    if isinstance(candidate, dict):
                        language_context_for_meta = candidate

                metacognition_snapshot = self.metacognition.update(
                    time_point=self.current_time,
                    decision_result=decision_result if isinstance(decision_result, dict) else None,
                    emotion_result=emotion_result if isinstance(emotion_result, dict) else None,
                    meta_analysis=meta_analysis if isinstance(meta_analysis, dict) else None,
                    self_model=self.self_model_state if isinstance(self.self_model_state, dict) else None,
                    attention_focus_bundle=attention_focus_bundle if isinstance(attention_focus_bundle, dict) else None,
                    memory_result=memory_result if isinstance(memory_result, dict) else None,
                    language_context=language_context_for_meta,
                    self_supervised_summary=self.self_supervised_summary if isinstance(self.self_supervised_summary, dict) else None,
                    plan_result=plan_result if isinstance(plan_result, dict) else None,
                    decision_context=decision_context if isinstance(decision_context, dict) else None,
                )
                if isinstance(metacognition_snapshot, dict):
                    self.metacognition_state = metacognition_snapshot
                    commands = metacognition_snapshot.get("commands")
                    if isinstance(commands, dict):
                        scale = commands.get("learning_rate_scale")
                        if isinstance(scale, (int, float)):
                            meta_learning_rate_scale = float(scale)
            except Exception as exc:  # pragma: no cover - optional path
                self.logger.debug("Metacognition update failed: %s", exc)
                metacognition_snapshot = None
                meta_learning_rate_scale = 1.0
                self.metacognition_state = {}

        # Provide neuromodulators to backends that support direct modulation (e.g. biophysical STDP).
        if isinstance(emotion_result, dict):
            neuromodulators = emotion_result.get("neuromodulators")
            if isinstance(neuromodulators, dict) and neuromodulators:
                setter = getattr(self.network, "set_neuromodulators", None)
                if callable(setter):
                    try:
                        setter(neuromodulators)
                    except Exception:  # pragma: no cover
                        pass

        # Optional: allow callers to directly drive the neural backend for this step.
        raw_network_input = inputs.get("network_input")
        if raw_network_input is not None:
            try:
                if isinstance(raw_network_input, (int, float)):
                    self.network.set_input([float(raw_network_input)])
                else:
                    self.network.set_input(list(raw_network_input))
            except Exception:  # pragma: no cover
                pass

        # 执行网络更新
        network_state = self.network.step(dt)

        # Provide neuromodulators to learning rules that support them (e.g. neuromodulated STDP).
        if isinstance(network_state, dict) and isinstance(emotion_result, dict):
            neuromodulators = emotion_result.get("neuromodulators")
            if isinstance(neuromodulators, dict) and neuromodulators:
                existing = network_state.get("neuromodulators")
                if isinstance(existing, dict):
                    existing.update(neuromodulators)
                else:
                    network_state["neuromodulators"] = dict(neuromodulators)

        # 应用学习规则
        learning_scale = float(meta_learning_rate_scale) if isinstance(meta_learning_rate_scale, (int, float)) else 1.0
        if not np.isfinite(learning_scale) or learning_scale <= 0:
            learning_scale = 1.0
        learning_scale = float(np.clip(learning_scale, 0.05, 5.0))

        patched_rules: List[Tuple[Dict[str, Any], float]] = []
        if abs(learning_scale - 1.0) > 1e-6:
            for rule in self.learning_rules:
                params = getattr(rule, "params", None)
                if not isinstance(params, dict):
                    continue
                base_lr = params.get("learning_rate")
                if isinstance(base_lr, (int, float)):
                    patched_rules.append((params, float(base_lr)))
                    params["learning_rate"] = float(base_lr) * learning_scale

        hebbian_potentiation = None
        if abs(learning_scale - 1.0) > 1e-6 and getattr(self, "hebbian_enabled", False):
            try:
                hebbian_potentiation = float(self.hebbian_params.get("potentiation", 0.0))
                self.hebbian_params["potentiation"] = hebbian_potentiation * learning_scale
            except Exception:
                hebbian_potentiation = None

        try:
            for rule in self.learning_rules:
                rule.update(network_state, dt)
            hebbian_stats = self._apply_hebbian_plasticity(network_state, dt)
        finally:
            for params, base_lr in patched_rules:
                params["learning_rate"] = base_lr
            if hebbian_potentiation is not None:
                try:
                    self.hebbian_params["potentiation"] = hebbian_potentiation
                except Exception:
                    pass

        # 记录结果
        self.simulation_results["times"].append(self.current_time)
        self.simulation_results["spikes"].append(network_state["spikes"])
        self.simulation_results["voltages"].append(network_state["voltages"])
        self.simulation_results["weights"].append(network_state["weights"])
        if "predictive_error" in self.simulation_results:
            self.simulation_results["predictive_error"].append(
                self.self_supervised_summary.get("prediction_error") if self.self_supervised_summary else None
            )
        if "reconstruction_error" in self.simulation_results:
            self.simulation_results["reconstruction_error"].append(
                self.self_supervised_summary.get("reconstruction_loss") if self.self_supervised_summary else None
            )
        if "self_supervised" in self.simulation_results:
            self.simulation_results["self_supervised"].append(self.self_supervised_summary or {})

        # 记录认知状态
        cognitive_state = {
            "perception": perception_result,
            "attention": attention_result,
            "memory": memory_result,
            "decision": decision_result,
            "emotion": emotion_result,
            "curiosity": curiosity_result,
            "self_supervised": self.self_supervised_summary,
            "personality": self.personality_state,
        }
        if language_snapshot is not None:
            cognitive_state["language"] = language_snapshot
        if hebbian_stats:
            cognitive_state.setdefault("plasticity", {})["hebbian"] = hebbian_stats
        if plan_result is not None:
            cognitive_state["planner"] = plan_result
        if meta_analysis is not None:
            cognitive_state["meta_reasoning"] = meta_analysis
        if motor_result is not None:
            cognitive_state["motor"] = motor_result
        if self.attention_focus_state:
            cognitive_state.setdefault("attention_manager", self.attention_focus_state)
        if self.self_model_state:
            cognitive_state["self_model"] = self.self_model_state
        if metacognition_snapshot is not None:
            cognitive_state["metacognition"] = metacognition_snapshot
        self.simulation_results["cognitive_states"].append(cognitive_state)

        if self._experience_enabled():
            cfg = self._experience_config()
            try:
                interval = int(cfg.get("interval_steps", 1))
            except Exception:
                interval = 1
            if interval <= 0:
                interval = 1

            self._experience_step += 1
            if self._experience_step % interval == 0:
                pending_action = None
                pending_confidence = None
                pending_predicted_reward = None
                if isinstance(decision_result, dict):
                    pending_action = decision_result.get("decision")
                    pending_confidence = decision_result.get("confidence")
                    pending_predicted_reward = decision_result.get("predicted_reward")

                concept = None
                semantic_focus = None
                if isinstance(attention_directives, dict):
                    semantic_focus = attention_directives.get("semantic_focus")
                if isinstance(semantic_focus, str) and semantic_focus.strip():
                    concept = semantic_focus.strip()
                elif isinstance(semantic_focus, (list, tuple)) and semantic_focus:
                    for item in semantic_focus:
                        text = str(item).strip()
                        if text:
                            concept = text
                            break

                self._pending_experience = {
                    "timestamp": float(self.current_time),
                    "stage": (self.config.get("metadata", {}) or {}).get("stage"),
                    "concept": concept,
                    "decision": pending_action,
                    "confidence": pending_confidence,
                    "predicted_reward": pending_predicted_reward,
                    "perception_context": perception_context if isinstance(perception_context, dict) else {},
                }

        # 触发事件
        self._trigger_event("step", {
            "time": self.current_time,
            "network_state": network_state,
            "cognitive_state": cognitive_state
        })

        return {
            "time": self.current_time,
            "network_state": network_state,
            "cognitive_state": cognitive_state
        }

    def sleep_consolidate(
        self,
        *,
        duration: float = 1_000.0,
        dt: float = 100.0,
        max_traces: int = 32,
        extract_knowledge: bool = True,
    ) -> Dict[str, Any]:
        """Run an offline sleep/consolidation phase without stepping the neural network."""

        try:
            duration_value = float(duration)
        except Exception:
            duration_value = 0.0
        try:
            dt_value = float(dt)
        except Exception:
            dt_value = 0.0
        if dt_value <= 0:
            dt_value = 100.0
        steps = int(duration_value / dt_value) if duration_value > 0 else 0
        if steps <= 0 and duration_value > 0:
            steps = 1

        memory_process = getattr(self, "memory", None)
        memory_system = getattr(memory_process, "memory_system", None) if memory_process is not None else None
        runner = getattr(memory_process, "_run_async", None) if memory_process is not None else None
        if memory_system is None or not callable(runner):
            return {"steps": steps, "memory": {}, "knowledge_triples": 0, "skipped": True}

        last_stats: Dict[str, Any] = {}
        for _ in range(steps):
            self.current_time += dt_value
            try:
                snapshot = runner(memory_system.update_memory_system(dt_value, sleep_mode=True))
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.debug("Sleep consolidation update failed: %s", exc)
                break
            if isinstance(snapshot, dict):
                last_stats = snapshot

        inserted_total = 0
        if extract_knowledge and self.knowledge_graph is not None:
            hippocampal = getattr(memory_system, "hippocampal_system", None)
            ca1 = getattr(hippocampal, "ca1_memories", None) if hippocampal is not None else None
            if isinstance(ca1, dict) and ca1:
                try:
                    trace_limit = int(max_traces)
                except Exception:
                    trace_limit = 32
                trace_limit = max(0, min(trace_limit, 512))
                for trace in list(ca1.values())[:trace_limit]:
                    content = getattr(trace, "content", None)
                    if not isinstance(content, dict) or not content:
                        continue
                    triples: List[Triple] = []
                    raw_triples = content.get("triples")
                    if isinstance(raw_triples, list):
                        for entry in raw_triples[:64]:
                            if isinstance(entry, (list, tuple)) and len(entry) == 3:
                                triples.append((str(entry[0]), str(entry[1]), str(entry[2])))
                    relations = content.get("relations")
                    if isinstance(relations, list):
                        for rel in relations[:64]:
                            if not isinstance(rel, dict):
                                continue
                            head = str(rel.get("head", "")).strip()
                            dependent = str(rel.get("dependent", "")).strip()
                            relation = str(rel.get("relation", "")).strip()
                            if not head or not dependent or not relation:
                                continue
                            if head.upper() == "ROOT":
                                continue
                            triples.append((head, relation, dependent))
                    if content.get("event") == "experience":
                        triples.extend(self._extract_experience_triples(content, max_triples=16))
                    if not triples:
                        continue
                    try:
                        inserted_total += self.knowledge_graph.upsert_triples(
                            triples,
                            default_metadata={
                                "source": "sleep_consolidation",
                                "timestamp": float(self.current_time),
                            },
                        )
                    except Exception:
                        continue

        return {"steps": steps, "memory": last_stats, "knowledge_triples": inserted_total}

    def _apply_constraint_arbitration(
        self,
        decision_result: Dict[str, Any],
        constraint_evaluations: Dict[str, Any],
        options: Optional[Sequence[Any]],
        plan_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(decision_result, dict):
            return decision_result

        normalized_map: Dict[str, Any] = {str(k): v for k, v in constraint_evaluations.items()}
        conflicts: List[Dict[str, Any]] = []

        decision = decision_result.get("decision")
        decision_key = str(decision) if decision is not None else None

        def satisfied(key: Optional[str]) -> bool:
            if key is None:
                return True
            info = normalized_map.get(key)
            return info is None or info.get("satisfied", True)

        if decision_key and not satisfied(decision_key):
            info = normalized_map.get(decision_key, {})
            conflicts.append(
                {
                    "action": decision,
                    "violations": info.get("violations", []),
                }
            )
            candidate_actions: List[Any] = []
            if isinstance(plan_result, dict):
                for candidate in plan_result.get("candidates", []):
                    if isinstance(candidate, dict):
                        action = candidate.get("action")
                    else:
                        action = candidate
                    if action is not None:
                        candidate_actions.append(action)
            if isinstance(options, (list, tuple)):
                candidate_actions.extend(options)

            replacement: Optional[Any] = None
            for alt in candidate_actions:
                alt_key = str(alt)
                if alt_key == decision_key:
                    continue
                info_alt = normalized_map.get(alt_key)
                if info_alt is None:
                    info_alt = self.knowledge_graph.evaluate_action_constraints(
                        alt,
                        self.knowledge_constraints,
                    )
                    normalized_map[alt_key] = info_alt
                if info_alt.get("satisfied", True):
                    replacement = alt
                    break

            resolution = decision_result.setdefault("constraint_resolution", {})
            resolution["previous"] = decision
            if replacement is not None:
                decision_result.setdefault("notes", []).append(
                    f"Decision adjusted from {decision} to {replacement} due to constraint violation."
                )
                decision_result["decision"] = replacement
                decision_result["confidence"] = min(
                    float(decision_result.get("confidence", 0.0)),
                    0.6,
                )
                resolution["selected_alternative"] = replacement
            else:
                resolution["selected_alternative"] = None
                decision_result.setdefault("notes", []).append(
                    f"Decision {decision} violates constraints but no alternative satisfied them."
                )

        decision_result["constraint_conflicts"] = conflicts
        decision_result["constraint_evaluations"] = normalized_map
        return decision_result

    def _process_motor(self, decision_result: Dict[str, Any], inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not getattr(self, "motor_enabled", False) or self.motor_system is None:
            return None

        intent: Any = inputs.get("motor_intent")
        if intent is None and isinstance(decision_result, dict):
            intent = decision_result.get("decision")
        if intent is None:
            return None

        feedback = inputs.get("motor_feedback", {}) or {}

        try:
            result = self.motor_system.compute(intent, feedback)
            self.motor_last_command = result.get("commands")
            self.motor_status.pop("error", None)
            return result
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.exception("Motor control failed: %s", exc)
            self.motor_status["error"] = str(exc)
            return {"error": str(exc)}

    def _update_attention_blackboard(
        self,
        *,
        perception_result: Dict[str, Any],
        attention_result: Dict[str, Any],
        memory_result: Dict[str, Any],
        persistent_snapshot: Optional[Dict[str, Any]],
        motivation_snapshot: Optional[Dict[str, Any]],
        curiosity_result: Dict[str, Any],
        decision_context: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.attention_manager_enabled or self.attention_manager is None:
            return None

        manager = self.attention_manager
        try:
            perception_vector = perception_result.get("perception_output")
            if isinstance(perception_vector, list):
                vector = [float(v) for v in perception_vector if isinstance(v, (int, float))]
                summary = {
                    "type": "sensory_vector",
                    "mean": float(np.mean(vector)) if vector else 0.0,
                    "max": float(np.max(vector)) if vector else 0.0,
                    "count": len(perception_vector),
                }
                self._add_blackboard_item(
                    "perception.core",
                    raw_payload=vector or perception_vector,
                    payload=summary,
                )

            for modality in ("vision", "auditory", "somatosensory"):
                modality_data = perception_result.get(modality)
                if isinstance(modality_data, dict) and modality_data:
                    summary = self._sanitize_payload(modality_data)
                    summary["modality"] = modality
                    self._add_blackboard_item(
                        f"perception.{modality}",
                        raw_payload=modality_data,
                        payload=summary,
                    )

            fusion_data = perception_result.get("multimodal_fusion")
            if isinstance(fusion_data, dict) and fusion_data:
                fusion_summary = self._sanitize_payload(fusion_data)
                fusion_summary["modality"] = "multimodal_fusion"
                self._add_blackboard_item(
                    "perception.multimodal_fusion",
                    raw_payload=fusion_data,
                    payload=fusion_summary,
                )

            if isinstance(self.self_supervised_summary, dict) and self.self_supervised_summary:
                latent_preview = self.self_supervised_summary.get("latent_preview")
                if latent_preview is not None:
                    latent_summary = {
                        "type": "latent_preview",
                        "dim": len(latent_preview) if isinstance(latent_preview, list) else None,
                    }
                    self._add_blackboard_item(
                        "self_supervised.latent",
                        raw_payload=latent_preview,
                        payload=latent_summary,
                    )

            focus_payload = attention_result.get("attention_focus")
            if focus_payload is not None:
                self._add_blackboard_item(
                    "attention.focus_process",
                    raw_payload=focus_payload,
                )

            attention_map = attention_result.get("attention_map")
            if attention_map is not None:
                self._add_blackboard_item(
                    "attention.map",
                    raw_payload=attention_map,
                )

            retrieved_memories = memory_result.get("retrieved", [])
            if isinstance(retrieved_memories, list):
                for item in retrieved_memories[:5]:
                    if isinstance(item, dict):
                        payload = {
                            "content": item.get("content"),
                            "score": item.get("score"),
                            "goal": item.get("metadata", {}).get("goal") if isinstance(item.get("metadata"), dict) else None,
                        }
                        confidence = item.get("score")
                        self._add_blackboard_item(
                            "memory.retrieved",
                            raw_payload=item,
                            payload=payload,
                            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                        )

            if isinstance(persistent_snapshot, dict):
                retrieved = persistent_snapshot.get("retrieved", [])
                if isinstance(retrieved, list):
                    for item in retrieved[:5]:
                        if isinstance(item, dict):
                            payload = {
                                "content": item.get("content"),
                                "score": item.get("score"),
                                "timestamp": item.get("timestamp"),
                            }
                            confidence = item.get("score")
                            self._add_blackboard_item(
                                "memory.persistent",
                                raw_payload=item,
                                payload=payload,
                                confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                            )

            if isinstance(motivation_snapshot, dict):
                motivation_vector = motivation_snapshot.get("motivation", {})
                if isinstance(motivation_vector, dict):
                    for goal, value in list(motivation_vector.items())[:8]:
                        if isinstance(value, (int, float)):
                            payload = {"goal": goal, "value": float(value)}
                            strength = float(np.clip(value, 0.0, 1.0))
                            self._add_blackboard_item(
                                "motivation.drive",
                                raw_payload=payload,
                                payload=payload,
                                salience=strength,
                                confidence=strength,
                            )

            curiosity_drive = curiosity_result.get("drive")
            if isinstance(curiosity_drive, (int, float)):
                drive_value = float(np.clip(curiosity_drive, -1.0, 1.0))
                self._add_blackboard_item(
                    "curiosity.drive",
                    raw_payload={"drive": drive_value},
                    salience=float(abs(drive_value)),
                    novelty=float(abs(drive_value)),
                )

            goals = inputs.get("goals", [])
            if goals:
                self._add_blackboard_item(
                    "task.goals",
                    raw_payload={"goals": goals},
                )

            options = inputs.get("decision_options", [])
            if options:
                self._add_blackboard_item(
                    "task.options",
                    raw_payload={"options": options},
                )

            attention_context: Dict[str, Any] = {}
            urgency = decision_context.get("task_urgency")
            if urgency is None:
                urgency = inputs.get("task_urgency", inputs.get("urgency"))
            if isinstance(urgency, (int, float)):
                attention_context["task_urgency"] = float(np.clip(urgency, 0.0, 1.0))
            else:
                attention_context["task_urgency"] = 0.5

            attention_context["goal_count"] = len(goals) if isinstance(goals, (list, tuple)) else 0
            attention_context["option_count"] = len(options) if isinstance(options, (list, tuple)) else 0

            motivation_vector: Dict[str, float] = {}
            if isinstance(motivation_snapshot, dict):
                raw_vector = motivation_snapshot.get("motivation", {})
                if isinstance(raw_vector, dict):
                    for key, value in raw_vector.items():
                        if isinstance(value, (int, float)):
                            motivation_vector[str(key)] = float(np.clip(value, 0.0, 1.0))

            focus_bundle = manager.select_focus(
                context=attention_context,
                motivation=motivation_vector,
            )
            focus_bundle.setdefault("context", attention_context)
            focus_bundle.setdefault("motivation", motivation_vector)
            focus_bundle.setdefault("workspace_size", len(manager.workspace.entries))
            return focus_bundle
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.debug("Attention manager update failed: %s", exc)
            return None

    def _apply_hebbian_plasticity(self, network_state: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """Apply a simplified Hebbian update based on co-activation."""

        if not getattr(self, "hebbian_enabled", False):
            return {}

        synapse_manager = getattr(self.network, "synapse_manager", None)
        synapses = getattr(synapse_manager, "synapses", {}) if synapse_manager is not None else {}
        if not synapses:
            return {}

        spikes_raw: List[Dict[str, Any]] = []
        if isinstance(network_state, dict):
            spikes_raw = network_state.get("spikes", []) or []
        mapping = getattr(self.network, "_column_neuron_to_global", {})
        spiking_neurons: set[int] = set()
        for entry in spikes_raw:
            if not isinstance(entry, dict):
                continue
            gid = entry.get("neuron_global")
            if gid is None:
                column = entry.get("column")
                neuron = entry.get("neuron")
                if column is not None and neuron is not None:
                    gid = mapping.get((int(column), int(neuron)))
            if gid is None:
                continue
            try:
                spiking_neurons.add(int(gid))
            except (TypeError, ValueError):
                continue

        potentiation = float(self.hebbian_params.get("potentiation", 0.0))
        decay = float(self.hebbian_params.get("decay", 0.0))
        max_weight = float(self.hebbian_params.get("max_weight", 5.0))
        min_weight = float(self.hebbian_params.get("min_weight", -5.0))

        updated = 0
        coactive_pairs = 0

        for synapse in synapses.values():
            if hasattr(synapse, "current_weight"):
                weight_attr = "current_weight"
            elif hasattr(synapse, "weight"):
                weight_attr = "weight"
            else:
                continue

            try:
                weight = float(getattr(synapse, weight_attr))
            except (TypeError, ValueError):
                continue

            pre_id = getattr(synapse, "pre_neuron_id", None)
            post_id = getattr(synapse, "post_neuron_id", None)
            pre_active = pre_id in spiking_neurons if pre_id is not None else False
            post_active = post_id in spiking_neurons if post_id is not None else False

            if pre_active and post_active:
                coactive_pairs += 1
                sign = 1.0 if weight >= 0.0 else -1.0
                weight += sign * potentiation
            elif decay > 0.0:
                attenuation = max(0.0, 1.0 - decay * dt)
                weight *= attenuation

            weight = float(np.clip(weight, min_weight, max_weight))
            setattr(synapse, weight_attr, weight)
            if weight_attr == "current_weight" and hasattr(synapse, "weight"):
                synapse.weight = weight
            updated += 1

        weights_view = network_state.get("weights") if isinstance(network_state, dict) else None
        if isinstance(weights_view, dict):
            runtime_synapses = getattr(self.network, "_runtime_synapses", [])
            for entry in runtime_synapses:
                if not isinstance(entry, dict):
                    continue
                key = entry.get("key")
                syn = entry.get("synapse")
                if key in weights_view and syn is not None:
                    new_weight = None
                    if hasattr(syn, "weight"):
                        new_weight = getattr(syn, "weight")
                    elif hasattr(syn, "current_weight"):
                        new_weight = getattr(syn, "current_weight")
                    if new_weight is not None:
                        try:
                            weights_view[key] = float(new_weight)
                        except (TypeError, ValueError):
                            pass

        stats = {"updated_synapses": updated}
        if spiking_neurons:
            stats["active_neurons"] = len(spiking_neurons)
        if coactive_pairs:
            stats["coactive_pairs"] = coactive_pairs

        self.hebbian_metrics = stats
        return stats if updated else {}


    def _add_blackboard_item(
        self,
        source: str,
        *,
        raw_payload: Any = None,
        payload: Optional[Dict[str, Any]] = None,
        salience: Optional[float] = None,
        confidence: Optional[float] = None,
        novelty: Optional[float] = None,
    ) -> None:
        if not self.attention_manager_enabled or self.attention_manager is None:
            return
        if raw_payload is None and payload is None:
            return

        safe_payload = self._sanitize_payload(payload if payload is not None else raw_payload)
        estimate_base = raw_payload if raw_payload is not None else safe_payload
        base_salience, base_confidence, base_novelty = self._estimate_attention_metrics(estimate_base)

        metric_salience = float(np.clip(salience if salience is not None else base_salience, 0.0, 1.0))
        metric_confidence = float(np.clip(confidence if confidence is not None else base_confidence, 0.0, 1.0))
        metric_novelty = float(np.clip(novelty if novelty is not None else base_novelty, 0.0, 1.0))

        self.attention_manager.add(
            source=source,
            payload=safe_payload,
            salience=metric_salience,
            confidence=metric_confidence,
            novelty=metric_novelty,
        )

    @staticmethod
    def _estimate_attention_metrics(payload: Any) -> Tuple[float, float, float]:
        numeric_values = BrainSimulation._extract_numeric_values(payload)

        def _from_payload(keys: Tuple[str, ...]) -> Optional[float]:
            if isinstance(payload, dict):
                for key in keys:
                    value = payload.get(key)
                    if isinstance(value, (int, float)):
                        return float(np.clip(value, 0.0, 1.0))
            return None

        salience = _from_payload(("salience", "saliency", "activation", "score"))
        if salience is None and numeric_values:
            salience = float(np.clip(np.tanh(np.mean(np.abs(numeric_values))), 0.0, 1.0))
        if salience is None:
            salience = 0.5

        confidence = _from_payload(("confidence", "probability", "reliability"))
        if confidence is None and numeric_values:
            confidence = float(np.clip(np.tanh(np.mean(np.abs(numeric_values)) * 0.8), 0.1, 1.0))
        if confidence is None:
            confidence = 0.5

        novelty = _from_payload(("novelty", "surprise"))
        if novelty is None and len(numeric_values) > 1:
            novelty = float(np.clip(np.std(numeric_values), 0.0, 1.0))
        if novelty is None:
            novelty = 0.6

        return salience, confidence, novelty

    @staticmethod
    def _extract_numeric_values(payload: Any, limit: int = 32) -> List[float]:
        values: List[float] = []

        def _append(value: Any) -> None:
            if isinstance(value, (int, float)):
                values.append(float(value))

        if isinstance(payload, np.ndarray):
            payload = payload.tolist()

        if isinstance(payload, dict):
            for value in payload.values():
                if len(values) >= limit:
                    break
                if isinstance(value, (int, float)):
                    _append(value)
                elif isinstance(value, dict):
                    for sub_value in value.values():
                        if len(values) >= limit:
                            break
                        _append(sub_value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if len(values) >= limit:
                            break
                        _append(item)
        elif isinstance(payload, (list, tuple)):
            for item in payload:
                if len(values) >= limit:
                    break
                _append(item)
        elif isinstance(payload, (int, float)):
            _append(payload)

        return values

    @staticmethod
    def _extract_vector_preview(value: Any, limit: int = 32) -> Optional[List[float]]:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            array = value.astype(float, copy=False).reshape(-1)
        else:
            try:
                array = np.asarray(value, dtype=float).reshape(-1)
            except Exception:
                return None
        if array.size == 0:
            return None
        preview = array[: max(1, int(limit))]
        return [float(x) for x in preview]

    def _build_perception_embedding_context(
        self,
        *,
        perception_result: Dict[str, Any],
        attention_directives: Dict[str, Any],
        preview_limit: int = 32,
    ) -> Dict[str, Any]:
        """Build a compact, serialisable perception snapshot for memory/language use."""
        if not isinstance(perception_result, dict) or not perception_result:
            return {}

        limit = int(preview_limit)
        if limit <= 0:
            limit = 32

        context: Dict[str, Any] = {}
        if isinstance(attention_directives, dict):
            weights = attention_directives.get("modality_weights")
            if isinstance(weights, dict) and weights:
                cleaned: Dict[str, float] = {}
                for key, value in weights.items():
                    try:
                        cleaned[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
                if cleaned:
                    context["modality_weights"] = cleaned
            semantic_focus = attention_directives.get("semantic_focus")
            if semantic_focus:
                context["semantic_focus"] = semantic_focus

        for modality in ("vision", "auditory", "somatosensory"):
            payload = perception_result.get(modality)
            if not isinstance(payload, dict) or not payload:
                continue
            embedding = payload.get("embedding")
            preview = self._extract_vector_preview(embedding, limit=limit)
            if preview is None:
                continue
            entry: Dict[str, Any] = {"embedding_preview": preview, "dim": int(len(preview))}
            if "confidence" in payload:
                entry["confidence"] = payload.get("confidence")
            if "attention_weight" in payload:
                entry["attention_weight"] = payload.get("attention_weight")
            context[modality] = entry

        structured_payload = perception_result.get("structured")
        if isinstance(structured_payload, dict) and structured_payload:
            summary = structured_payload.get("summary_embedding")
            if summary is None:
                embeddings = structured_payload.get("embeddings")
                if isinstance(embeddings, dict):
                    summary = embeddings.get("summary")
            preview = self._extract_vector_preview(summary, limit=limit)
            if preview is not None:
                entry = {"summary_embedding_preview": preview, "dim": int(len(preview))}
                if "attention_weight" in structured_payload:
                    entry["attention_weight"] = structured_payload.get("attention_weight")
                context["structured"] = entry

        fusion = perception_result.get("multimodal_fusion")
        if isinstance(fusion, dict) and fusion:
            preview = self._extract_vector_preview(fusion.get("embedding"), limit=limit)
            if preview is not None:
                entry = {
                    "embedding_preview": preview,
                    "dim": int(len(preview)),
                    "modalities": fusion.get("modalities"),
                    "attention_weights": fusion.get("attention_weights"),
                }
                context["multimodal_fusion"] = entry
                # Convenience alias: treat fused embedding as a concept-level summary.
                context.setdefault("concept_embedding_preview", preview)

        return context

    @staticmethod
    def _sanitize_payload(
        payload: Any,
        max_list_items: int = 5,
        max_string_length: int = 256,
    ) -> Dict[str, Any]:
        if isinstance(payload, np.ndarray):
            payload = payload.tolist()

        if isinstance(payload, dict):
            sanitized: Dict[str, Any] = {}
            for key, value in payload.items():
                key_str = str(key)
                if isinstance(value, (int, float, bool)) or value is None:
                    sanitized[key_str] = value
                elif isinstance(value, str):
                    sanitized[key_str] = value[:max_string_length]
                elif isinstance(value, (list, tuple)):
                    preview = []
                    for item in list(value)[:max_list_items]:
                        if isinstance(item, (int, float, bool)) or item is None:
                            preview.append(item)
                        elif isinstance(item, str):
                            preview.append(item[:max_string_length])
                        else:
                            preview.append(str(item)[:max_string_length])
                    sanitized[key_str] = preview
                elif isinstance(value, dict):
                    sanitized[key_str] = BrainSimulation._sanitize_payload(value, max_list_items, max_string_length)
                else:
                    sanitized[key_str] = str(value)[:max_string_length]
            return sanitized

        if isinstance(payload, (list, tuple)):
            preview_numeric = [float(v) for v in payload if isinstance(v, (int, float))]
            if preview_numeric:
                return {
                    "mean": float(np.mean(preview_numeric)),
                    "max": float(np.max(preview_numeric)),
                    "min": float(np.min(preview_numeric)),
                    "count": len(payload),
                }
            preview = [str(payload[i])[:max_string_length] for i in range(min(len(payload), max_list_items))]
            return {"items": preview, "count": len(payload)}

        if isinstance(payload, (int, float, bool)) or payload is None:
            return {"value": payload}

        return {"value": str(payload)[:max_string_length]}

    def run(self, 
            inputs_sequence: List[Dict[str, Any]], 
            duration: float, 
            dt: float) -> Dict[str, Any]:
        """
        运行模拟
        
        Args:
            inputs_sequence: 输入序列，每个时间步的输入数据字典
            duration: 模拟持续时间
            dt: 时间步长
            
        Returns:
            包含模拟结果的字典
        """
        # 重置模拟状态
        self.reset()
        
        # 计算时间步数
        steps = int(duration / dt)
        
        # 触发事件
        self._trigger_event("simulation_start", {
            "duration": duration,
            "dt": dt,
            "steps": steps
        })
        
        # 执行模拟
        for step in range(steps):
            # 获取当前时间步的输入
            inputs = {}
            if step < len(inputs_sequence):
                inputs = inputs_sequence[step]
            
            # 执行一步模拟
            self.step(inputs, dt)
        
        # 触发事件
        self._trigger_event("simulation_end", {
            "results": self.simulation_results
        })
        
        return self.simulation_results

    def get_results(self) -> Dict[str, Any]:
        """Return a deep copy of the latest simulation results."""
        return deepcopy(self.simulation_results)

    def get_metrics(self) -> Dict[str, Any]:
        """Compute lightweight runtime metrics for monitoring dashboards."""
        times = self.simulation_results.get("times", [])
        spikes = self.simulation_results.get("spikes", [])
        voltages = self.simulation_results.get("voltages", [])
        weights = self.simulation_results.get("weights", [])

        total_steps = len(times)
        total_spikes = 0
        for entry in spikes:
            if isinstance(entry, dict):
                for value in entry.values():
                    if isinstance(value, (list, tuple, set)):
                        total_spikes += len(value)
                    elif value is not None:
                        total_spikes += 1
            elif isinstance(entry, (list, tuple, set)):
                total_spikes += len(entry)

        metrics: Dict[str, Any] = {
            "is_running": self.is_running,
            "current_time_ms": float(self.current_time),
            "total_steps": total_steps,
            "total_spikes": int(total_spikes),
            "average_spikes_per_step": float(total_spikes / total_steps) if total_steps else 0.0,
            "recorded_neurons": len(voltages[-1]) if voltages else 0,
            "recorded_synapses": len(weights[-1]) if weights else 0,
        }

        if hasattr(self.network, "neurons"):
            metrics["neurons"] = len(getattr(self.network, "neurons", {}))
        if hasattr(self.network, "total_synapses"):
            try:
                metrics["synapses"] = int(getattr(self.network, "total_synapses"))
            except Exception:
                metrics["synapses"] = int(getattr(self.network, "total_synapses", 0))
        elif hasattr(self.network, "synapses"):
            metrics["synapses"] = len(getattr(self.network, "synapses", {}))

        predictive_errors = [
            value
            for value in self.simulation_results.get("predictive_error", [])
            if isinstance(value, (int, float))
        ]
        if predictive_errors:
            metrics["average_prediction_error"] = float(np.mean(predictive_errors))

        reconstruction_errors = [
            value
            for value in self.simulation_results.get("reconstruction_error", [])
            if isinstance(value, (int, float))
        ]
        if reconstruction_errors:
            metrics["average_reconstruction_loss"] = float(np.mean(reconstruction_errors))

        return metrics

    def get_status(self) -> Dict[str, Any]:
        """Return a consolidated status dictionary for external monitors."""
        status: Dict[str, Any] = {
            "backend": getattr(self.backend, "name", "unknown"),
            "config_profile": self.config.get("metadata", {}).get("profile"),
            "dt_ms": self.config.get("simulation", {}).get("dt", 1.0),
        }
        status.update(self.get_metrics())
        if hasattr(self.network, "global_step"):
            status["global_step"] = getattr(self.network, "global_step")
        return status

    def start(self, dt: Optional[float] = None) -> None:
        """Start a continuous simulation loop using the configured timestep."""
        sim_dt = dt if dt is not None else self.config.get("simulation", {}).get("dt", 1.0)
        if sim_dt <= 0:
            raise ValueError("Simulation timestep 'dt' must be positive")
        self.start_continuous_simulation(float(sim_dt))

    def stop(self) -> None:
        """Stop a running continuous simulation loop."""
        self.stop_continuous_simulation()

    def update_parameters(self, updates: Dict[str, Any]) -> None:
        """Apply runtime configuration overrides to the simulation."""
        if not updates:
            return
        if not isinstance(updates, dict):
            raise TypeError("updates must be provided as a dictionary")

        merged = self._merge_config(self.config, updates)
        self.config = self._apply_defaults(merged)

        component_map = [
            ("perception", self.perception),
            ("attention", self.attention),
            ("memory", self.memory),
            ("decision", self.decision),
        ]
        for key, component in component_map:
            if key in updates and component is not None:
                params = getattr(component, "params", None)
                override = updates.get(key, {})
                if isinstance(params, dict) and isinstance(override, dict):
                    try:
                        new_params = self._merge_config(params, override)
                    except Exception:
                        new_params = dict(params, **override)
                    component.params = new_params
                elif hasattr(component, "configure"):
                    try:
                        component.configure(override)
                    except Exception:
                        self.logger.debug("Component %s does not support configure override", key)

        if "self_supervised" in updates:
            self.config["self_supervised"] = self.config.get("self_supervised", {})
            try:
                self.config["self_supervised"] = self._merge_config(
                    self.config.get("self_supervised", {}), updates.get("self_supervised", {})
                )
            except Exception:
                self.config["self_supervised"] = updates.get("self_supervised", {})
            self._initialize_self_supervised_predictor()

        if "network" in updates and isinstance(updates["network"], dict):
            network_override = updates["network"]
            current_network_cfg = getattr(self.network, "config", {})
            if isinstance(current_network_cfg, dict):
                try:
                    new_network_cfg = self._merge_config(current_network_cfg, network_override)
                except Exception:
                    new_network_cfg = dict(current_network_cfg, **network_override)
                try:
                    self.network.config = new_network_cfg  # type: ignore[attr-defined]
                except Exception:
                    pass
            if network_override.get("learning_rules"):
                self.learning_rules = []
                learning_rules_cfg = self.config.get("network", {}).get("learning_rules", {})
                for rule_name, rule_params in learning_rules_cfg.items():
                    if rule_params.get("enabled", False):
                        try:
                            rule = create_learning_rule(rule_name, self.network, rule_params)
                            self.learning_rules.append(rule)
                        except Exception:
                            self.logger.debug("Failed to update learning rule %s", rule_name)

        if "simulation" in updates:
            sim_cfg = self.config.setdefault("simulation", {})
            dt_override = sim_cfg.get("dt")
            if dt_override is not None and dt_override <= 0:
                raise ValueError("Simulation 'dt' must be positive after update")

    def start_continuous_simulation(self, dt: float) -> None:
        """
        开始连续模拟
        
        Args:
            dt: 时间步长
        """
        if self.is_running:
            return
        
        self.is_running = True
        
        # 创建模拟线程
        self.simulation_thread = threading.Thread(
            target=self._continuous_simulation_loop,
            args=(dt,)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # 触发事件
        self._trigger_event("continuous_simulation_start", {
            "dt": dt
        })
    
    def stop_continuous_simulation(self) -> None:
        """停止连续模拟"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 等待线程结束
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
            self.simulation_thread = None
        
        # 触发事件
        self._trigger_event("continuous_simulation_stop", {
            "results": self.simulation_results
        })
    
    def _continuous_simulation_loop(self, dt: float) -> None:
        """
        连续模拟循环
        
        Args:
            dt: 时间步长
        """
        while self.is_running:
            # 执行一步模拟
            self.step({}, dt)
            
            # 控制模拟速度
            time.sleep(dt / 1000.0)  # 将毫秒转换为秒
    
    def register_event_callback(self, event_name: str, callback: Callable) -> None:
        """
        注册事件回调

        Args:
            event_name: 事件名称
            callback: 回调函数
        """
        if event_name not in self.event_callbacks:
            self.event_callbacks[event_name] = []

        self.event_callbacks[event_name].append(callback)

    def _register_builtin_event_handlers(self) -> None:
        """挂载内建事件处理器，例如动态模块注册表调整。"""

        self.register_event_callback(
            "module_registry_update", self._on_module_registry_update
        )

    def _on_module_registry_update(self, event_data: Dict[str, Any]) -> None:
        """根据事件数据动态增删或替换模块实现。

        event_data 支持以下键：

        - ``updates``: {component: implementation}，替换或新增实现。
        - ``remove``: [component, ...]，卸载并从注册表移除的组件名。
        - ``component_configs``: {component: config}，用于新组件的配置。
        - ``rebuild``: bool，是否立即重建实例。
        """

        updates = event_data.get("updates") or {}
        removals = event_data.get("remove") or []
        component_configs = event_data.get("component_configs") or {}
        rebuild = bool(event_data.get("rebuild", False))

        for component, implementation in updates.items():
            self._module_factory.register(component, implementation)
        for component in removals:
            self._module_factory.unregister(component)
            self._dynamic_module_configs.pop(component, None)
            if self._module_container:
                self._module_container.components.pop(component, None)

        if component_configs:
            self._dynamic_module_configs.update(component_configs)

        if rebuild:
            rebuild_targets = set(updates.keys()) | set(component_configs.keys())
            self._rebuild_modules(targets=rebuild_targets or None)
        elif self._module_container is not None:
            self.dynamic_modules = {
                name: module
                for name, module in self._module_container.components.items()
                if name not in {"perception", "attention", "memory", "decision"}
            }
    
    def _trigger_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """
        触发事件
        
        Args:
            event_name: 事件名称
            event_data: 事件数据
        """
        if event_name in self.event_callbacks:
            for callback in self.event_callbacks[event_name]:
                try:
                    callback(event_data)
                except Exception as e:
                    print(f"事件回调错误: {e}")
    
    def save_state(self, filepath: str) -> None:
        """
        保存模拟状态
        
        Args:
            filepath: 文件路径
        """
        # 创建状态字典
        neurons = getattr(self.network, "neurons", {}) or {}
        synapses = getattr(self.network, "synapses", {}) or {}

        neuron_states = {}
        if hasattr(neurons, "items"):
            neuron_iter = neurons.items()
        else:
            neuron_iter = []

        for neuron_id, neuron in neuron_iter:
            voltage = getattr(neuron, "voltage", getattr(neuron, "_voltage", 0.0))
            spike_history = getattr(
                neuron,
                "spike_history",
                getattr(neuron, "_spike_history", []),
            )
            neuron_states[neuron_id] = {
                "voltage": voltage,
                "spike_history": spike_history,
            }

        synapse_states = {}
        if hasattr(synapses, "items"):
            synapse_iter = synapses.items()
        elif hasattr(synapses, "values"):
            synapse_iter = ((None, synapse) for synapse in synapses.values())
        else:
            synapse_iter = []

        for _, synapse in synapse_iter:
            pre_id = getattr(synapse, "pre_id", None)
            post_id = getattr(synapse, "post_id", None)
            if pre_id is None or post_id is None:
                continue
            weight = getattr(synapse, "weight", getattr(synapse, "_weight", 0.0))
            synapse_states[f"{pre_id}_{post_id}"] = {"weight": weight}

        persistent_snapshot = None
        if getattr(self, "persistent_memory", None) is not None:
            try:
                persistent_snapshot = {
                    "working": self.persistent_memory.working_items(),
                    "retrieved": [],
                }
            except Exception:
                persistent_snapshot = None

        attention_snapshot = None
        if self.attention_manager_enabled and self.attention_manager is not None:
            try:
                manager_state = self.attention_manager.to_dict()
            except Exception:
                manager_state = {}
            focus_copy: Dict[str, Any] = {}
            if isinstance(self.attention_focus_state, dict):
                try:
                    focus_copy = deepcopy(self.attention_focus_state)
                except Exception:
                    focus_copy = dict(self.attention_focus_state)
            attention_snapshot = {
                "manager": manager_state,
                "focus": focus_copy,
            }

        self_model_snapshot = None
        if self.self_model_enabled and self.self_model is not None:
            try:
                self_module_state = self.self_model.to_dict()
            except Exception:
                self_module_state = {}
            self_state_copy: Dict[str, Any] = {}
            if isinstance(self.self_model_state, dict):
                try:
                    self_state_copy = deepcopy(self.self_model_state)
                except Exception:
                    self_state_copy = dict(self.self_model_state)
            self_model_snapshot = {
                "module": self_module_state,
                "state": self_state_copy,
            }

        metacognition_snapshot = None
        if getattr(self, "metacognition_enabled", False) and getattr(self, "metacognition", None) is not None:
            try:
                metacog_module_state = self.metacognition.to_dict()
            except Exception:
                metacog_module_state = {}
            metacog_state_copy: Dict[str, Any] = {}
            if isinstance(getattr(self, "metacognition_state", None), dict):
                try:
                    metacog_state_copy = deepcopy(self.metacognition_state)
                except Exception:
                    metacog_state_copy = dict(self.metacognition_state)
            metacognition_snapshot = {
                "module": metacog_module_state,
                "state": metacog_state_copy,
            }

        comprehensive_memory_snapshot = None
        memory_cfg = self.config.get("memory", {}) if isinstance(self.config, dict) else {}
        persistence_cfg = memory_cfg.get("state_persistence", {}) if isinstance(memory_cfg, dict) else {}
        if not isinstance(persistence_cfg, dict):
            persistence_cfg = {}
        try:
            max_working = int(persistence_cfg.get("max_working", 64))
        except Exception:
            max_working = 64
        try:
            max_episodic = int(persistence_cfg.get("max_episodic", 256))
        except Exception:
            max_episodic = 256
        max_working = max(0, min(max_working, 1_024))
        max_episodic = max(0, min(max_episodic, 5_000))

        memory_process = getattr(self, "memory", None)
        memory_system = getattr(memory_process, "memory_system", None) if memory_process is not None else None
        if memory_system is not None:
            try:
                wm_snapshot = None
                wm = getattr(memory_system, "working_memory", None)
                if wm is not None:
                    items = list(getattr(wm, "items", []) or [])[:max_working]
                    weights = list(getattr(wm, "attention_weights", []) or [])[: len(items)]
                    wm_items = [self._sanitize_payload(item) for item in items]
                    wm_weights = []
                    for weight in weights:
                        try:
                            wm_weights.append(float(weight))
                        except Exception:
                            wm_weights.append(1.0)
                    wm_snapshot = {"items": wm_items, "attention_weights": wm_weights}

                episodic_snapshot = None
                hippocampal = getattr(memory_system, "hippocampal_system", None)
                ca1 = getattr(hippocampal, "ca1_memories", None) if hippocampal is not None else None
                if isinstance(ca1, dict):
                    episodes = []
                    for trace in list(ca1.values())[:max_episodic]:
                        content = getattr(trace, "content", None)
                        context = getattr(trace, "context", None)
                        if not isinstance(content, dict):
                            content = self._sanitize_payload(content)
                        if not isinstance(context, dict):
                            context = self._sanitize_payload(context)
                        state_value = getattr(getattr(trace, "consolidation_state", None), "value", None)
                        if state_value is None:
                            state_value = getattr(getattr(trace, "consolidation_state", None), "name", None)
                        episodes.append(
                            {
                                "trace_id": getattr(trace, "trace_id", None),
                                "content": content,
                                "context": context,
                                "encoding_time": getattr(trace, "encoding_time", None),
                                "last_access_time": getattr(trace, "last_access_time", None),
                                "access_count": getattr(trace, "access_count", None),
                                "strength": getattr(trace, "strength", None),
                                "stability": getattr(trace, "stability", None),
                                "retrievability": getattr(trace, "retrievability", None),
                                "consolidation_state": state_value,
                                "consolidation_progress": getattr(trace, "consolidation_progress", None),
                            }
                        )
                    episodic_snapshot = episodes

                if wm_snapshot is not None or episodic_snapshot is not None:
                    comprehensive_memory_snapshot = {
                        "working_memory": wm_snapshot,
                        "episodic": episodic_snapshot,
                    }
            except Exception:
                comprehensive_memory_snapshot = None

        memory_state = {
            "working_memory": getattr(self.memory, "working_memory", {}),
            "long_term_memory": getattr(self.memory, "long_term_memory", {}),
            "memory_strengths": getattr(self.memory, "memory_strengths", {}),
        }
        if comprehensive_memory_snapshot is not None:
            memory_state["comprehensive"] = comprehensive_memory_snapshot

        state = {
            "config": self.config,
            "current_time": self.current_time,
            "network_state": {
                "neuron_states": neuron_states,
                "synapse_states": synapse_states,
            },
            "cognitive_state": {
                "memory": memory_state,
                "attention": {
                    "focus_state": self.attention_focus_state,
                },
                "decision": {
                    "action_values": getattr(self.decision, "action_values", {}),
                    "decision_history": getattr(self.decision, "decision_history", []),
                },
                "emotion": self.emotion_state,
                "curiosity": {
                    "last_drive": self.last_curiosity_drive,
                },
                "personality": self.personality_state,
                "motor": {
                    "status": self.motor_status,
                    "last_command": self.motor_last_command,
                },
            },
            "knowledge": {
                "symbolic_reasoner": (
                    self.symbolic_reasoner.to_dict()
                    if getattr(self, "symbolic_reasoner", None) is not None
                    else None
                ),
                "constraints": [
                    {
                        "description": str(getattr(constraint, "description", "")),
                        "required": list(getattr(constraint, "required", []) or []),
                        "forbidden": list(getattr(constraint, "forbidden", []) or []),
                    }
                    for constraint in (getattr(self, "knowledge_constraints", []) or [])
                ],
            },
            "persistent_memory": self.persistent_memory.to_dict() if getattr(self, "persistent_memory", None) else None,
            "emotion_motivation": self.emotion_motivation.to_dict() if getattr(self, "emotion_motivation", None) else None,
            "attention": attention_snapshot,
            "self_model": self_model_snapshot,
            "metacognition": metacognition_snapshot,
        }
        # 保存到文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    
    def upgrade_stage(
        self,
        stage: str,
        *,
        overrides: Optional[Dict[str, Any]] = None,
        base_profile: Optional[str] = None,
        preserve_state: bool = True,
    ) -> None:
        """Reconfigure the simulation to a new curriculum stage.

        The stage preset comes from :func:`BrainSimulationSystem.config.stage_profiles.build_stage_config`.
        When ``preserve_state`` is enabled, the method round-trips through the existing
        JSON persistence layer so that matching neuron/synapse state is restored into
        the expanded network.
        """

        next_config = build_stage_config(
            stage,
            overrides=overrides or {},
            base_profile=base_profile,
        )

        import tempfile

        fd, tmp_path = tempfile.mkstemp(prefix="brainsim_stage_", suffix=".json")
        os.close(fd)
        try:
            if preserve_state:
                self.save_state(tmp_path)
                with open(tmp_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            else:
                payload = {
                    "config": {},
                    "current_time": 0.0,
                    "network_state": {"neuron_states": {}, "synapse_states": {}},
                    "cognitive_state": {},
                    "persistent_memory": None,
                    "emotion_motivation": None,
                    "attention": None,
                    "self_model": None,
                    "metacognition": None,
                    "knowledge": None,
                }

            payload["config"] = next_config
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self.load_state(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def load_state(self, filepath: str) -> None:
        """
        加载模拟状态
        
        Args:
            filepath: 文件路径
        """
        # 从文件加载
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        # 更新配置并重新创建网络
        saved_config = state.get("config", {})
        self.config = self._apply_defaults(saved_config)
        self.network = self.backend.build_network(self.config)

        # 重建学习规则
        self.learning_rules = []
        learning_rules_cfg = self.config.get("network", {}).get("learning_rules", {})
        for rule_name, rule_params in learning_rules_cfg.items():
            if isinstance(rule_params, dict) and rule_params.get("enabled", False):
                try:
                    rule = create_learning_rule(rule_name, self.network, rule_params)
                except Exception:  # pragma: no cover - invalid saved configs should not crash load
                    self.logger.debug("Failed to recreate learning rule %s", rule_name)
                    continue
                self.learning_rules.append(rule)

        # 重新创建认知过程以绑定新网络
        modules_cfg = self.config.get("modules", {})
        registry_overrides = modules_cfg.get("registry") if isinstance(modules_cfg, dict) else None
        if not isinstance(registry_overrides, dict):
            registry_overrides = None
        self._module_factory = ModuleFactory(
            self.network,
            registry=registry_overrides,
            logger=self.logger,
        )
        self._register_module_builders()
        self._dynamic_module_configs = self._load_dynamic_module_configs(modules_cfg)
        self._module_container = None
        self._initialize_core_modules()
        self._language_semantic_attached = False
        self._initialize_high_level_systems()
        self._pending_experience = None
        self._experience_step = 0

        knowledge_payload = state.get("knowledge") if isinstance(state, dict) else None
        if isinstance(knowledge_payload, dict):
            symbolic = knowledge_payload.get("symbolic_reasoner")
            if isinstance(symbolic, dict):
                try:
                    restored_reasoner = SymbolicReasoner.from_dict(symbolic)
                except Exception as exc:
                    self.logger.debug("Failed to restore symbolic reasoner: %s", exc)
                else:
                    self.symbolic_reasoner = restored_reasoner
                    self.knowledge_graph = restored_reasoner.knowledge

            constraints = knowledge_payload.get("constraints")
            if isinstance(constraints, list):
                restored_constraints: List[KnowledgeConstraint] = []
                for raw in constraints:
                    if not isinstance(raw, dict):
                        continue
                    try:
                        restored_constraints.append(
                            KnowledgeConstraint(
                                description=str(raw.get("description", "constraint")),
                                required=[tuple(pred) for pred in raw.get("required", [])],
                                forbidden=[tuple(pred) for pred in raw.get("forbidden", [])],
                            )
                        )
                    except Exception:
                        continue
                if restored_constraints:
                    self.knowledge_constraints = restored_constraints

            planner_cfg = self.config.get("planner", {})
            self.planner_enabled = bool(planner_cfg.get("enabled", False))
            if self.planner_enabled:
                try:
                    controller_cfg = planner_cfg.get("controller")
                    if controller_cfg is None:
                        controller_cfg = {k: v for k, v in planner_cfg.items() if k != "enabled"}
                    self.planner = HybridPlanner(self.knowledge_graph, self.symbolic_reasoner, controller_cfg)
                except Exception as exc:
                    self.logger.warning("Planner initialization failed: %s", exc)
                    self.planner = None
                    self.planner_enabled = False
            else:
                self.planner = None

        persistent_data = state.get("persistent_memory")
        if persistent_data and getattr(self, "persistent_memory", None) is not None:
            try:
                self.persistent_memory.from_dict(persistent_data)
            except Exception as exc:
                self.logger.debug("Failed to restore persistent memory: %s", exc)

        attention_data = state.get("attention")
        if attention_data and self.attention_manager is not None:
            manager_state = attention_data.get("manager", {})
            if isinstance(manager_state, dict):
                try:
                    self.attention_manager.from_dict(manager_state)
                except Exception as exc:
                    self.logger.debug("Failed to restore attention manager state: %s", exc)
            focus_state = attention_data.get("focus")
            if isinstance(focus_state, dict):
                self.attention_focus_state = focus_state
            else:
                self.attention_focus_state = {}
        else:
            self.attention_focus_state = {}

        self_model_data = state.get("self_model")
        if self_model_data and self.self_model is not None:
            module_state = self_model_data.get("module", {})
            if isinstance(module_state, dict):
                try:
                    self.self_model.from_dict(module_state)
                except Exception as exc:
                    self.logger.debug("Failed to restore self model state: %s", exc)
            state_snapshot = self_model_data.get("state")
            if isinstance(state_snapshot, dict):
                self.self_model_state = state_snapshot
            else:
                self.self_model_state = {}
        else:
            if self.self_model is not None:
                try:
                    self.self_model.reset()
                except Exception:
                    pass
            self.self_model_state = {}

        metacognition_data = state.get("metacognition")
        if metacognition_data and getattr(self, "metacognition", None) is not None:
            module_state = metacognition_data.get("module", {})
            if isinstance(module_state, dict):
                try:
                    self.metacognition.from_dict(module_state)
                except Exception as exc:
                    self.logger.debug("Failed to restore metacognition state: %s", exc)
            state_snapshot = metacognition_data.get("state")
            if isinstance(state_snapshot, dict):
                self.metacognition_state = state_snapshot
            else:
                self.metacognition_state = {}
        else:
            if getattr(self, "metacognition", None) is not None:
                try:
                    self.metacognition.reset()
                except Exception:
                    pass
            self.metacognition_state = {}

        emotion_motivation_data = state.get("emotion_motivation")
        if emotion_motivation_data and getattr(self, "emotion_motivation", None) is not None:
            try:
                self.emotion_motivation.from_dict(emotion_motivation_data)
            except Exception as exc:
                self.logger.debug("Failed to restore emotion motivation state: %s", exc)

        motor_state = state.get("cognitive_state", {}).get("motor", {})
        if isinstance(motor_state, dict):
            status = motor_state.get("status")
            if isinstance(status, dict):
                self.motor_status.update(status)
            self.motor_last_command = motor_state.get("last_command")

        # 更新时间
        self.current_time = state["current_time"]

        # 更新神经元状态
        neuron_container = getattr(self.network, "neurons", None)
        if hasattr(neuron_container, "items"):
            for neuron_id, neuron_state in state["network_state"]["neuron_states"].items():
                try:
                    lookup_id = int(neuron_id)
                except (TypeError, ValueError):
                    lookup_id = neuron_id
                if lookup_id in neuron_container:
                    neuron = neuron_container[lookup_id]
                    if hasattr(neuron, "_voltage"):
                        neuron._voltage = neuron_state.get("voltage")
                    elif hasattr(neuron, "_v"):
                        neuron._v = neuron_state.get("voltage")
                    elif hasattr(neuron, "voltage"):
                        try:
                            setattr(neuron, "voltage", neuron_state.get("voltage"))
                        except Exception:  # pragma: no cover - attribute may be read-only
                            pass

                    if hasattr(neuron, "_spike_history"):
                        neuron._spike_history = neuron_state.get("spike_history", [])
                    elif hasattr(neuron, "spike_history"):
                        try:
                            setattr(neuron, "spike_history", neuron_state.get("spike_history", []))
                        except Exception:  # pragma: no cover
                            pass

        synapse_container = getattr(self.network, "synapses", None)
        if synapse_container is not None:
            for synapse_key, synapse_state in state["network_state"]["synapse_states"].items():
                try:
                    pre_id, post_id = map(int, synapse_key.split("_"))
                except (TypeError, ValueError):
                    continue
                synapse_id = (pre_id, post_id)
                try:
                    synapse = synapse_container[synapse_id]
                except Exception:
                    continue
                if hasattr(synapse, "_weight"):
                    synapse._weight = synapse_state.get("weight")
                elif hasattr(synapse, "weight"):
                    try:
                        setattr(synapse, "weight", synapse_state.get("weight"))
                    except Exception:  # pragma: no cover
                        pass

        memory_state = state.get("cognitive_state", {}).get("memory", {})
        memory_obj = getattr(self, "memory", None)
        if memory_obj is not None:
            comprehensive = memory_state.get("comprehensive")
            memory_system = getattr(memory_obj, "memory_system", None)
            if isinstance(comprehensive, dict) and memory_system is not None:
                wm_snapshot = comprehensive.get("working_memory")
                try:
                    wm = getattr(memory_system, "working_memory", None)
                    if wm is not None:
                        for attr in ("_entries", "_order"):
                            bucket = getattr(wm, attr, None)
                            if hasattr(bucket, "clear"):
                                try:
                                    bucket.clear()
                                except Exception:
                                    pass
                        items_list = getattr(wm, "items", None)
                        weights_list = getattr(wm, "attention_weights", None)
                        if isinstance(items_list, list):
                            items_list.clear()
                        if isinstance(weights_list, list):
                            weights_list.clear()

                        if isinstance(wm_snapshot, dict):
                            items = wm_snapshot.get("items") or []
                            weights = wm_snapshot.get("attention_weights") or []
                            if isinstance(items, list):
                                add_item = getattr(wm, "add_item", None)
                                if callable(add_item):
                                    for idx, item in enumerate(items):
                                        weight = 1.0
                                        if isinstance(weights, list) and idx < len(weights):
                                            try:
                                                weight = float(weights[idx])
                                            except Exception:
                                                weight = 1.0
                                        try:
                                            add_item(item, weight)
                                        except TypeError:
                                            add_item(item)
                                        except Exception:
                                            continue
                except Exception as exc:  # pragma: no cover - best effort restore
                    self.logger.debug("Failed to restore working memory snapshot: %s", exc)

                episodic = comprehensive.get("episodic")
                if isinstance(episodic, list):
                    hippocampal = getattr(memory_system, "hippocampal_system", None)
                    ca1 = getattr(hippocampal, "ca1_memories", None) if hippocampal is not None else None
                    ca3 = getattr(hippocampal, "ca3_memories", None) if hippocampal is not None else None
                    dg = getattr(hippocampal, "dg_memories", None) if hippocampal is not None else None
                    for bucket in (ca1, ca3, dg):
                        if isinstance(bucket, dict):
                            bucket.clear()
                    for attr in ("ca1_ca3_connections", "dg_ca3_connections"):
                        bucket = getattr(hippocampal, attr, None) if hippocampal is not None else None
                        if isinstance(bucket, dict):
                            bucket.clear()
                    network_graph = getattr(hippocampal, "ca3_recurrent_network", None) if hippocampal is not None else None
                    if hasattr(network_graph, "clear"):
                        try:
                            network_graph.clear()
                        except Exception:
                            pass

                    for payload in episodic:
                        if not isinstance(payload, dict):
                            continue
                        content = payload.get("content")
                        context = payload.get("context")
                        if not isinstance(content, dict):
                            continue
                        if not isinstance(context, dict):
                            context = {}
                        try:
                            result = memory_obj.process(
                                {
                                    "store": {
                                        "memory_type": "EPISODIC",
                                        "content": content,
                                        "context": context,
                                    }
                                }
                            )
                        except Exception:
                            continue

                        stored_id = result.get("stored_id") if isinstance(result, dict) else None
                        if not stored_id or not isinstance(ca1, dict):
                            continue
                        trace = ca1.get(stored_id)
                        if trace is None:
                            continue
                        for key in (
                            "encoding_time",
                            "last_access_time",
                            "access_count",
                            "strength",
                            "stability",
                            "retrievability",
                            "consolidation_progress",
                        ):
                            if payload.get(key) is None or not hasattr(trace, key):
                                continue
                            try:
                                setattr(trace, key, payload.get(key))
                            except Exception:
                                continue
                        state_value = payload.get("consolidation_state")
                        if state_value and hasattr(trace, "consolidation_state"):
                            try:
                                from BrainSimulationSystem.models.memory import ConsolidationState

                                if isinstance(state_value, str):
                                    try:
                                        trace.consolidation_state = ConsolidationState(state_value)
                                    except Exception:
                                        trace.consolidation_state = ConsolidationState[str(state_value).upper()]
                            except Exception:
                                pass
            if hasattr(memory_obj, "working_memory"):
                memory_obj.working_memory = memory_state.get("working_memory", {})
            if hasattr(memory_obj, "long_term_memory"):
                memory_obj.long_term_memory = memory_state.get("long_term_memory", {})
            if hasattr(memory_obj, "memory_strengths"):
                memory_obj.memory_strengths = memory_state.get("memory_strengths", {})

        decision_state = state.get("cognitive_state", {}).get("decision", {})
        decision_obj = getattr(self, "decision", None)
        if decision_obj is not None:
            if hasattr(decision_obj, "action_values"):
                decision_obj.action_values = decision_state.get("action_values", {})
            if hasattr(decision_obj, "decision_history"):
                decision_obj.decision_history = decision_state.get("decision_history", [])
        cognitive_state = state.get("cognitive_state", {})

        emotion_state = cognitive_state.get("emotion", {})
        if isinstance(emotion_state, dict):
            self.emotion_state = emotion_state
            emotion_obj = getattr(self, "emotion_system", None)
            if emotion_obj is not None:
                emotions = emotion_state.get("emotions", {})
                if isinstance(emotions, dict):
                    stored_emotions = getattr(emotion_obj, "emotions", {})
                    if isinstance(stored_emotions, dict):
                        for name, value in emotions.items():
                            emotion_entry = stored_emotions.get(name)
                            if emotion_entry is not None and hasattr(emotion_entry, "intensity"):
                                try:
                                    emotion_entry.intensity = float(value)
                                except Exception:  # pragma: no cover - defensive fallback
                                    pass
                neuromodulators = emotion_state.get("neuromodulators", {})
                if isinstance(neuromodulators, dict):
                    attr_map = {
                        "dopamine": "dopamine_level",
                        "serotonin": "serotonin_level",
                        "norepinephrine": "norepinephrine_level",
                    }
                    for key, attr in attr_map.items():
                        if key in neuromodulators and hasattr(emotion_obj, attr):
                            try:
                                setattr(emotion_obj, attr, float(neuromodulators[key]))
                            except Exception:  # pragma: no cover - defensive fallback
                                pass
        else:
            self.emotion_state = {}

        curiosity_state = cognitive_state.get("curiosity", {})
        if isinstance(curiosity_state, dict):
            try:
                self.last_curiosity_drive = float(curiosity_state.get("last_drive", self.last_curiosity_drive))
            except (TypeError, ValueError):
                pass
        else:
            self.last_curiosity_drive = 0.0

        personality_state = cognitive_state.get("personality", {})
        if isinstance(personality_state, dict):
            self.personality_state = personality_state
            baseline = personality_state.get("baseline", {})
            if self.personality_evolver is not None and isinstance(baseline, dict):
                try:
                    self.personality_evolver.base_traits.update(baseline)
                except Exception:  # pragma: no cover - defensive fallback
                    pass
            dynamic = personality_state.get("dynamic", {})
            if self.personality_dynamics is not None and isinstance(dynamic, dict) and hasattr(self.personality_dynamics, "base_traits"):
                try:
                    self.personality_dynamics.base_traits.update(dynamic)
                except Exception:  # pragma: no cover - defensive fallback
                    pass
        else:
            self.personality_state = {
                "baseline": {},
                "dynamic": {},
                "environment": {},
            }

        if isinstance(self.personality_state, dict) and "environment" not in self.personality_state:
            self.personality_state["environment"] = {}

