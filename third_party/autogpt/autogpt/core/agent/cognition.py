"""Cognition adapters bridging :class:`SimpleAgent` and neuromorphic backends."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from pydantic import Field

from autogpt.core.brain.config import BrainBackend, BrainSimulationConfig, WholeBrainConfig
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.resource.model_providers.schema import CompletionModelFunction
from modules.brain.serving import BrainServingClient, BrainServingError
from modules.brain.state import (
    BrainCycleResult,
    CognitiveIntent,
    EmotionSnapshot,
    FeelingSnapshot,
    PersonalityProfile,
)
from modules.brain.backends import (
    BrainBackendInitError,
    BrainBackendProtocol,
    create_brain_backend,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


class BrainAdapterConfiguration(SystemConfiguration):
    """Configuration wrapper for the simple cognition adapter."""

    backend: BrainBackend = Field(default=BrainBackend.WHOLE_BRAIN)
    whole_brain: WholeBrainConfig = Field(default_factory=WholeBrainConfig)
    brain_simulation: BrainSimulationConfig = Field(default_factory=BrainSimulationConfig)


class SimpleBrainAdapterSettings(SystemSettings):
    configuration: BrainAdapterConfiguration


class SimpleBrainAdapter(Configurable):
    """Drive ``SimpleAgent`` cognition via :class:`WholeBrainSimulation`."""

    default_settings = SimpleBrainAdapterSettings(
        name="simple_brain_adapter",
        description=(
            "Routes planning and action selection through the WholeBrain "
            "neuromorphic simulation."
        ),
        configuration=BrainAdapterConfiguration(),
    )

    #: Mapping from cognitive intention to preferred ability names.
    _INTENTION_ABILITY_PREFERENCES: Mapping[str, tuple[str, ...]] = {
        "observe": ("self_assess", "lint_code", "run_tests"),
        "explore": ("run_tests", "self_assess", "create_new_ability"),
        "approach": ("run_tests", "write_file", "generate_tests"),
        "withdraw": ("self_assess", "evaluate_metrics", "lint_code"),
    }

    def __init__(
        self,
        settings: SimpleBrainAdapterSettings,
        logger: logging.Logger,
        event_bus: Any | None = None,
    ) -> None:
        self._configuration = settings.configuration
        self._logger = logger
        self._event_bus = event_bus
        self._serving_config = getattr(self._configuration.whole_brain, "serving", None)
        self._serving_client: BrainServingClient | None = None
        self._serving_enabled = bool(getattr(self._serving_config, "enabled", False))
        self._serving_fallback = bool(
            getattr(self._serving_config, "fallback_to_local", True)
            if self._serving_config
            else True
        )
        self._serving_service_id: str | None = None
        if self._serving_enabled and self._serving_config is not None:
            try:
                self._serving_client = BrainServingClient.from_config(
                    self._serving_config,
                    logger_obj=self._logger,
                )
            except Exception as exc:  # pragma: no cover - defensive initialisation
                self._logger.warning(
                    "Brain serving initialisation failed; reverting to local brain: %s",
                    exc,
                )
                self._serving_client = None
                self._serving_enabled = False
            else:
                self._logger.info(
                    "Brain serving enabled (protocol=%s, endpoint=%s, model=%s).",
                    getattr(self._serving_config, "protocol", "http"),
                    getattr(self._serving_config, "endpoint", "remote"),
                    getattr(self._serving_config, "model_name", "transformer-brain"),
                )
                self._register_serving_service()
        backend_choice = getattr(self._configuration, "backend", BrainBackend.WHOLE_BRAIN)
        self._brain: BrainBackendProtocol
        try:
            self._brain = create_brain_backend(
                backend_choice,
                whole_brain_config=self._configuration.whole_brain,
                brain_simulation_config=self._configuration.brain_simulation,
            )
        except BrainBackendInitError as exc:
            self._logger.warning(
                "Falling back to WholeBrain backend for SimpleBrainAdapter (%s).",
                exc,
            )
            self._brain = create_brain_backend(
                BrainBackend.WHOLE_BRAIN,
                whole_brain_config=self._configuration.whole_brain,
                brain_simulation_config=self._configuration.brain_simulation,
            )
        self._last_cycle: BrainCycleResult | None = None
        self._last_remote_job_id: str | None = None

    def _process_cycle(
        self,
        input_payload: Mapping[str, Any],
        *,
        callsite: str,
        context: Mapping[str, Any] | None = None,
    ) -> BrainCycleResult:
        if not self._serving_client:
            return self._brain.process_cycle(dict(input_payload))

        effective_context: Dict[str, Any] = {"callsite": callsite}
        if context:
            effective_context.update({str(k): v for k, v in context.items()})

        try:
            response = self._serving_client.infer(
                dict(input_payload),
                context=effective_context,
            )
        except BrainServingError as exc:
            self._emit_serving_metrics({"errors": 1.0}, "error", {"reason": str(exc)})
            if self._serving_fallback:
                self._logger.warning(
                    "Remote brain serving failed (%s); using local simulation fallback.",
                    exc,
                )
                return self._brain.process_cycle(dict(input_payload))
            raise

        if response.job_id:
            self._last_remote_job_id = response.job_id
        if response.cycle is not None:
            self._emit_serving_metrics(
                response.metrics or {},
                response.status,
                {"job_id": response.job_id} if response.job_id else None,
            )
            return response.cycle

        self._logger.info(
            "Remote brain serving returned status '%s' (job_id=%s).",
            response.status,
            response.job_id or "n/a",
        )
        self._emit_serving_metrics(
            response.metrics or {},
            response.status,
            {"job_id": response.job_id} if response.job_id else None,
        )
        if self._serving_fallback:
            self._logger.debug(
                "Falling back to local WholeBrainSimulation due to missing remote result."
            )
            return self._brain.process_cycle(dict(input_payload))
        raise BrainServingError(
            f"Remote brain serving completed with status '{response.status}' without a cycle result."
        )

    def _register_serving_service(self) -> None:
        if self._serving_service_id or self._serving_config is None:
            return
        try:
            from modules.environment import register_model_service
        except Exception:
            return
        try:
            descriptor = register_model_service(
                getattr(self._serving_config, "model_name", "transformer-brain"),
                self._serving_config.rpc_config(),
                metadata={"source": "SimpleBrainAdapter"},
                event_bus=self._event_bus,
            )
        except Exception:
            self._logger.debug("Model environment registration failed.", exc_info=True)
            descriptor = None
        if descriptor is not None:
            self._serving_service_id = descriptor.service_id

    def _emit_serving_metrics(
        self,
        metrics: Dict[str, Any],
        status: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._serving_service_id:
            return
        numeric_metrics = {
            str(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        if not numeric_metrics:
            numeric_metrics = {"status_indicator": 1.0}
        metadata = {"status": status}
        if extra_metadata:
            metadata.update({str(k): v for k, v in extra_metadata.items()})
        try:
            from modules.environment import report_service_signal

            report_service_signal(
                self._serving_service_id,
                numeric_metrics,
                metadata=metadata,
                event_bus=self._event_bus,
            )
        except Exception:
            self._logger.debug(
                "Failed to emit service metrics for %s",
                self._serving_service_id,
                exc_info=True,
            )

    def bind_event_bus(self, event_bus: Any | None) -> None:
        """Attach an event bus used for telemetry publishing."""

        self._event_bus = event_bus
        if self._serving_service_id is None:
            self._register_serving_service()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def build_initial_plan(
        self,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        ability_specs: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a bootstrap plan from the current brain state."""

        input_payload = self._compose_cycle_input(
            agent_name=agent_name,
            agent_role=agent_role,
            agent_goals=agent_goals,
            abilities=ability_specs,
        )
        context = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "cycle": "initial_plan",
            "goals": list(agent_goals),
        }
        brain_result = self._process_cycle(
            input_payload,
            callsite="build_initial_plan",
            context=context,
        )
        self._last_cycle = brain_result
        metadata = self._summarise_cycle(brain_result)

        plan_steps = brain_result.intent.plan or ["clarify_objective"]
        plan_dict = {
            "task_list": self._plan_steps_to_tasks(plan_steps, agent_goals),
            "backend": "whole_brain",
            "intention": brain_result.intent.intention,
            "confidence": float(brain_result.intent.confidence),
            "thoughts": metadata,
        }
        return plan_dict, metadata

    async def determine_next_ability(
        self,
        *,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        task: Any | None,
        ability_specs: list[CompletionModelFunction],
        cycle_index: int,
        backlog_size: int,
        completed: int,
        state_context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Select the next ability using the neuromorphic backend."""

        input_payload = self._compose_cycle_input(
            agent_name=agent_name,
            agent_role=agent_role,
            agent_goals=agent_goals,
            task=task,
            abilities=[spec.name for spec in ability_specs],
            cycle_index=cycle_index,
            backlog_size=backlog_size,
            completed=completed,
            state_context=state_context,
        )
        context = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "cycle": "determine_next_ability",
            "cycle_index": cycle_index,
            "backlog_size": backlog_size,
            "completed": completed,
        }
        brain_result = self._process_cycle(
            input_payload,
            callsite="determine_next_ability",
            context=context,
        )
        self._last_cycle = brain_result
        metadata = self._summarise_cycle(brain_result)

        ability_name, ability_args = self._select_ability(
            brain_result.intent,
            ability_specs,
        )
        payload = {
            "next_ability": ability_name,
            "ability_arguments": ability_args,
            "backend": "whole_brain",
            "confidence": float(brain_result.intent.confidence),
            "plan": list(brain_result.intent.plan),
            "reasoning": metadata.get("analysis"),
        }
        return payload, metadata

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compose_cycle_input(
        self,
        *,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        abilities: Iterable[str],
        task: Any | None = None,
        cycle_index: int = 0,
        backlog_size: int = 0,
        completed: int = 0,
        state_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare structured inputs for :meth:`WholeBrainSimulation.process_cycle`."""

        state_context = state_context or {}
        ability_list = list(abilities)

        text_fragments: list[str] = [f"Role: {agent_role}"]
        if agent_goals:
            text_fragments.append("Goals: " + "; ".join(agent_goals[:3]))
        if ability_list:
            text_fragments.append("Abilities: " + ", ".join(ability_list))

        context = {
            "cycle_count": float(max(0, cycle_index)),
            "backlog": float(max(0, backlog_size)),
            "completed": float(max(0, completed)),
            "ability_count": float(len(ability_list)),
        }
        context.update(
            {
                f"state_{key}": _safe_float(value)
                for key, value in state_context.items()
                if isinstance(value, (int, float))
            }
        )

        vision = [
            min(1.0, context["cycle_count"] / 10.0),
            min(1.0, context["backlog"] / 10.0),
            min(1.0, context["completed"] / (backlog_size + completed + 1 or 1)),
        ]
        auditory = [
            min(1.0, len(agent_goals) / 5.0),
            min(1.0, len(ability_list) / 5.0),
            min(1.0, backlog_size / 5.0),
        ]
        somatosensory = [
            min(1.0, cycle_index / 8.0),
            min(1.0, backlog_size / 8.0),
            1.0 if state_context.get("enough_info") else 0.0,
        ]

        if task is not None:
            description = getattr(task, "objective", str(task))
            text_fragments.append(f"Task: {description}")
            context["task_priority"] = _safe_float(getattr(task, "priority", 0))
            context["task_cycles"] = _safe_float(
                getattr(getattr(task, "context", None), "cycle_count", 0)
            )
            ready = getattr(task, "ready_criteria", []) or []
            acceptance = getattr(task, "acceptance_criteria", []) or []
            if ready:
                text_fragments.append("Ready: " + "; ".join(map(str, ready)))
            if acceptance:
                text_fragments.append("Done when: " + "; ".join(map(str, acceptance)))

        return {
            "agent_id": agent_name,
            "text": "\n".join(text_fragments),
            "context": context,
            "vision": vision,
            "auditory": auditory,
            "somatosensory": somatosensory,
            "is_salient": bool(backlog_size > 0 and cycle_index > 0),
        }

    def _plan_steps_to_tasks(
        self, steps: Iterable[str], agent_goals: list[str]
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for index, raw_step in enumerate(steps, start=1):
            step = str(raw_step)
            normalized = step.replace("_", " ").replace("-", " ").strip()
            objective = normalized.capitalize() or "Clarify objective"
            task_type = self._infer_task_type(step)
            ready_hint = f"Outline how to {normalized.lower()}"
            acceptance_hint = f"Summarise the outcome of {normalized.lower()}"
            tasks.append(
                {
                    "objective": objective,
                    "type": task_type,
                    "priority": index,
                    "ready_criteria": [ready_hint],
                    "acceptance_criteria": [acceptance_hint],
                }
            )
        if not tasks:  # pragma: no cover - defensive
            tasks.append(
                {
                    "objective": "Review goals",
                    "type": "plan",
                    "priority": 1,
                    "ready_criteria": ["List current goals"],
                    "acceptance_criteria": [
                        "Document a concrete action supporting the primary goal"
                    ],
                }
            )
        if agent_goals:
            tasks[0]["ready_criteria"].append(
                f"Ensure alignment with goal: {agent_goals[0]}"
            )
        return tasks

    def _infer_task_type(self, step: str) -> str:
        lowered = step.lower()
        if any(keyword in lowered for keyword in ("write", "engage", "create")):
            return "write"
        if any(keyword in lowered for keyword in ("test", "verify", "run")):
            return "test"
        if any(keyword in lowered for keyword in ("code", "implement", "fix")):
            return "code"
        if any(keyword in lowered for keyword in ("scan", "assess", "observe", "review")):
            return "research"
        return "plan"

    def _select_ability(
        self,
        intent: CognitiveIntent,
        ability_specs: list[CompletionModelFunction],
    ) -> Tuple[str, dict[str, Any]]:
        ability_by_name = {spec.name: spec for spec in ability_specs}
        preference = self._INTENTION_ABILITY_PREFERENCES.get(
            intent.intention, ("self_assess",)
        )

        for name in preference:
            spec = ability_by_name.get(name)
            if spec and self._callable_without_required_args(spec):
                return name, {}

        for spec in ability_specs:
            if self._callable_without_required_args(spec):
                return spec.name, {}

        if ability_specs:
            self._logger.debug(
                "No ability without required arguments available; returning '%s'",
                ability_specs[0].name,
            )
            return ability_specs[0].name, {}

        return "self_assess", {}

    def _callable_without_required_args(self, spec: CompletionModelFunction) -> bool:
        if not spec.parameters:
            return True
        return not any(param.required for param in spec.parameters.values())

    def _summarise_cycle(self, result: BrainCycleResult) -> dict[str, Any]:
        def _emotion_payload(emotion: EmotionSnapshot) -> dict[str, float | str]:
            return {
                "primary": getattr(emotion.primary, "value", str(emotion.primary)),
                "intensity": float(emotion.intensity),
                "mood": float(emotion.mood),
                "dimensions": {k: float(v) for k, v in emotion.dimensions.items()},
                "context": {k: float(v) for k, v in emotion.context.items()},
                "decay": float(emotion.decay),
            }

        payload = {
            "backend": "whole_brain",
            "intention": result.intent.intention,
            "plan": list(result.intent.plan),
            "confidence": float(result.intent.confidence),
            "weights": {k: float(v) for k, v in result.intent.weights.items()},
            "tags": list(result.intent.tags),
            "analysis": "; ".join(result.intent.plan)
            if result.intent.plan
            else result.intent.intention,
            "emotion": _emotion_payload(result.emotion),
            "curiosity": asdict(result.curiosity),
            "personality": asdict(result.personality),
            "metrics": {k: float(v) for k, v in (result.metrics or {}).items()},
            "metadata": {k: v for k, v in (result.metadata or {}).items() if v is not None},
        }
        if result.thoughts:
            payload["thoughts"] = {
                "focus": result.thoughts.focus,
                "summary": result.thoughts.summary,
                "plan": list(result.thoughts.plan),
                "tags": list(result.thoughts.tags),
            }
        if result.feeling:
            payload["feeling"] = {
                "descriptor": result.feeling.descriptor,
                "valence": float(result.feeling.valence),
                "arousal": float(result.feeling.arousal),
                "mood": float(result.feeling.mood),
                "confidence": float(result.feeling.confidence),
                "context_tags": list(result.feeling.context_tags),
            }
        return payload


__all__ = [
    "SimpleBrainAdapter",
    "SimpleBrainAdapterSettings",
    "BrainAdapterConfiguration",
]
