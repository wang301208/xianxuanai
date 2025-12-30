from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Callable, Sequence, Mapping
from uuid import uuid4


from auto_gpt_plugin_template import AutoGPTPluginTemplate
from events import EventBus, create_event_bus
from events.client import EventClient
from events.coordination import TaskStatus, TaskStatusEvent
from forge.sdk.model import Task
from pydantic import Field, validator

if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.core.prompting.base import PromptStrategy
    from autogpt.core.resource.model_providers.schema import (
        AssistantChatMessage,
        ChatModelInfo,
        ChatModelProvider,
        ChatModelResponse,
    )
    from autogpt.models.command_registry import CommandRegistry
    from knowledge import UnifiedKnowledgeBase
    from reasoning import DecisionEngine
from reasoning.decision_engine import ActionDirective

from autogpt.agents.utils.prompt_scratchpad import PromptScratchpad
from autogpt.config import ConfigBuilder
from autogpt.config.ai_directives import AIDirectives
from autogpt.config.ai_profile import AIProfile
from autogpt.core.brain.config import (
    BrainBackend,
    BrainSimulationConfig,
    TransformerBrainConfig,
    WholeBrainConfig,
)
from autogpt.core.brain.transformer_brain import TransformerBrain
from autogpt.core.brain.encoding import build_brain_inputs
from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.prompting.schema import (
    ChatMessage,
   ChatPrompt,
   CompletionModelFunction,
)
from autogpt.core.resource.model_providers.schema import ChatModelResponse
from autogpt.core.resource.model_providers.openai import (
    OPEN_AI_CHAT_MODELS,
    OpenAIModelName,
)
from autogpt.core.runner.client_lib.logging.helpers import dump_prompt
from autogpt.file_storage.base import FileStorage
from autogpt.llm.providers.openai import get_openai_command_specs
from autogpt.models.action_history import (
    Action,
    ActionHistoryConfiguration,
    ActionResult,
    EpisodicActionHistory,
)
from autogpt.models.context_item import StaticContextItem
from .features.context import get_agent_context
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT

from modules.brain.adapter import WholeBrainAgentAdapter
from modules.brain.backends import BrainBackendInitError, create_brain_backend
from backend.knowledge.registry import get_graph_store_instance
from modules.knowledge import ActionGuard, collect_knowledge_context_async
from monitoring import WorkspaceMessage, global_workspace
from backend.memory import LongTermMemory, WorkingMemory

logger = logging.getLogger(__name__)

CommandName = str
CommandArgs = dict[str, str]
AgentThoughts = dict[str, Any]


class BaseAgentConfiguration(SystemConfiguration):
    allow_fs_access: bool = UserConfigurable(default=False)

    fast_llm: OpenAIModelName = UserConfigurable(default=OpenAIModelName.GPT3_16k)
    smart_llm: OpenAIModelName = UserConfigurable(default=OpenAIModelName.GPT4)
    use_functions_api: bool = UserConfigurable(default=False)

    default_cycle_instruction: str = DEFAULT_TRIGGERING_PROMPT
    """The default instruction passed to the AI for a thinking cycle."""

    big_brain: bool = UserConfigurable(default=True)
    """
    Whether this agent uses the configured smart LLM (default) to think,
    as opposed to the configured fast LLM. Enabling this disables hybrid mode.
    """

    brain: TransformerBrainConfig = Field(default_factory=TransformerBrainConfig)
    """Configuration for the optional Transformer-based brain module."""

    use_transformer_brain: bool = UserConfigurable(default=False)
    """Whether to initialize the ``TransformerBrain`` component."""

    brain_backend: BrainBackend = UserConfigurable(
        default=BrainBackend.WHOLE_BRAIN
    )
    """Selects the cognitive backend used when ``big_brain`` is enabled."""

    whole_brain: WholeBrainConfig = Field(default_factory=WholeBrainConfig)
    """Configuration for the biologically inspired whole-brain simulation."""

    brain_simulation: BrainSimulationConfig = Field(default_factory=BrainSimulationConfig)
    """Configuration for the BrainSimulationSystem-backed adapter."""

    use_knowledge_base: bool = UserConfigurable(default=False)
    """Whether to attach a ``UnifiedKnowledgeBase`` instance."""

    use_decision_engine: bool = UserConfigurable(default=False)
    """Whether to provide a ``DecisionEngine`` for reasoning."""

    cycle_budget: Optional[int] = 1
    """
    The number of cycles that the agent is allowed to run unsupervised.

    `None` for unlimited continuous execution,
    `1` to require user approval for every step,
    `0` to stop the agent.
    """

    cycles_remaining = cycle_budget

    knowledge_context_enabled: bool = UserConfigurable(default=True)
    """Toggle automatic prompt augmentation with knowledge graph context."""

    knowledge_context_top_k: int = UserConfigurable(default=5)
    """Maximum number of knowledge snippets to surface within the working prompt."""

    knowledge_context_relation_limit: int = UserConfigurable(default=3)
    """Maximum number of relations per concept to include."""
    """The number of cycles remaining within the `cycle_budget`."""

    cycle_count = 0
    """The number of cycles that the agent has run since its initialization."""

    send_token_limit: Optional[int] = None
    """
    The token limit for prompt construction. Should leave room for the completion;
    defaults to 75% of `llm.max_tokens`.
    """

    history: ActionHistoryConfiguration = Field(
        default_factory=ActionHistoryConfiguration
    )
    """Configuration for action history handling."""

    working_memory_capacity: int = UserConfigurable(default=20)
    """Maximum number of recent context entries retained in working memory."""

    metacognitive_review_enabled: bool = UserConfigurable(default=True)
    """Enable internal self-review between draft and final reasoning outputs."""

    metacognitive_review_require_revision: bool = UserConfigurable(default=True)
    """Force a revision call when the reviewer flags blocking issues."""

    metacognitive_review_max_issue_notes: int = UserConfigurable(default=5)
    """Upper bound on how many review issues are threaded back into the revision prompt."""

    plugins: list[AutoGPTPluginTemplate] = Field(default_factory=list, exclude=True)

    class Config:
        arbitrary_types_allowed = True  # Necessary for plugins

    @validator("plugins", each_item=True)
    def validate_plugins(cls, p: AutoGPTPluginTemplate | Any):
        assert issubclass(
            p.__class__, AutoGPTPluginTemplate
        ), f"{p} does not subclass AutoGPTPluginTemplate"
        assert (
            p.__class__.__name__ != "AutoGPTPluginTemplate"
        ), f"Plugins must subclass AutoGPTPluginTemplate; {p} is a template instance"
        return p

    @validator("use_functions_api")
    def validate_openai_functions(cls, v: bool, values: dict[str, Any]):
        if v:
            smart_llm = values["smart_llm"]
            fast_llm = values["fast_llm"]
            assert all(
                [
                    not any(s in name for s in {"-0301", "-0314"})
                    for name in {smart_llm, fast_llm}
                ]
            ), (
                f"Model {smart_llm} does not support OpenAI Functions. "
                "Please disable OPENAI_FUNCTIONS or choose a suitable model."
            )
        return v

    @validator("brain_backend", pre=True, always=True)
    def _ensure_brain_backend(cls, v: Any, values: dict[str, Any]):
        if isinstance(v, BrainBackend):
            return v
        if v:
            try:
                return BrainBackend(str(v).lower())
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Unsupported brain backend '{v}'") from exc
        if values.get("use_transformer_brain"):
            return BrainBackend.TRANSFORMER
        return BrainBackend.WHOLE_BRAIN

    @validator("use_transformer_brain", always=True)
    def _sync_transformer_flag(cls, v: bool, values: dict[str, Any]) -> bool:
        backend = values.get("brain_backend")
        if isinstance(backend, BrainBackend) and backend != BrainBackend.TRANSFORMER:
            return False
        return bool(v)


class BaseAgentSettings(SystemSettings):
    agent_id: str = ""
    workspace_id: str | None = None

    ai_profile: AIProfile = Field(default_factory=lambda: AIProfile(ai_name="AutoGPT"))
    """The AI profile or "personality" of the agent."""

    directives: AIDirectives = Field(
        default_factory=lambda: AIDirectives.from_file(
            ConfigBuilder.default_settings.prompt_settings_file
        )
    )
    """Directives (general instructional guidelines) for the agent."""

    task: Task = Field(
        default_factory=lambda: Task(
            input="Terminate immediately",
            additional_input=None,
            created_at=datetime.now(),
            modified_at=datetime.now(),
            task_id=str(uuid4()),
            artifacts=[],
        ),
        description="The user-given task that the agent is working on.",
    )

    config: BaseAgentConfiguration = Field(default_factory=BaseAgentConfiguration)
    """The configuration for this BaseAgent subsystem instance."""

    history: EpisodicActionHistory = Field(default_factory=EpisodicActionHistory)
    """(STATE) The action history of the agent."""

    working_memory_items: list[str] = Field(default_factory=list)
    """(STATE) Rolling buffer of recent context stored in working memory."""

    long_term_memory_path: str | None = None
    """(STATE) Relative path to the agent's long-term memory database."""


class BaseAgent(Configurable[BaseAgentSettings], ABC):
    """Base class for all AutoGPT agent classes."""

    ThoughtProcessOutput = tuple[CommandName, CommandArgs, AgentThoughts]

    default_settings = BaseAgentSettings(
        name="BaseAgent",
        description=__doc__,
    )

    def __init__(
        self,
        settings: BaseAgentSettings,
        llm_provider: ChatModelProvider,
        prompt_strategy: PromptStrategy,
        command_registry: CommandRegistry,
        file_storage: FileStorage,
        legacy_config: Config,
        event_bus: EventBus | None = None,
        brain: TransformerBrain | None = None,
        whole_brain: Any | None = None,
        knowledge_base: "UnifiedKnowledgeBase" | None = None,
        decision_engine: "DecisionEngine" | None = None,
    ):
        self.state = settings
        self.config = settings.config
        self.ai_profile = settings.ai_profile
        self.directives = settings.directives
        self.event_history = settings.history

        bus = event_bus or create_event_bus(
            legacy_config.event_bus_backend,
            host=legacy_config.event_bus_redis_host,
            port=legacy_config.event_bus_redis_port,
            password=legacy_config.event_bus_redis_password or None,
        )
        self.event_client = EventClient(bus)
        self.event_bus = bus

        self.legacy_config = legacy_config
        """LEGACY: Monolithic application configuration."""

        cognition = getattr(self, "_cognition", None)
        if cognition is not None and hasattr(cognition, "bind_event_bus"):
            try:
                cognition.bind_event_bus(self.event_bus)
            except Exception:
                logger.debug("Unable to bind event bus to cognition module.", exc_info=True)

        self.llm_provider = llm_provider

        self.prompt_strategy = prompt_strategy

        self.command_registry = command_registry
        """The registry containing all commands available to the agent."""

        self._prompt_scratchpad: PromptScratchpad | None = None

        # Optional cognitive modules
        self.brain_backend = self.config.brain_backend
        self.brain: TransformerBrain | None = brain
        self.whole_brain: Any | None = whole_brain
        self._whole_brain_adapter: WholeBrainAgentAdapter | None = None
        self._action_guard = ActionGuard()
        self._async_executor = None
        self._active_background_tasks: list[asyncio.Task[Any]] = []
        self._resolve_async_executor()
        self._workspace_cursor: int | None = None
        self._last_error_context: dict[str, Any] | None = None

        if self.brain_backend == BrainBackend.TRANSFORMER:
            if self.brain is None and self.config.use_transformer_brain:
                self.brain = TransformerBrain(self.config.brain)
        else:
            self.brain = None

        structured_backend = self.brain_backend in (
            BrainBackend.WHOLE_BRAIN,
            BrainBackend.BRAIN_SIMULATION,
        )
        if structured_backend:
            if self.whole_brain is None:
                try:
                    self.whole_brain = create_brain_backend(
                        self.brain_backend,
                        whole_brain_config=self.config.whole_brain,
                        brain_simulation_config=self.config.brain_simulation,
                    )
                except BrainBackendInitError as exc:
                    logger.warning(
                        "Failed to initialise %s backend: %s",
                        self.brain_backend.value,
                        exc,
                    )
                    self.whole_brain = None
            else:
                if self.brain_backend == BrainBackend.WHOLE_BRAIN:
                    brain_kwargs = self.config.whole_brain.to_simulation_kwargs()
                    runtime = brain_kwargs.get("config")
                    if runtime is not None and hasattr(
                        self.whole_brain, "update_config"
                    ):
                        try:
                            self.whole_brain.update_config(runtime)
                        except Exception:
                            logger.debug(
                                "Brain backend runtime update failed.", exc_info=True
                            )
                    for attr in (
                        "neuromorphic_encoding",
                        "encoding_steps",
                        "encoding_time_scale",
                        "max_neurons",
                        "max_cache_size",
                    ):
                        if attr in brain_kwargs and hasattr(self.whole_brain, attr):
                            setattr(self.whole_brain, attr, brain_kwargs[attr])
                elif self.brain_backend == BrainBackend.BRAIN_SIMULATION:
                    overrides = self.config.brain_simulation.resolved_overrides()
                    if overrides and hasattr(self.whole_brain, "update_config"):
                        try:
                            self.whole_brain.update_config(overrides=overrides)
                        except Exception:
                            logger.debug(
                                "Brain backend override update failed.",
                                exc_info=True,
                            )
            if self.whole_brain is not None:
                self._whole_brain_adapter = WholeBrainAgentAdapter(self)
                self.config.big_brain = True
            else:
                self._whole_brain_adapter = None
        else:
            self.whole_brain = None
            self._whole_brain_adapter = None
            self.config.big_brain = False
        if self.whole_brain is None:
            self.config.big_brain = False

        self.knowledge_base = knowledge_base
        self.decision_engine = decision_engine

        # Reflect presence of optional modules in configuration
        self.config.brain_backend = self.brain_backend
        self.config.use_transformer_brain = (
            self.brain is not None and self.brain_backend == BrainBackend.TRANSFORMER
        )
        self.config.use_knowledge_base = self.knowledge_base is not None
        self.config.use_decision_engine = self.decision_engine is not None

        if self.whole_brain is not None and self.knowledge_base is not None:
            attach = getattr(self.whole_brain, "attach_knowledge_base", None)
            if callable(attach):
                try:
                    attach(self.knowledge_base)
                except Exception:
                    logger.debug(
                        "Failed to attach knowledge base via brain backend hook.",
                        exc_info=True,
                    )
            elif hasattr(self.whole_brain, "knowledge_base"):
                try:
                    self.whole_brain.knowledge_base = self.knowledge_base
                except Exception:
                    logger.debug(
                        "Failed to assign knowledge base attribute on brain backend.",
                        exc_info=True,
                    )

        self._brain_pending_interaction: dict[str, Any] | None = None
        self._pending_directive: ActionDirective | None = None

        # Support multi-inheritance and mixins for subclasses
        super(BaseAgent, self).__init__()
        logger.debug(f"Created {__class__} '{self.ai_profile.ai_name}'")

    def _resolve_async_executor(self):
        if self._async_executor is not None:
            return self._async_executor

        try:
            executor_state = global_workspace.state("async.executor")
        except Exception:
            executor_state = None

        candidate = None
        if executor_state is not None:
            if hasattr(executor_state, "submit"):
                candidate = executor_state
            elif isinstance(executor_state, dict):
                candidate = executor_state.get("executor")

        if candidate is not None and hasattr(candidate, "submit"):
            self._async_executor = candidate
        return self._async_executor

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------
    def heartbeat(self) -> None:
        """Emit a heartbeat event for this agent."""
        try:
            self.event_bus.publish(
                "agent.heartbeat",
                {"agent": self.state.agent_id, "time": time.time()},
            )
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to publish heartbeat", exc_info=True)
    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------
    def report_status(
        self,
        task_id: str,
        status: TaskStatus,
        detail: str | None = None,
        *,
        summary: str | None = None,
        knowledge_statements: Optional[List[str]] = None,
        knowledge_facts: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report the current status of a task to the coordinator."""

        metadata = dict(metadata or {})
        if status == TaskStatus.FAILED and self._last_error_context:
            context = dict(self._last_error_context)
            error_reason = context.get("error_reason")
            if not detail and error_reason:
                detail = error_reason
            if error_reason and "error_reason" not in metadata:
                metadata["error_reason"] = error_reason
            failing_command = context.get("command")
            if failing_command and "failing_command" not in metadata:
                metadata["failing_command"] = failing_command
            command_args = context.get("command_args")
            if command_args is not None and "failing_command_args" not in metadata:
                metadata["failing_command_args"] = command_args
            error_info = context.get("error")
            if error_info is not None and "error" not in metadata:
                metadata["error"] = error_info
            metadata.setdefault("error_context", context)

        event = TaskStatusEvent(
            agent_id=self.state.agent_id,
            task_id=task_id,
            status=status,
            detail=detail,
            summary=summary,
            knowledge_statements=knowledge_statements,
            knowledge_facts=knowledge_facts,
            metadata=metadata or None,
        )
        self.event_client.publish("agent.status", event.to_dict())

    # ------------------------------------------------------------------
    # Decision coordination
    # ------------------------------------------------------------------
    def queue_directive(self, directive: ActionDirective) -> None:
        """Schedule *directive* to be applied before the next command executes."""

        self._pending_directive = directive

    def before_execute(
        self,
        command_name: str,
        command_args: dict[str, str],
        agent_thoughts: AgentThoughts | None,
    ) -> ActionDirective:
        """Allow supervisory components to review the next action."""

        pending = self._pending_directive
        self._pending_directive = None

        review: ActionDirective | None = None
        context: dict[str, Any] = {
            "cycle": self.config.cycle_count,
            "goals": list(self.ai_profile.ai_goals or []),
            "thoughts": agent_thoughts,
            "original_command": command_name,
        }
        if self.decision_engine is not None:
            try:
                review = self.decision_engine.review_action(
                    self.state.agent_id,
                    command_name,
                    command_args,
                    context=context,
                )
            except Exception:
                logger.debug("Decision engine review failed", exc_info=True)
                review = None

        if review is not None:
            review = review.resolve(command_name, command_args)

        if pending is not None:
            directive = pending.resolve(command_name, command_args)
            if (
                review is not None
                and directive.approved
                and not directive.requires_replan
            ):
                metadata: dict[str, Any] = {}
                if directive.metadata:
                    metadata.update(directive.metadata)
                if review.metadata:
                    metadata.update(review.metadata)
                directive = directive.copy_with(
                    command_name=review.command_name,
                    command_args=review.command_args,
                    rationale=review.rationale or directive.rationale,
                    requires_replan=directive.requires_replan or review.requires_replan,
                    metadata=metadata or None,
                )
        elif review is not None:
            directive = review
        else:
            directive = ActionDirective.approve(command_name, command_args)

        payload = {
            "agent": self.state.agent_id,
            "command": directive.command_name,
            "approved": directive.approved,
            "requires_replan": directive.requires_replan,
        }
        if directive.rationale:
            payload["rationale"] = directive.rationale
        if directive.metadata:
            payload["metadata"] = directive.metadata
        payload["directive"] = directive.to_dict()
        try:
            self.event_client.publish("agent.action.directive", payload)
        except Exception:
            logger.debug("Failed to publish directive event", exc_info=True)

        return directive

    def after_execute(
        self, directive: ActionDirective, result: ActionResult
    ) -> None:
        """Notify supervisory systems of the command outcome."""

        if self.decision_engine is not None:
            try:
                self.decision_engine.record_outcome(
                    self.state.agent_id,
                    directive.command_name or "",
                    result,
                    metadata={
                        "directive": directive.to_dict(),
                        "cycle": self.config.cycle_count,
                    },
                )
            except Exception:
                logger.debug("Decision engine outcome recording failed", exc_info=True)

        payload = {
            "agent": self.state.agent_id,
            "command": directive.command_name,
            "status": getattr(result, "status", "unknown"),
            "approved": directive.approved,
            "requires_replan": directive.requires_replan,
        }
        if directive.rationale:
            payload["rationale"] = directive.rationale
        if directive.metadata:
            payload["metadata"] = directive.metadata
        payload["directive"] = directive.to_dict()

        if getattr(result, "status", "") == "error":
            error_context: dict[str, Any] = {
                "command": directive.command_name,
                "error_reason": getattr(result, "reason", None),
            }
            if directive.command_args:
                serialized_args = dict(directive.command_args)
                payload["command_args"] = serialized_args
                error_context["command_args"] = serialized_args
            if getattr(result, "reason", None) is not None:
                payload["error_reason"] = result.reason
            error_info = getattr(result, "error", None)
            if error_info is not None:
                serialized_error = error_info.dict()
                payload["error"] = serialized_error
                error_context["error"] = serialized_error
            if error_context.get("command") is None:
                error_context.pop("command")
            if error_context.get("error_reason") is None:
                error_context.pop("error_reason")
            if error_context.get("error") is None:
                error_context.pop("error")
            self._last_error_context = error_context
        else:
            self._last_error_context = None
        try:
            self.event_client.publish("agent.action.outcome", payload)
        except Exception:
            logger.debug("Failed to publish action outcome", exc_info=True)

        self._record_action_outcome_to_workspace(directive, result)

    @property
    def llm(self) -> ChatModelInfo:
        """The LLM that the agent uses to think."""
        llm_name = (
            self.config.smart_llm if self.config.big_brain else self.config.fast_llm
        )
        return OPEN_AI_CHAT_MODELS[llm_name]

    @property
    def send_token_limit(self) -> int:
        return self.config.send_token_limit or self.llm.max_tokens * 3 // 4

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    def persist_working_context(
        self, content: str, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        """Store ``content`` in working memory and update serialized state."""

        try:
            self.working_memory.store(content, metadata=metadata)
            # Keep state in sync for persistence between steps
            self.state.working_memory_items = self.working_memory.retrieve()
        except Exception:
            logger.debug("Failed to persist working memory entry", exc_info=True)

    def archive_long_term_summary(
        self, summary: str, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        """Store ``summary`` in long-term memory."""

        try:
            self.long_term_memory.store(summary, metadata=metadata)
        except Exception:
            logger.debug("Failed to archive long-term memory entry", exc_info=True)

    def search_recent_context(
        self, query: str, *, limit: int = 3
    ) -> dict[str, list[Any]]:
        """Search working and long-term memory for ``query``."""

        working_hits = list(self.working_memory.search(query, limit=limit))
        long_term_hits = list(self.long_term_memory.search(query, limit=limit))
        return {"working": working_hits, "long_term": long_term_hits}

    async def propose_action(self) -> ThoughtProcessOutput:
        """Proposes the next action to execute, based on the task and current state.

        Emits start, end and error events around the thinking cycle, including simple
        timing metrics.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """

        cycle = self.config.cycle_count + 1
        self.event_client.publish(
            "agent.cycle.start", {"agent": self.state.agent_id, "cycle": cycle}
        )
        start_time = perf_counter()
        try:
            # Scratchpad as surrogate PromptGenerator for plugin hooks
            self._prompt_scratchpad = PromptScratchpad()
            self._brain_pending_interaction = None
            backend = self.config.brain_backend
            use_structured_brain = (
                backend in (BrainBackend.WHOLE_BRAIN, BrainBackend.BRAIN_SIMULATION)
                and self.whole_brain is not None
                and self._whole_brain_adapter is not None
            )
            if use_structured_brain:
                brain_inputs = self._whole_brain_adapter.build_cycle_input(
                    instruction=self.config.default_cycle_instruction,
                )
                brain_result = self.whole_brain.process_cycle(brain_inputs)
                result = self._whole_brain_adapter.translate_cycle_result(brain_result)
                self.config.cycle_count += 1
                self._record_planning_to_workspace(
                    result,
                    cycle=self.config.cycle_count,
                    backend=self.brain_backend.value,
                )
                self.event_client.publish(
                    "agent.cycle.end",
                    {
                        "agent": self.state.agent_id,
                        "cycle": self.config.cycle_count,
                        "metrics": {"duration": perf_counter() - start_time},
                    },
                )
                return result
            if (
                backend == BrainBackend.TRANSFORMER
                and self.config.big_brain
                and self.brain is not None
            ):
                observation, memory_ctx = build_brain_inputs(self, self.config.brain.dim)
                obs_snapshot = observation.detach().clone() if hasattr(observation, "detach") else observation
                mem_snapshot = memory_ctx.detach().clone() if hasattr(memory_ctx, "detach") else memory_ctx
                thought = self.brain.think(observation, memory_ctx)
                result = self.brain.propose_action(thought)
                if getattr(self.brain, "supports_online_learning", False):
                    self._brain_pending_interaction = {
                        "observation": obs_snapshot,
                        "memory": mem_snapshot,
                        "brain_result": result,
                        "cycle": cycle,
                        "timestamp": time.time(),
                    }
                self._maybe_record_brain_sample(obs_snapshot, mem_snapshot, result, cycle)
                self.config.cycle_count += 1
                self._record_planning_to_workspace(
                    result,
                    cycle=self.config.cycle_count,
                    backend="transformer_brain",
                )
                self.event_client.publish(
                    "agent.cycle.end",
                    {
                        "agent": self.state.agent_id,
                        "cycle": self.config.cycle_count,
                        "metrics": {"duration": perf_counter() - start_time},
                    },
                )
                return result

            if backend != BrainBackend.LLM:
                logger.warning(
                    "Brain backend '%s' unavailable; falling back to LLM provider.",
                    backend.value if isinstance(backend, BrainBackend) else backend,
                )

            prompt: ChatPrompt = await self.build_prompt(
                scratchpad=self._prompt_scratchpad
            )
            prompt = self.on_before_think(prompt, scratchpad=self._prompt_scratchpad)

            logger.debug(f"Executing prompt:\n{dump_prompt(prompt)}")
            function_specs: list[CompletionModelFunction] | None = None
            if self.config.use_functions_api:
                function_specs = (
                    get_openai_command_specs(
                        self.command_registry.list_available_commands(self)
                    )
                    + list(self._prompt_scratchpad.commands.values())
                )

            background_tasks = self._launch_llm_side_tasks()
            llm_future = asyncio.create_task(
                self.llm_provider.create_chat_completion(
                    prompt.messages,
                    functions=function_specs if function_specs else None,
                    model_name=self.llm.name,
                    completion_parser=lambda message: message,
                )
            )

            pending_background: list[asyncio.Task[Any]] = []
            try:
                pending_background = await self._monitor_llm_and_tasks(
                    llm_future, background_tasks
                )
                response = await llm_future
                self.config.cycle_count += 1

                final_response, _review = await self._metacognitive_review_and_revise(
                    prompt,
                    self._prompt_scratchpad,
                    response,
                    functions=function_specs,
                )

                result = self.on_response(
                    llm_response=final_response,
                    prompt=prompt,
                    scratchpad=self._prompt_scratchpad,
                )
                self._record_planning_to_workspace(
                    result,
                    cycle=self.config.cycle_count,
                    backend="llm",
                    prompt=prompt,
                )
            finally:
                remaining_tasks = [
                    task
                    for task in (pending_background or background_tasks)
                    if not task.done()
                ]
                await self._finalize_background_tasks(remaining_tasks)

            self.event_client.publish(
                "agent.cycle.end",
                {
                    "agent": self.state.agent_id,
                    "cycle": self.config.cycle_count,
                    "metrics": {"duration": perf_counter() - start_time},
                },
            )
            return result
        except Exception as e:
            self.event_client.publish(
                "agent.cycle.error",
                {
                    "agent": self.state.agent_id,
                    "cycle": cycle,
                    "error": str(e),
                },
            )
            raise


    def _maybe_record_brain_sample(
        self,
        observation,
        memory_ctx,
        brain_result: ThoughtProcessOutput,
        cycle: int,
    ) -> None:
        """Optionally append a brain training sample to the configured dataset."""

        log_path = self.config.brain.dataset_logging_path
        if not log_path:
            return

        try:
            import torch  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            logger.debug("PyTorch not available; skipping brain sample logging")
            return

        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            command, args, info = brain_result
            action_vec = info.get("action", [])
            logits = torch.as_tensor(action_vec, dtype=torch.float32)
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
            memory_tensor = torch.as_tensor(memory_ctx, dtype=torch.float32)

            action_index = int(logits.argmax().item()) if logits.numel() else None
            sample = {
                "cycle": cycle,
                "command": command,
                "args": args,
                "observation": observation_tensor.tolist(),
                "memory": memory_tensor.tolist(),
                "action_logits": logits.tolist(),
                "action_index": action_index,
            }
            thought = info.get("thought")
            if thought is not None:
                sample["thought"] = thought

            with path.open("a", encoding="utf-8") as handle:
                json.dump(sample, handle, ensure_ascii=False)
                handle.write("\n")
        except Exception:  # pragma: no cover - logging only
            logger.debug("Failed to record transformer brain sample", exc_info=True)
    @abstractmethod
    async def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    ) -> ActionResult:
        """Executes the given command, if any, and returns the agent's response.

        Params:
            command_name: The name of the command to execute, if any.
            command_args: The arguments to pass to the command, if any.
            user_input: The user's input, if any.

        Returns:
            ActionResult: An object representing the result(s) of the command.
        """
        ...

    async def build_prompt(
        self,
        scratchpad: PromptScratchpad,
        extra_commands: Optional[list[CompletionModelFunction]] = None,
        extra_messages: Optional[list[ChatMessage]] = None,
        **extras,
    ) -> ChatPrompt:
        """Constructs a prompt using `self.prompt_strategy`.

        Params:
            scratchpad: An object for plugins to write additional prompt elements to.
                (E.g. commands, constraints, best practices)
            extra_commands: Additional commands that the agent has access to.
            extra_messages: Additional messages to include in the prompt.
        """

        if not extra_commands:
            extra_commands = []
        extra_messages = list(extra_messages or [])

        # Apply additions from plugins
        for plugin in self.config.plugins:
            if not plugin.can_handle_post_prompt():
                continue
            plugin.post_prompt(scratchpad)
        ai_directives = self.directives.copy(deep=True)
        ai_directives.resources += scratchpad.resources
        ai_directives.constraints += scratchpad.constraints
        ai_directives.best_practices += scratchpad.best_practices
        extra_commands += list(scratchpad.commands.values())

        extras_dict = dict(extras)
        context_tasks = [
            asyncio.create_task(self._build_knowledge_messages(extras_dict), name="knowledge"),
            asyncio.create_task(self._prepare_memory_summary(), name="memory"),
            asyncio.create_task(self._collect_external_context(extras_dict), name="external"),
            asyncio.create_task(self._build_workspace_messages(), name="workspace"),
        ]
        context_results = await asyncio.gather(
            *context_tasks, return_exceptions=True
        )
        for label, result in zip(("knowledge", "memory", "external", "workspace"), context_results):
            if isinstance(result, Exception):
                logger.debug(
                    "Failed to prepare %s context: %s",
                    label,
                    result,
                    exc_info=(type(result), result, result.__traceback__),
                )
                continue
            if result:
                extra_messages.extend(result)

        prompt = self.prompt_strategy.build_prompt(
            task=self.state.task,
            ai_profile=self.ai_profile,
            ai_directives=ai_directives,
            commands=get_openai_command_specs(
                self.command_registry.list_available_commands(self)
            )
            + extra_commands,
            event_history=self.event_history,
            max_prompt_tokens=self.send_token_limit,
            count_tokens=lambda x: self.llm_provider.count_tokens(x, self.llm.name),
            count_message_tokens=lambda x: self.llm_provider.count_message_tokens(
                x, self.llm.name
            ),
            extra_messages=extra_messages,
            **extras,
        )

        return prompt

    def on_before_think(
        self,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ChatPrompt:
        """Called after constructing the prompt but before executing it.

        Calls the `on_planning` hook of any enabled and capable plugins, adding their
        output to the prompt.

        Params:
            prompt: The prompt that is about to be executed.
            scratchpad: An object for plugins to write additional prompt elements to.
                (E.g. commands, constraints, best practices)

        Returns:
            The prompt to execute
        """
        current_tokens_used = self.llm_provider.count_message_tokens(
            prompt.messages, self.llm.name
        )
        plugin_count = len(self.config.plugins)
        for i, plugin in enumerate(self.config.plugins):
            if not plugin.can_handle_on_planning():
                continue
            plugin_response = plugin.on_planning(scratchpad, prompt.raw())
            if not plugin_response or plugin_response == "":
                continue
            message_to_add = ChatMessage.system(plugin_response)
            tokens_to_add = self.llm_provider.count_message_tokens(
                message_to_add, self.llm.name
            )
            if current_tokens_used + tokens_to_add > self.send_token_limit:
                logger.debug(f"Plugin response too long, skipping: {plugin_response}")
                logger.debug(f"Plugins remaining at stop: {plugin_count - i}")
                break
            prompt.messages.insert(
                -1, message_to_add
            )  # HACK: assumes cycle instruction to be at the end
            current_tokens_used += tokens_to_add
        return prompt

    def on_response(
        self,
        llm_response: ChatModelResponse,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ThoughtProcessOutput:
        """Called upon receiving a response from the chat model.

        Calls `self.parse_and_process_response()`.

        Params:
            llm_response: The raw response from the chat model.
            prompt: The prompt that was executed.
            scratchpad: An object containing additional prompt elements from plugins.
                (E.g. commands, constraints, best practices)

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """

        command_name, command_args, agent_thoughts = llm_response.parsed_result

        if command_name:
            reasoning = (
                agent_thoughts.get("thoughts", {}).get("reasoning", "")
                if isinstance(agent_thoughts, dict)
                else ""
            )
            self.event_history.register_action(
                Action(name=command_name, args=command_args, reasoning=reasoning)
            )

        agent_context = get_agent_context(self)
        if agent_context is not None and isinstance(agent_thoughts, dict):
            context_text = agent_thoughts.get("context")
            if context_text:
                agent_context.add(
                    StaticContextItem(
                        description="assistant_context",
                        source=None,
                        content=context_text,
                    )
                )

        return command_name, command_args, agent_thoughts

    @abstractmethod
    def parse_and_process_response(
        self,
        llm_response: AssistantChatMessage,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ThoughtProcessOutput:
        """Validate, parse & process the LLM's response.

        Must be implemented by derivative classes: no base implementation is provided,
        since the implementation depends on the role of the derivative Agent.

        Params:
            llm_response: The raw response from the chat model.
            prompt: The prompt that was executed.
            scratchpad: An object containing additional prompt elements from plugins.
                (E.g. commands, constraints, best practices)

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """
        pass

    # ------------------------------------------------------------------
    # Knowledge fusion helpers
    # ------------------------------------------------------------------
    async def _build_knowledge_messages(self, extras: Dict[str, Any]) -> list[ChatMessage]:
        if not getattr(self.config, "knowledge_context_enabled", True):
            return []

        knowledge_base = (
            self.knowledge_base if getattr(self.config, "use_knowledge_base", False) else None
        )
        query = self._select_knowledge_query(extras)
        if not query:
            return []

        timeout = getattr(self.config, "knowledge_context_timeout", 6.0)
        executor = self._resolve_async_executor()
        snippets = await collect_knowledge_context_async(
            query,
            knowledge_base=knowledge_base,
            top_k=getattr(self.config, "knowledge_context_top_k", 5),
            relation_limit=getattr(self.config, "knowledge_context_relation_limit", 3),
            timeout=timeout,
            executor=executor,
        )
        if not snippets:
            return []

        bullet_list = "\n".join(f"- {snippet}" for snippet in snippets)
        message_text = (
            "## Retrieved Knowledge\n"
            f"The following facts were surfaced for the query \"{query}\":\n"
            f"{bullet_list}\n"
            "Use this information as authoritative context when planning or answering."
        )
        logger.debug("Injecting %d knowledge snippets into prompt.", len(snippets))
        self._publish_workspace_message(
            message_type="knowledge.retrieval",
            payload={"query": query, "snippets": snippets},
            summary=self._truncate_text(snippets[0]) if snippets else None,
            tags=("knowledge", "retrieval"),
            importance=min(1.0, 0.25 + 0.05 * len(snippets)),
        )
        return [ChatMessage.system(message_text)]

    async def _prepare_memory_summary(self) -> list[ChatMessage]:
        timeout = getattr(self.config, "memory_summary_timeout", 8.0)
        try:
            await asyncio.wait_for(
                self.event_history.handle_compression(
                    self.llm_provider, self.legacy_config
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.debug(
                "Action history compression timed out after %.1fs; skipping summary.",
                timeout,
            )
            return []
        except Exception:
            logger.debug("Action history compression failed during prompt prep.", exc_info=True)
            return []

        summaries = [
            episode.summary
            for episode in self.event_history.episodes[-5:]
            if episode.summary
        ]
        if not summaries:
            return []

        bullet_list = "\n".join(f"- {summary}" for summary in summaries)
        message_text = (
            "## Recent Memory Summary\n"
            "Key outcomes from the last interactions:\n"
            f"{bullet_list}"
        )
        return [ChatMessage.system(message_text)]

    async def _collect_external_context(self, extras: Dict[str, Any]) -> list[ChatMessage]:
        timeout = getattr(self.config, "action_guard_timeout", 5.0)
        try:
            restricted = await asyncio.wait_for(
                asyncio.to_thread(self._evaluate_command_safety),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.debug(
                "Action guard evaluation timed out after %.1fs; skipping alerts.",
                timeout,
            )
            return []
        except Exception:
            logger.debug("Action guard evaluation failed during prompt prep.", exc_info=True)
            return []

        if not restricted:
            return []

        bullet_list = "\n".join(
            f"- {command}: {reason}" for command, reason in restricted.items()
        )
        message_text = (
            "## Action Guard Alerts\n"
            "The knowledge graph flagged these command constraints:\n"
            f"{bullet_list}\n"
            "Ensure requirements are satisfied before executing the above tools."
        )
        return [ChatMessage.system(message_text)]

    async def _build_workspace_messages(self) -> list[ChatMessage]:
        updates = self._consume_workspace_updates(limit=6)
        if not updates:
            return []

        lines = [self._format_workspace_message(message) for message in updates]
        content = "## Global Workspace Updates\n" + "\n".join(f"- {line}" for line in lines)
        return [ChatMessage.system(content)]

    def _workspace_source(self) -> str:
        identifier = self.state.agent_id
        if identifier:
            return identifier
        ai_name = getattr(self.ai_profile, "ai_name", None) or "agent"
        return f"agent:{ai_name}"

    def _truncate_text(self, text: str | None, limit: int = 240) -> str | None:
        if text is None:
            return None
        value = str(text)
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + "…"

    def _prepare_workspace_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalised = self._normalise_for_json(payload, limit=768)
        if isinstance(normalised, dict):
            return normalised
        return {"data": normalised}

    def _publish_workspace_message(
        self,
        *,
        message_type: str,
        payload: Dict[str, Any],
        summary: str | None = None,
        tags: Sequence[str] | None = None,
        importance: float = 0.0,
        attention: Optional[Sequence[float] | float] = None,
        propagate: bool = True,
    ) -> None:
        try:
            workspace_message = WorkspaceMessage(
                type=message_type,
                source=self._workspace_source(),
                payload=self._prepare_workspace_payload(payload),
                summary=self._truncate_text(summary),
                tags=tuple(tags or ()),
                importance=float(importance),
            )
            global_workspace.publish_message(
                workspace_message,
                attention=attention,
                propagate=propagate,
            )
        except Exception:
            logger.debug("Failed to publish workspace message", exc_info=True)

    def _consume_workspace_updates(
        self,
        *,
        limit: int = 6,
        types: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> list[WorkspaceMessage]:
        try:
            updates, cursor = global_workspace.get_updates(
                cursor=self._workspace_cursor,
                limit=limit,
                types=types,
                tags=tags,
                exclude_sources=[self._workspace_source()],
            )
        except Exception:
            logger.debug("Failed to fetch workspace updates", exc_info=True)
            return []

        if cursor is not None:
            self._workspace_cursor = cursor
        return updates

    def _format_workspace_message(self, message: WorkspaceMessage) -> str:
        timestamp = datetime.fromtimestamp(message.timestamp).strftime("%H:%M:%S")
        summary = message.summary
        if not summary:
            payload_preview = self._normalise_for_json(message.payload, limit=320)
            if isinstance(payload_preview, (dict, list)):
                summary = json.dumps(payload_preview, ensure_ascii=False)
            else:
                summary = str(payload_preview)
        summary_text = self._truncate_text(summary) or "(no summary)"
        tag_text = " ".join(f"#{tag}" for tag in message.tags) if message.tags else ""
        base = f"{timestamp} [{message.type}] {summary_text}"
        return f"{base} {tag_text}".strip()

    def _record_planning_to_workspace(
        self,
        result: ThoughtProcessOutput,
        *,
        cycle: int,
        backend: str,
        prompt: ChatPrompt | None = None,
    ) -> None:
        try:
            command_name, command_args, thoughts = result
        except Exception:
            return

        summary: Any = None
        if isinstance(thoughts, dict):
            summary = (
                thoughts.get("summary")
                or thoughts.get("plan")
                or thoughts.get("thoughts", {}).get("plan")
                or thoughts.get("thoughts", {}).get("reasoning")
            )
        elif isinstance(thoughts, str):
            summary = thoughts

        if not summary and command_name:
            summary = f"Proposed {command_name}"

        if isinstance(summary, (dict, list)):
            summary = json.dumps(self._normalise_for_json(summary, limit=320), ensure_ascii=False)

        payload: Dict[str, Any] = {
            "cycle": cycle,
            "backend": backend,
            "command": command_name,
            "arguments": self._normalise_for_json(command_args, limit=512),
            "thoughts": self._normalise_for_json(thoughts, limit=768),
        }
        if prompt is not None:
            payload["prompt_sections"] = self._extract_prompt_knowledge_sections(prompt)

        self._publish_workspace_message(
            message_type="agent.plan",
            payload=payload,
            summary=summary if isinstance(summary, str) else None,
            tags=("agent", "plan", backend),
            importance=1.0,
        )

    def _record_action_outcome_to_workspace(
        self,
        directive: ActionDirective,
        result: ActionResult,
    ) -> None:
        if result is None:
            return
        payload: Dict[str, Any] = {
            "cycle": self.config.cycle_count,
            "directive": directive.to_dict(),
            "command": directive.command_name,
        }

        try:
            result_payload = result.dict() if hasattr(result, "dict") else result
        except Exception:
            result_payload = repr(result)

        payload["result"] = self._normalise_for_json(result_payload, limit=768)
        status = getattr(result, "status", None)
        if status:
            payload["status"] = status

        summary: Optional[str] = None
        if isinstance(result_payload, dict):
            summary = (
                result_payload.get("outputs")
                or result_payload.get("reason")
                or result_payload.get("feedback")
            )
        if not summary and status and directive.command_name:
            summary = f"{directive.command_name} -> {status}"
        if not summary and directive.rationale:
            summary = directive.rationale
        if not summary and directive.command_name:
            summary = directive.command_name

        self._publish_workspace_message(
            message_type="agent.action_result",
            payload=payload,
            summary=summary,
            tags=("agent", "action", directive.command_name or "unknown"),
            importance=1.0,
        )

    def _evaluate_command_safety(self) -> dict[str, str]:
        restricted: dict[str, str] = {}
        for name, command in self.command_registry.commands.items():
            try:
                result = self._action_guard.evaluate(name, {}, context={})
            except Exception:
                continue
            if not result.allowed and result.reason:
                restricted[name] = result.reason
        return restricted

    def _launch_llm_side_tasks(self) -> list[asyncio.Task[Any]]:
        tasks: list[asyncio.Task[Any]] = []
        tasks.append(asyncio.create_task(self._run_memory_consolidation_background()))
        tasks.extend(self._start_tool_prefetch_tasks())
        return [task for task in tasks if task is not None]

    async def _run_memory_consolidation_background(self) -> None:
        timeout = getattr(self.config, "memory_consolidation_timeout", 10.0)
        try:
            await asyncio.wait_for(
                self.event_history.handle_compression(
                    self.llm_provider, self.legacy_config
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.debug(
                "Background memory consolidation timed out after %.1fs.", timeout
            )
        except Exception:
            logger.debug("Background memory consolidation failed.", exc_info=True)

    def _start_tool_prefetch_tasks(self) -> list[asyncio.Task[Any]]:
        tasks: list[asyncio.Task[Any]] = []
        for name, command in self.command_registry.commands.items():
            prefetch = getattr(command, "prefetch", None)
            if not callable(prefetch):
                continue
            try:
                sig = inspect.signature(prefetch)
            except (TypeError, ValueError):
                continue
            requires_args = any(
                parameter.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and parameter.default is inspect._empty
                for parameter in sig.parameters.values()
            )
            if requires_args:
                continue
            task = asyncio.create_task(
                self._execute_prefetch_callable(prefetch, name),
                name=f"prefetch:{name}",
            )
            tasks.append(task)
        return tasks

    async def _execute_prefetch_callable(
        self, prefetch: Callable[..., Any], command_name: str
    ) -> None:
        timeout = getattr(self.config, "tool_prefetch_timeout", 3.0)
        try:
            if inspect.iscoroutinefunction(prefetch):
                await asyncio.wait_for(prefetch(), timeout=timeout)
                return
            result = await asyncio.wait_for(
                asyncio.to_thread(prefetch),
                timeout=timeout,
            )
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug(
                "Prefetch for command '%s' timed out after %.1fs.",
                command_name,
                timeout,
            )
        except TypeError:
            logger.debug(
                "Skipping prefetch for command '%s' due to incompatible signature.",
                command_name,
            )
        except Exception:
            logger.debug(
                "Prefetch for command '%s' failed.", command_name, exc_info=True
            )

    async def _monitor_llm_and_tasks(
        self,
        llm_task: asyncio.Task[Any],
        background_tasks: list[asyncio.Task[Any]],
    ) -> list[asyncio.Task[Any]]:
        pending = set(background_tasks)
        while pending and not llm_task.done():
            done, pending = await asyncio.wait(
                pending | {llm_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=0.1,
            )
            if llm_task in done:
                break
            for task in done:
                if task is not llm_task:
                    self._handle_background_task_result(task)
        pending.discard(llm_task)
        return [task for task in pending if not task.done()]

    def _handle_background_task_result(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.debug(
                "Background task '%s' failed: %s",
                task.get_name() or repr(task),
                exception,
                exc_info=(type(exception), exception, exception.__traceback__),
            )

    async def _finalize_background_tasks(
        self, tasks: list[asyncio.Task[Any]]
    ) -> None:
        if not tasks:
            return
        timeout = getattr(self.config, "background_task_timeout", 5.0)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            for task in tasks:
                task.cancel()
            logger.debug(
                "Background tasks exceeded %.1fs and were cancelled.", timeout
            )
            return

        for result in results:
            if isinstance(result, Exception):
                logger.debug(
                    "Background task raised an exception: %s",
                    result,
                    exc_info=(type(result), result, result.__traceback__),
                )

    def _select_knowledge_query(self, extras: Dict[str, Any]) -> str:
        for key in ("user_input", "cycle_instruction", "query"):
            value = extras.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        task = getattr(self.state, "task", None)
        if task is None:
            return ""

        additional_input = getattr(task, "additional_input", None)
        if isinstance(additional_input, str) and additional_input.strip():
            return additional_input.strip()
        if isinstance(additional_input, (list, tuple)):
            for entry in reversed(additional_input):
                if isinstance(entry, str) and entry.strip():
                    return entry.strip()

        primary_input = getattr(task, "input", None)
        if isinstance(primary_input, str) and primary_input.strip():
            return primary_input.strip()
        return ""

    # ------------------------------------------------------------------
    # Metacognitive helpers
    # ------------------------------------------------------------------
    def _get_self_model_snapshot(self) -> Dict[str, Any]:
        """Return the latest SELF node view for metacognitive review."""

        # Prefer the runtime self-model if the whole brain is active.
        brain = getattr(self, "whole_brain", None)
        if brain is not None and hasattr(brain, "self_model"):
            try:
                snapshot = brain.self_model.snapshot()
                if isinstance(snapshot, dict):
                    return snapshot
            except Exception:  # pragma: no cover - defensive best effort
                logger.debug("Failed to query whole-brain self-model.", exc_info=True)

        # Fall back to the knowledge graph SELF node.
        try:
            graph = get_graph_store_instance()
            data = graph.query(node_id="SELF")
            nodes = data.get("nodes", []) if isinstance(data, dict) else []
            for node in nodes:
                node_id = getattr(node, "id", None) or getattr(node, "node_id", None)
                if str(node_id).upper() == "SELF":
                    properties = getattr(node, "properties", {}) or {}
                    if isinstance(properties, dict):
                        return dict(properties)
        except Exception:  # pragma: no cover - defensive best effort
            logger.debug("Failed to retrieve SELF node from knowledge graph.", exc_info=True)
        return {}

    def _extract_prompt_knowledge_sections(self, prompt: ChatPrompt) -> list[str]:
        """Collect retrieved knowledge messages from the issued prompt."""

        sections: list[str] = []
        for msg in getattr(prompt, "messages", []):
            content = getattr(msg, "content", None)
            role = getattr(msg, "role", None)
            if not content or role != ChatMessage.Role.SYSTEM:
                continue
            if content.startswith("## Retrieved Knowledge"):
                sections.append(content)
        return sections

    def _normalise_for_json(self, data: Any, *, depth: int = 0, limit: int = 2048) -> Any:
        """Lightweight sanitizer to keep review payloads JSON serializable."""

        if depth > 3:
            text = str(data)
            return (text[: limit] + "…") if len(text) > limit else text

        if data is None or isinstance(data, (bool, int, float)):
            return data

        if isinstance(data, str):
            if len(data) > limit:
                return data[:limit] + "…"
            return data

        if isinstance(data, dict):
            result: dict[str, Any] = {}
            for key, value in list(data.items())[:64]:
                result[str(key)] = self._normalise_for_json(value, depth=depth + 1, limit=limit)
            return result

        if isinstance(data, (list, tuple, set)):
            normalized = [
                self._normalise_for_json(item, depth=depth + 1, limit=limit) for item in list(data)[:64]
            ]
            return normalized

        return str(data)

    def _parse_review_output(self, text: str) -> Dict[str, Any]:
        """Parse reviewer output, tolerating fenced code blocks."""

        if not text:
            return {}

        cleaned = text.strip()
        if cleaned.startswith("```"):
            fence_end = cleaned.find("```", 3)
            if fence_end != -1:
                inner = cleaned[cleaned.find("\n", 3) + 1 : fence_end]
                cleaned = inner.strip()

        def _attempt_parse(candidate: str) -> Optional[Dict[str, Any]]:
            try:
                data = json.loads(candidate)
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None

        parsed = _attempt_parse(cleaned)
        if parsed is not None:
            return parsed

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = _attempt_parse(cleaned[start : end + 1])
            if parsed is not None:
                return parsed

        return {"raw": cleaned}

    def _attach_review_metadata(
        self,
        result: ThoughtProcessOutput,
        review_data: Optional[Dict[str, Any]],
    ) -> ThoughtProcessOutput:
        if not review_data:
            return result

        command_name, command_args, agent_thoughts = result
        review_summary = self._normalise_for_json(review_data, limit=1024)

        if isinstance(agent_thoughts, dict):
            container = agent_thoughts.get("meta")
            if not isinstance(container, dict):
                container = {"previous_meta": container} if container is not None else {}
            container["self_review"] = review_summary
            agent_thoughts["meta"] = container
        else:
            agent_thoughts = {
                "original_thoughts": agent_thoughts,
                "meta": {"self_review": review_summary},
            }
        return command_name, command_args, agent_thoughts

    async def _invoke_metacognitive_reviewer(
        self,
        prompt: ChatPrompt,
        draft_response: ChatModelResponse,
    ) -> Dict[str, Any]:
        """Run the reviewer role to analyse the draft assistant reply."""

        draft_text = (draft_response.response.content or "").strip()
        if not draft_text:
            return {}

        payload: Dict[str, Any] = {
            "cycle_index": self.config.cycle_count + 1,
            "draft_response": draft_text,
            "self_state": self._normalise_for_json(self._get_self_model_snapshot(), limit=768),
            "knowledge": self._normalise_for_json(self._extract_prompt_knowledge_sections(prompt), limit=768),
            "agent_name": getattr(self.ai_profile, "ai_name", "agent"),
            "goals": self._normalise_for_json(getattr(self.ai_profile, "ai_goals", []), limit=256),
        }

        task = getattr(self.state, "task", None)
        if task is not None:
            task_info: Dict[str, Any] = {}
            for attr in ("id", "task_id", "input", "additional_input", "extra_info"):
                value = getattr(task, attr, None)
                if value is not None:
                    task_info[attr] = value
            if task_info:
                payload["task"] = self._normalise_for_json(task_info, limit=512)

        reviewer_messages = [
            ChatMessage.system(
                (
                    "You are the agent's internal self-check reviewer. Analyse the `draft_response` using the "
                    "provided `self_state` (known skills, weaknesses) and `knowledge` (authoritative facts). "
                    "Identify logical flaws, contradictions, missing steps, or rule violations that would prevent "
                    "safe execution. Respond ONLY with JSON containing:\n"
                    "- approved: boolean (true if the draft is acceptable as-is)\n"
                    "- requires_revision: boolean (true if changes are required before execution)\n"
                    "- issues: array of concise strings describing problems (max 5)\n"
                    "- suggestions: string with actionable revision guidance (can be empty)\n"
                    "- confidence: number between 0 and 1 reflecting review certainty"
                )
            ),
            ChatMessage.user(json.dumps(self._normalise_for_json(payload, limit=1024), ensure_ascii=False)),
        ]

        try:
            review_response = await self.llm_provider.create_chat_completion(
                reviewer_messages,
                model_name=self.llm.name,
                completion_parser=lambda message: message,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Metacognitive reviewer call failed.", exc_info=True)
            return {}

        review_text = (review_response.response.content or "").strip()
        review_data = self._parse_review_output(review_text)
        review_data.setdefault("raw", review_text)
        issues = review_data.get("issues")
        if isinstance(issues, list):
            max_notes = max(1, getattr(self.config, "metacognitive_review_max_issue_notes", 5))
            review_data["issues"] = [
                str(issue)[:280] for issue in issues[:max_notes] if isinstance(issue, (str, int, float))
            ]
        review_data.setdefault("timestamp", datetime.utcnow().isoformat(timespec="seconds"))
        return review_data

    async def _metacognitive_review_and_revise(
        self,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
        draft_response: ChatModelResponse,
        *,
        functions: Optional[list[CompletionModelFunction]] = None,
    ) -> tuple[ChatModelResponse, Optional[Dict[str, Any]]]:
        """Review the draft, optionally revise, and attach metadata."""

        review_data: Dict[str, Any] | None = None
        if getattr(self.config, "metacognitive_review_enabled", True):
            review_data = await self._invoke_metacognitive_reviewer(prompt, draft_response)

        def _finalise(response: ChatModelResponse, meta: Optional[Dict[str, Any]]) -> ChatModelResponse:
            parsed = self.parse_and_process_response(response.response, prompt, scratchpad)
            response.parsed_result = self._attach_review_metadata(parsed, meta)
            return response

        if not review_data:
            return _finalise(draft_response, None), None

        approved = bool(review_data.get("approved", False))
        requires_revision = bool(review_data.get("requires_revision", False))
        issues_present = bool(review_data.get("issues"))
        must_revise = requires_revision or (
            issues_present and getattr(self.config, "metacognitive_review_require_revision", True)
        )

        if not must_revise and approved:
            return _finalise(draft_response, review_data), review_data

        feedback_lines: list[str] = []
        for issue in review_data.get("issues", []):
            feedback_lines.append(f"- {issue}")
        suggestions = str(review_data.get("suggestions", "") or "").strip()
        feedback_text = "Internal self-review feedback:\n"
        if feedback_lines:
            feedback_text += "\n".join(feedback_lines)
        if suggestions:
            feedback_text += ("\n" if feedback_lines else "") + f"Suggested revision: {suggestions}"

        revision_messages = list(prompt.messages)
        revision_messages.append(
            ChatMessage(role=ChatMessage.Role.ASSISTANT, content=draft_response.response.content or "")
        )
        revision_messages.append(ChatMessage.system(feedback_text))
        revision_messages.append(
            ChatMessage.user(
                "Revise your previous response to address every listed issue while preserving valid reasoning. "
                "Return your answer in the identical format required for agent outputs."
            )
        )

        try:
            revised_response = await self.llm_provider.create_chat_completion(
                revision_messages,
                model_name=self.llm.name,
                completion_parser=lambda message: message,
                functions=functions if functions else None,
            )
        except Exception:  # pragma: no cover - fall back to draft
            logger.debug("Metacognitive revision call failed; using draft output.", exc_info=True)
            review_data["revision_applied"] = False
            return _finalise(draft_response, review_data), review_data

        review_data["revision_applied"] = True
        return _finalise(revised_response, review_data), review_data
