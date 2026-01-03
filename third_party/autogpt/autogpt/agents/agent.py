from __future__ import annotations

import inspect
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import sentry_sdk  # type: ignore
except Exception:  # pragma: no cover - optional dependency absent
    class _SentryStub:
        @staticmethod
        def capture_exception(*_args: Any, **_kwargs: Any) -> None:
            return None

    sentry_sdk = _SentryStub()  # type: ignore[assignment]
from pydantic import Field

from autogpt.core.configuration import Configurable, LearningConfiguration
from autogpt.core.prompting import ChatPrompt
from autogpt.core.resource.model_providers import (
    AssistantChatMessage,
    ChatMessage,
    ChatModelProvider,
)
from autogpt.core.learning import ExperienceLearner
from autogpt.core.learning.experience_store import ExperienceLogStore, ExperienceRecorder
from autogpt.core.self_improvement import SelfImprovementEngine
from autogpt.file_storage.base import FileStorage
from autogpt.logs.log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.logs.utils import fmt_kwargs
from autogpt.models.action_history import (
    ActionErrorResult,
    ActionInterruptedByHuman,
    ActionResult,
    ActionSuccessResult,
)
from autogpt.core.ability.schema import AbilityResult, Knowledge
from autogpt.models.command import CommandOutput
from autogpt.models.context_item import ContextItem

from events.coordination import TaskStatus

try:  # pragma: no cover - optional import path
    from reasoning.decision_engine import ActionDirective
except ModuleNotFoundError:  # pragma: no cover - fallback for local repo layout
    from backend.reasoning.decision_engine import ActionDirective
from backend.memory import LongTermMemory, WorkingMemory

from .base import BaseAgent, BaseAgentConfiguration, BaseAgentSettings
from .features.agent_file_manager import AgentFileManagerMixin
from .features.context import ContextMixin
from .features.watchdog import WatchdogMixin
from .prompt_strategies.one_shot import (
    OneShotAgentPromptConfiguration,
    OneShotAgentPromptStrategy,
)
from .utils.exceptions import (
    AgentException,
    AgentTerminated,
    CommandExecutionError,
    DuplicateOperationError,
    UnknownCommandError,
)

if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.models.command_registry import CommandRegistry
    from autogpt.core.brain.transformer_brain import TransformerBrain
    from knowledge import UnifiedKnowledgeBase
    from reasoning import DecisionEngine

logger = logging.getLogger(__name__)


class AgentConfiguration(BaseAgentConfiguration):
    """Configuration for the primary Agent."""

    learning: LearningConfiguration = Field(
        default_factory=LearningConfiguration,
        description="Experience learning settings",
    )


class AgentSettings(BaseAgentSettings):
    config: AgentConfiguration = Field(default_factory=AgentConfiguration)
    prompt_config: OneShotAgentPromptConfiguration = Field(
        default_factory=(
            lambda: OneShotAgentPromptStrategy.default_configuration.copy(deep=True)
        )
    )


class Agent(
    ContextMixin,
    AgentFileManagerMixin,
    WatchdogMixin,
    BaseAgent,
    Configurable[AgentSettings],
):
    """AutoGPT's primary Agent; uses one-shot prompting."""

    default_settings: AgentSettings = AgentSettings(
        name="Agent",
        description=__doc__,
    )

    prompt_strategy: OneShotAgentPromptStrategy

    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: ChatModelProvider,
        command_registry: CommandRegistry,
        file_storage: FileStorage,
        legacy_config: Config,
        brain: TransformerBrain | None = None,
        whole_brain: Any | None = None,
        knowledge_base: "UnifiedKnowledgeBase" | None = None,
        decision_engine: "DecisionEngine" | None = None,
    ):
        prompt_strategy = OneShotAgentPromptStrategy(
            configuration=settings.prompt_config,
            logger=logger,
        )
        super().__init__(
            settings=settings,
            llm_provider=llm_provider,
            prompt_strategy=prompt_strategy,
            command_registry=command_registry,
            file_storage=file_storage,
            legacy_config=legacy_config,
            brain=brain,
            whole_brain=whole_brain,
            knowledge_base=knowledge_base,
            decision_engine=decision_engine,
        )
        self._pending_knowledge_statements: List[str] = []
        self._pending_knowledge_facts: List[Dict[str, Any]] = []
        log_path = Path(self.config.learning.log_path)
        self._experience_store = ExperienceLogStore(
            log_path=log_path,
            max_bytes=self.config.learning.max_log_bytes if self.config.learning.max_log_bytes else None,
        )
        self._experience_recorder = ExperienceRecorder(
            store=self._experience_store,
            max_summary_chars=self.config.learning.max_summary_chars,
        )
        self._experience_learner = ExperienceLearner(
            memory=self._experience_store,
            config=self.config.learning,
            logger=logger,
        )
        baseline_command_availability: dict[str, Any] = {}
        commands = getattr(self.command_registry, "commands", None)
        try:
            items = commands.items() if commands is not None else []
        except Exception:
            items = []
        try:
            for name, cmd in items:
                available = getattr(cmd, "available", None)
                if available is not None and not callable(available):
                    baseline_command_availability[name] = available
        except TypeError:
            baseline_command_availability = {}
        self._baseline_command_availability = baseline_command_availability
        self._self_improvement_engine = (
            SelfImprovementEngine(
                config=self.config.learning,
                store=self._experience_store,
                logger=logger,
            )
            if self.config.learning.auto_improve
            else None
        )
        self._current_improvement_plan: dict[str, Any] | None = None
        self._preferred_commands: set[str] = set()
        self._improvement_hint: str | None = None
        self._cycles_since_improvement = 0
        self._load_improvement_plan()

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        """Timestamp the agent was created; only used for structured debug logging."""

        self.log_cycle_handler = LogCycleHandler()
        """LogCycleHandler for structured debug logging."""

        self.working_memory = WorkingMemory(self.config.working_memory_capacity)
        for item in self.state.working_memory_items:
            self.working_memory.store(item)

        memory_path = self.state.long_term_memory_path or f"agents/{self.state.agent_id}/long_term_memory.sqlite"
        self.state.long_term_memory_path = memory_path
        root_storage = getattr(self, "_file_storage", None)
        base_root = getattr(root_storage, "root", None) if root_storage is not None else None
        try:
            base_root_path = Path(base_root) if base_root else Path(".")
        except TypeError:
            base_root_path = Path(".")

        long_term_path = base_root_path / memory_path
        try:
            long_term_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug("Unable to create long-term memory directory.", exc_info=True)
        self.long_term_memory = LongTermMemory(long_term_path)

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
        """Report current task status via the event bus."""

        statements: List[str] = knowledge_statements or []
        facts: List[Dict[str, Any]] = knowledge_facts or []

        if status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
            flushed_statements, flushed_facts = self._flush_pending_knowledge()
            statements.extend(flushed_statements)
            facts.extend(flushed_facts)

        meta = dict(metadata or {})
        meta.setdefault("cycles", self.config.cycle_count)

        super().report_status(
            task_id,
            status,
            detail,
            summary=summary,
            knowledge_statements=statements or None,
            knowledge_facts=facts or None,
            metadata=meta,
        )

    async def build_prompt(
        self,
        *args,
        extra_messages: Optional[list[ChatMessage]] = None,
        include_os_info: Optional[bool] = None,
        **kwargs,
    ) -> ChatPrompt:
        if not extra_messages:
            extra_messages = []

        # Clock
        extra_messages.append(
            ChatMessage.system(f"The current time and date is {time.strftime('%c')}"),
        )
        if self._improvement_hint:
            extra_messages.append(
                ChatMessage.system(f"Strategy hint: {self._improvement_hint}")
            )

        if include_os_info is None:
            include_os_info = self.legacy_config.execute_local_commands

        return await super().build_prompt(
            *args,
            extra_messages=extra_messages,
            include_os_info=include_os_info,
            **kwargs,
        )

    def on_before_think(self, *args, **kwargs) -> ChatPrompt:
        prompt = super().on_before_think(*args, **kwargs)

        self.log_cycle_handler.log_count_within_cycle = 0
        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            prompt.raw(),
            CURRENT_CONTEXT_FILE_NAME,
        )
        return prompt

    def parse_and_process_response(
        self, llm_response: AssistantChatMessage, *args, **kwargs
    ) -> Agent.ThoughtProcessOutput:
        for plugin in self.config.plugins:
            if not plugin.can_handle_post_planning():
                continue
            llm_response.content = plugin.post_planning(llm_response.content or "")

        (
            command_name,
            arguments,
            assistant_reply_dict,
        ) = self.prompt_strategy.parse_response_content(llm_response)

        # Check if command_name and arguments are already in the event_history
        if self.event_history.matches_last_command(command_name, arguments):
            raise DuplicateOperationError(
                f"The command {command_name} with arguments {arguments} "
                f"has been just executed."
            )

        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            assistant_reply_dict,
            NEXT_ACTION_FILE_NAME,
        )

        return command_name, arguments, assistant_reply_dict

    async def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    ) -> ActionResult:
        result: ActionResult

        directive = self.before_execute(command_name, command_args, None)
        directive = directive.resolve(command_name, command_args)
        command_name = directive.command_name or command_name
        command_args = directive.command_args or command_args

        if not directive.approved:
            reason = directive.rationale or "Action requires replanning as directed."
            result = ActionErrorResult(reason=reason)
        elif command_name == "human_feedback":
            result = ActionInterruptedByHuman(feedback=user_input)
            self.log_cycle_handler.log_cycle(
                self.ai_profile.ai_name,
                self.created_at,
                self.config.cycle_count,
                user_input,
                USER_INPUT_FILE_NAME,
            )
        else:
            for plugin in self.config.plugins:
                if not plugin.can_handle_pre_command():
                    continue
                command_name, command_args = plugin.pre_command(
                    command_name, command_args
                )

            try:
                return_value = await execute_command(
                    command_name=command_name,
                    arguments=command_args,
                    agent=self,
                )

                # Intercept ContextItem if one is returned by the command
                if type(return_value) is tuple and isinstance(
                    return_value[1], ContextItem
                ):
                    context_item = return_value[1]
                    return_value = return_value[0]
                    logger.debug(
                        f"Command {command_name} returned a ContextItem: {context_item}"
                    )
                    self.context.add(context_item)

                result = ActionSuccessResult(outputs=return_value)
            except AgentTerminated:
                raise
            except AgentException as e:
                result = ActionErrorResult.from_exception(e)
                logger.warning(
                    f"{command_name}({fmt_kwargs(command_args)}) raised an error: {e}"
                )
                sentry_sdk.capture_exception(e)

            result_tlength = self.llm_provider.count_tokens(str(result), self.llm.name)
            if result_tlength > self.send_token_limit // 3:
                result = ActionErrorResult(
                    reason=f"Command {command_name} returned too much output. "
                    "Do not execute this command again with the same arguments."
                )

            for plugin in self.config.plugins:
                if not plugin.can_handle_post_command():
                    continue
                if result.status == "success":
                    result.outputs = plugin.post_command(command_name, result.outputs)
                elif result.status == "error":
                    result.reason = plugin.post_command(command_name, result.reason)

        self.after_execute(directive, result)

        if (
            self.brain is not None
            and getattr(self.brain, "supports_online_learning", False)
        ):
            pending = getattr(self, "_brain_pending_interaction", None)
            if pending is not None:
                try:
                    self.brain.complete_interaction(
                        observation=pending.get("observation"),
                        memory=pending.get("memory"),
                        brain_result=pending.get("brain_result"),
                        outcome=result,
                        metadata={
                            "command": command_name,
                            "args": {k: str(v) for k, v in command_args.items()},
                            "cycle": self.config.cycle_count,
                            "reward": self._reward_for_result(result),
                        },
                    )
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Failed to apply online learning update to transformer brain",
                        exc_info=True,
                    )
                finally:
                    self._brain_pending_interaction = None
        else:
            self._brain_pending_interaction = None

        if isinstance(result, ActionErrorResult):
            self._handle_action_failure(directive, result)

        # Update action history
        self.event_history.register_result(result)
        await self.event_history.handle_compression(
            self.llm_provider, self.legacy_config
        )

        self._record_experience(command_name, command_args, result)

        # Allow the agent to learn from its recent experience and adjust
        # command availability based on the learned success rates
        updated_weights = self._experience_learner.learn_from_experience()
        for name, weight in updated_weights.items():
            if cmd := self.command_registry.get_command(name):
                # Simple heuristic: disable commands with low success rate
                cmd.available = weight >= 0.5

        self._maybe_run_auto_improvement()

        return result

    def _handle_action_failure(
        self, directive: ActionDirective, result: ActionErrorResult
    ) -> None:
        """Emit failure signals and schedule replanning after an action error."""

        command_name = directive.command_name or "unknown"
        command_args = directive.command_args or {}

        error_detail_parts = [result.reason]
        if result.error is not None:
            error_detail_parts.append(
                f"{result.error.exception_type}: {result.error.message}"
            )
        error_detail = " | ".join(part for part in error_detail_parts if part)

        task = getattr(self, "task", None) or getattr(self.state, "task", None)
        task_id = getattr(task, "task_id", None)

        if task_id:
            try:
                self.report_status(
                    task_id,
                    status=TaskStatus.FAILED,
                    detail=error_detail,
                    metadata={
                        "command": command_name,
                        "cycle": self.config.cycle_count,
                    },
                )
            except Exception:  # pragma: no cover - best effort logging
                logger.debug("Failed to report task failure status", exc_info=True)

        failure_payload = {
            "agent": getattr(self.state, "agent_id", None),
            "task_id": task_id,
            "command": command_name,
            "arguments": self._serialise_args(command_args),
            "reason": result.reason,
            "cycle": self.config.cycle_count,
        }
        if result.error is not None:
            failure_payload["error"] = result.error.dict()
        try:
            self.event_client.publish("agent.action.failure", failure_payload)
        except Exception:  # pragma: no cover - best effort logging
            logger.debug("Failed to publish action failure event", exc_info=True)

        if not directive.requires_replan:
            rationale = (
                f"Replan required after failure of {command_name}: {result.reason}"
            )
            metadata = {
                "failed_command": command_name,
                "reason": result.reason,
            }
            if result.error is not None:
                metadata["error_type"] = result.error.exception_type
                metadata["error_message"] = result.error.message
            try:
                self.queue_directive(
                    ActionDirective.replan(rationale, metadata=metadata)
                )
            except Exception:  # pragma: no cover - best effort logging
                logger.debug("Failed to enqueue replanning directive", exc_info=True)

    @staticmethod
    def _reward_for_result(result: ActionResult) -> float:
        if isinstance(result, ActionSuccessResult):
            return 1.0
        if isinstance(result, ActionErrorResult):
            return -1.0
        if isinstance(result, ActionInterruptedByHuman):
            return 0.0
        return 0.0

    def _record_experience(self, command_name: str, command_args: dict, result: ActionResult) -> None:
        if not self.config.learning.enabled:
            return
        try:
            serialised_args = self._serialise_args(command_args)
        except Exception as exc:
            logger.debug(
                "Failed to serialise command arguments for experience log: %s",
                exc,
            )
            serialised_args = {k: str(v) for k, v in command_args.items()}

        if isinstance(result, ActionSuccessResult):
            summary_source = result.outputs
        elif isinstance(result, ActionErrorResult):
            summary_source = result.reason or str(result.error or "")
        elif isinstance(result, ActionInterruptedByHuman):
            summary_source = result.feedback
        else:
            summary_source = str(result)

        summary = (
            summary_source
            if isinstance(summary_source, str)
            else json.dumps(summary_source, ensure_ascii=False, default=str)
        )

        self._experience_recorder.record(
            task_id=self.task.task_id,
            cycle=self.config.cycle_count,
            command_name=command_name,
            command_args=serialised_args,
            result_status=getattr(result, "status", "unknown"),
            result_summary=summary,
            metadata={
                "agent_id": self.settings.agent_id,
            },
        )

    def _maybe_run_auto_improvement(self) -> None:
        if not self.config.learning.auto_improve or not self._self_improvement_engine:
            return
        self._cycles_since_improvement += 1
        interval = max(self.config.learning.improvement_interval or 1, 1)
        if self._cycles_since_improvement < interval:
            return
        self._cycles_since_improvement = 0
        plan = self._self_improvement_engine.evaluate_and_apply()
        if plan is not None:
            self._current_improvement_plan = plan
            self._preferred_commands = set(plan.get("preferred_commands") or [])
            self._update_improvement_hint(plan)
            self._apply_current_plan()
        else:
            self._load_improvement_plan()

    def ingest_evaluation_feedback(
        self,
        evaluation: Any | None = None,
        benchmark_results: Any | None = None,
    ) -> None:
        """Inject external evaluation results and trigger improvement planning."""

        if not self.config.learning.auto_improve or not self._self_improvement_engine:
            logger.debug("Auto-improvement disabled; skipping evaluation feedback")
            return

        if evaluation is None:
            summary = None
        elif isinstance(evaluation, dict):
            summary = evaluation
        elif hasattr(evaluation, "summary") and callable(
            getattr(evaluation, "summary")
        ):
            summary = evaluation.summary()
        else:
            summary = None
            logger.warning("Unsupported evaluation payload provided to agent", extra={"type": type(evaluation)})

        plan = self._self_improvement_engine.evaluate_and_apply(
            evaluation_summary=summary,
            benchmark_results=benchmark_results,
        )

        self._cycles_since_improvement = 0
        if plan is not None:
            self._current_improvement_plan = plan
            self._preferred_commands = set(plan.get("preferred_commands") or [])
            self._update_improvement_hint(plan)
            self._apply_current_plan()
        else:
            self._load_improvement_plan()

    def _load_improvement_plan(self) -> None:
        try:
            plan_path = Path(self.config.learning.plan_output_path)
        except Exception:
            logger.exception("Invalid plan output path")
            self._current_improvement_plan = None
            self._preferred_commands = set()
            self._improvement_hint = None
            return
        if plan_path.exists():
            try:
                plan = json.loads(plan_path.read_text(encoding='utf-8'))
            except Exception:
                logger.exception("Failed to load improvement plan")
                plan = None
        else:
            plan = None
        self._current_improvement_plan = plan
        self._preferred_commands = set(plan.get("preferred_commands") or []) if plan else set()
        self._update_improvement_hint(plan)
        self._apply_current_plan()

    def _update_improvement_hint(self, plan: dict[str, Any] | None) -> None:
        if plan:
            hint = (plan.get("prompt_hints") or "").strip()
            self._improvement_hint = hint or None
        else:
            self._improvement_hint = None

    def _apply_current_plan(self) -> None:
        if not hasattr(self, '_baseline_command_availability'):
            return
        plan = self._current_improvement_plan or {}
        disabled = set(plan.get("disabled_commands") or [])
        for name, baseline in self._baseline_command_availability.items():
            command = self.command_registry.get_command(name)
            if not command or callable(command.available):
                continue
            command.available = bool(baseline) and name not in disabled

    @staticmethod
    def _serialise_args(args: dict) -> dict:
        if not args:
            return {}
        return json.loads(json.dumps(args, ensure_ascii=False, default=str))

    def _publish_action_event(
        self,
        command_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        payload = {
            "agent": getattr(self.state, "agent_id", None),
            "task_id": getattr(getattr(self.state, "task", None), "task_id", None),
            "command": command_name,
            "arguments": self._serialise_args(arguments),
            "result": self._summarise_result(result),
            "cycle": self.config.cycle_count,
        }
        try:
            self.event_client.publish("agent.action.executed", payload)
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to publish action executed event.", exc_info=True)

    @staticmethod
    def _summarise_result(result: Any) -> str:
        if isinstance(result, tuple):
            return str(result[0])
        return str(result)

    def _capture_command_result(
        self,
        result: Any,
        command_name: str,
        arguments: dict[str, str],
    ) -> None:
        ability_result: AbilityResult | None = None
        if isinstance(result, AbilityResult):
            ability_result = result
        elif isinstance(result, tuple) and result and isinstance(result[0], AbilityResult):
            ability_result = result[0]

        if ability_result is None:
            return

        if ability_result.new_knowledge:
            self._register_new_knowledge(
                ability_result.new_knowledge,
                source=command_name,
            )

        message = ability_result.message
        if isinstance(message, str) and message.strip():
            self._pending_knowledge_statements.append(message.strip())

    def _register_new_knowledge(self, knowledge: Knowledge, *, source: Optional[str] = None) -> None:
        if knowledge is None or not knowledge.content:
            return

        metadata = dict(knowledge.content_metadata or {})
        src = source or metadata.get("source") or "ability"

        statements: List[str] = []
        data: Any = None
        content = knowledge.content
        if isinstance(content, str):
            text = content.strip()
            if text:
                statements.append(text)
                data = self._try_parse_json(text)
        elif isinstance(content, (dict, list)):
            data = content
            try:
                statements.append(json.dumps(content, ensure_ascii=False))
            except TypeError:
                statements.append(str(content))
        else:
            statements.append(str(content))

        if metadata:
            try:
                statements.append(json.dumps(metadata, ensure_ascii=False))
            except TypeError:
                statements.append(str(metadata))

        for statement in statements:
            stmt = statement.strip()
            if stmt:
                self._pending_knowledge_statements.append(stmt)

        if data is not None:
            facts = self._extract_facts_from_json_like(data, metadata, src)
            for fact in facts:
                if self._is_valid_fact(fact):
                    self._pending_knowledge_facts.append(fact)

    def _extract_facts_from_json_like(
        self,
        data: Any,
        metadata: Dict[str, Any],
        source: str,
    ) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                if {"subject", "predicate"} <= node.keys() and ("object" in node or "obj" in node):
                    fact = {
                        "subject": str(node.get("subject", "")).strip(),
                        "predicate": str(node.get("predicate", "")).strip(),
                        "object": str(node.get("object", node.get("obj", ""))).strip(),
                        "subject_id": self._optional_str(node.get("subject_id")),
                        "object_id": self._optional_str(node.get("object_id")),
                        "subject_description": self._optional_str(node.get("subject_description")),
                        "object_description": self._optional_str(node.get("object_description")),
                        "metadata": {**metadata, **(node.get("metadata") or {})},
                        "confidence": self._optional_float(node.get("confidence")),
                        "source": self._optional_str(node.get("source")) or source,
                    }
                    facts.append(fact)
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(data)
        return facts

    @staticmethod
    def _try_parse_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def _optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_valid_fact(fact: Dict[str, Any]) -> bool:
        return all(
            isinstance(fact.get(key), str) and fact[key]
            for key in ("subject", "predicate", "object")
        )

    def _flush_pending_knowledge(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        statements = list(self._pending_knowledge_statements)
        facts = list(self._pending_knowledge_facts)
        self._pending_knowledge_statements.clear()
        self._pending_knowledge_facts.clear()
        return statements, facts



#############
# Utilities #
#############


async def execute_command(
    command_name: str,
    arguments: dict[str, str],
    agent: Agent,
) -> CommandOutput:
    """Execute the command and return the result

    Args:
        command_name (str): The name of the command to execute
        arguments (dict): The arguments for the command
        agent (Agent): The agent that is executing the command

    Returns:
        str: The result of the command
    """
    guard = getattr(agent, "_action_guard", None)
    if guard is not None:
        try:
            guard_result = guard.evaluate(
                command_name,
                arguments,
                context={
                    "agent_id": getattr(agent.state, "agent_id", None),
                    "mode": getattr(agent.config, "mode", None),
                    "big_brain": getattr(agent.config, "big_brain", None),
                },
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Knowledge action guard evaluation failed.", exc_info=True)
        else:
            if not guard_result.allowed:
                payload = {
                    "agent": getattr(agent.state, "agent_id", None),
                    "command": command_name,
                    "arguments": agent._serialise_args(arguments),
                    "reason": guard_result.reason,
                    "violations": guard_result.violations,
                }
                try:
                    agent.event_client.publish("agent.action.blocked", payload)
                except Exception:  # pragma: no cover - event bus optional
                    logger.debug("Failed to publish action blocked event.", exc_info=True)
                raise CommandExecutionError(
                    guard_result.reason or f"Action '{command_name}' blocked by knowledge guard."
                )

    # Execute a native command with the same name or alias, if it exists
    if command := agent.command_registry.get_command(command_name):
        try:
            result = command(**arguments, agent=agent)
            if inspect.isawaitable(result):
                result = await result
            agent._capture_command_result(result, command_name, arguments)
            agent._publish_action_event(command_name, arguments, result)
            return result
        except AgentException:
            raise
        except Exception as e:
            raise CommandExecutionError(str(e))

    # Handle non-native commands (e.g. from plugins)
    if agent._prompt_scratchpad:
        for name, command in agent._prompt_scratchpad.commands.items():
            if (
                command_name == name
                or command_name.lower() == command.description.lower()
            ):
                try:
                    result = command.method(**arguments)
                    if inspect.isawaitable(result):
                        result = await result
                    agent._capture_command_result(result, command_name, arguments)
                    agent._publish_action_event(command_name, arguments, result)
                    return result
                except AgentException:
                    raise
                except Exception as e:
                    raise CommandExecutionError(str(e))

    raise UnknownCommandError(
        f"Cannot execute command '{command_name}': unknown command."
    )
