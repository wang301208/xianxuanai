import heapq
import logging
import subprocess
import re
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from autogpt.core.ability import (
    AbilityRegistrySettings,
    AbilityResult,
    SimpleAbilityRegistry,
)
from autogpt.core.agent.layered import LayeredAgent
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory import MemorySettings, SimpleMemory
from autogpt.core.planning import PlannerSettings, SimplePlanner, Task, TaskStatus
from autogpt.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
    SimplePluginService,
)
from autogpt.core.resource.model_providers import (
    CompletionModelFunction,
    OpenAIProvider,
    OpenAISettings,
    AssistantChatMessage,
    ChatModelProvider,
)
from autogpt.core.resource.model_providers.schema import ChatModelResponse, ModelProviderName
from modules.knowledge import KnowledgeAcquisitionManager, SelfTeacher
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceSettings
from autogpt.core.agent.cognition import SimpleBrainAdapter, SimpleBrainAdapterSettings
from autogpt.config import Config
from autogpt.core.knowledge_extractor import extract
from autogpt.core.knowledge_conflict import resolve_conflicts
from monitoring import ActionLogger
from autogpt.core.global_workspace import GlobalWorkspace
from backend.knowledge.registry import get_graph_store_instance
from backend.reasoning import (
    SymbolicReasoner,
    CommonsenseKnowledge,
    CommonsenseValidator,
    LogicConstraintObserver,
    CausalObserver,
    CommonsenseObserver,
    set_symbolic_reasoner,
    set_commonsense_validator,
    set_causal_reasoner,
)

if TYPE_CHECKING:  # pragma: no cover - typing only import
    from backend.world_model import WorldModel


class AgentSystems(SystemConfiguration):
    ability_registry: PluginLocation
    memory: PluginLocation
    cognition: PluginLocation
    openai_provider: PluginLocation
    planning: PluginLocation
    creative_planning: PluginLocation
    workspace: PluginLocation


class AgentConfiguration(SystemConfiguration):
    cycle_count: int
    max_task_cycle_count: int
    creation_time: str
    name: str
    role: str
    goals: list[str]
    systems: AgentSystems
    self_assess_frequency: int = 5


class AgentSystemSettings(SystemSettings):
    configuration: AgentConfiguration


class AgentSettings(BaseModel):
    agent: AgentSystemSettings
    ability_registry: AbilityRegistrySettings
    memory: MemorySettings
    cognition: SimpleBrainAdapterSettings = Field(
        default_factory=lambda: SimpleBrainAdapter.default_settings.copy(deep=True)
    )
    openai_providers: dict[str, OpenAISettings] = Field(default_factory=dict)
    planning: PlannerSettings
    creative_planning: PlannerSettings
    workspace: WorkspaceSettings

    def update_agent_name_and_goals(self, agent_goals: dict) -> None:
        self.agent.configuration.name = agent_goals["agent_name"]
        self.agent.configuration.role = agent_goals["agent_role"]
        self.agent.configuration.goals = agent_goals["agent_goals"]


class PerformanceEvaluator:
    """Score ability results based on success, cost, and duration."""

    def __init__(
        self,
        success_weight: float = 1.0,
        cost_weight: float = 0.1,
        duration_weight: float = 0.1,
    ) -> None:
        self._success_weight = success_weight
        self._cost_weight = cost_weight
        self._duration_weight = duration_weight

    def score(self, result: AbilityResult, cost: float, duration: float) -> float:
        success_score = 1.0 if result.success else 0.0
        return (
            self._success_weight * success_score
            - self._cost_weight * cost
            - self._duration_weight * duration
        )


class SimpleAgent(LayeredAgent, Configurable):
    default_settings = AgentSystemSettings(
        name="simple_agent",
        description="A simple agent.",
        configuration=AgentConfiguration(
            name="Entrepreneur-GPT",
            role=(
                "An AI designed to autonomously develop and run businesses with "
                "the sole goal of increasing your net worth."
            ),
            goals=[
                "Increase net worth",
                "Grow Twitter Account",
                "Develop and manage multiple businesses autonomously",
            ],
            cycle_count=0,
            max_task_cycle_count=3,
            creation_time="",
            systems=AgentSystems(
                ability_registry=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.ability.SimpleAbilityRegistry",
                ),
                memory=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.memory.SimpleMemory",
                ),
                cognition=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.agent.cognition.SimpleBrainAdapter",
                ),
                openai_provider=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route=(
                        "autogpt.core.resource.model_providers.OpenAIProvider"
                    ),
                ),
                planning=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.planning.SimplePlanner",
                ),
                creative_planning=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.planning.CreativePlanner",
                ),
                workspace=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.workspace.SimpleWorkspace",
                ),
            ),
            self_assess_frequency=5,
        ),
    )

    def __init__(
        self,
        settings: AgentSystemSettings,
        logger: logging.Logger,
        ability_registry: SimpleAbilityRegistry,
        memory: SimpleMemory,
        model_providers: dict[str | ModelProviderName, ChatModelProvider],
        planning: SimplePlanner | None,
        workspace: SimpleWorkspace,
        cognition: SimpleBrainAdapter | None = None,
        creative_planning: SimplePlanner | None = None,
        next_layer: Optional[LayeredAgent] = None,
        optimize_abilities: bool = False,
        world_model: "WorldModel | None" = None,
        world_model_hook: Optional[Callable[[Callable[["WorldModel"], None]], None]] = None,
    ):
        super().__init__(next_layer=next_layer)
        self._configuration = settings.configuration
        self._logger = logger
        self._ability_registry = ability_registry
        self._memory = memory
        self._model_providers = model_providers
        self._planning = planning
        self._default_planning = planning
        self._creative_planning = creative_planning
        self._workspace = workspace
        self._knowledge_acquisition = KnowledgeAcquisitionManager()
        self._self_teacher = SelfTeacher(
            logger=logger,
            world_model=world_model,
            world_model_hook=world_model_hook,
        )
        self._cognition = cognition
        self._cognition_metadata: dict[str, Any] | None = None
        self._task_queue: list[tuple[int, Task]] = []
        self._completed_tasks = []
        self._current_task = None
        self._next_ability = None
        self._performance_evaluator = PerformanceEvaluator()
        # Minimum score required for a task to be accepted into the queue
        self._task_quality_threshold: float = 0.5
        self._optimize_abilities = optimize_abilities
        self._ability_metrics: dict[str, list[float]] = {}
        self._action_logger = ActionLogger(
            self._workspace.root / "logs" / "actions.jsonl"
        )
        self._global_workspace = GlobalWorkspace()
        self._symbolic_reasoner: SymbolicReasoner | None = None
        self._commonsense_validator: CommonsenseValidator | None = None
        self._setup_reasoning_modules()
        self._memory_pointer = 0

    def _use_neuromorphic_backend(self) -> bool:
        return self._cognition is not None and not self._model_providers

    @classmethod
    def _init_workspace(
        cls, agent_settings: AgentSettings, logger: logging.Logger
    ) -> SimpleWorkspace:
        """Instantiate the workspace plugin.

        Parameters
        ----------
        agent_settings
            Settings loaded from the workspace directory.
        logger
            Logger used by the agent.

        Returns
        -------
        SimpleWorkspace
            The workspace instance created from the settings.
        """

        return cls._get_system_instance("workspace", agent_settings, logger)

    # ------------------------------------------------------------------
    # Reasoning integration
    # ------------------------------------------------------------------
    def _setup_reasoning_modules(self) -> None:
        graph = self._load_reasoning_graph()
        self._symbolic_reasoner = SymbolicReasoner(graph)
        knowledge = CommonsenseKnowledge()
        for subject, targets in graph.items():
            for target in targets:
                knowledge.add_fact(subject, "related_to", target, truth=True)
        self._commonsense_validator = CommonsenseValidator(knowledge)
        set_symbolic_reasoner(self._symbolic_reasoner)
        set_commonsense_validator(self._commonsense_validator)
        set_causal_reasoner(self._symbolic_reasoner.causal)
        self._global_workspace.register_observer(LogicConstraintObserver(self._symbolic_reasoner))
        self._global_workspace.register_observer(CausalObserver(self._symbolic_reasoner.causal))
        self._global_workspace.register_observer(CommonsenseObserver(self._commonsense_validator))

    def _load_reasoning_graph(self) -> dict[str, list[str]]:
        adjacency: dict[str, list[str]] = {}
        try:
            graph_store = get_graph_store_instance()
            data = graph_store.query()
            edges = data.get("edges", [])
            for edge in edges:
                source = getattr(edge, "source", None)
                target = getattr(edge, "target", None)
                if not source or not target:
                    continue
                bucket = adjacency.setdefault(source, [])
                if target not in bucket:
                    bucket.append(target)
        except Exception as err:  # pragma: no cover - defensive logging
            self._logger.debug("Failed to build reasoning graph: %s", err, exc_info=True)
        return adjacency

    @classmethod
    def _init_model_providers(
        cls, agent_settings: AgentSettings, logger: logging.Logger
    ) -> dict[str | ModelProviderName, ChatModelProvider]:
        """Create model provider instances from settings.

        Each provider configuration is converted into an instantiated
        :class:`ChatModelProvider` and keyed by either its string name or
        :class:`ModelProviderName` enum.

        Parameters
        ----------
        agent_settings
            Settings containing provider configurations.
        logger
            Logger used by the agent.

        Returns
        -------
        dict[str | ModelProviderName, ChatModelProvider]
            Mapping of provider identifiers to provider instances.
        """

        providers: dict[str | ModelProviderName, ChatModelProvider] = {}
        for name, provider_settings in agent_settings.openai_providers.items():
            key: str | ModelProviderName = name
            try:
                key = ModelProviderName(name)
            except Exception:
                # Leave key as string if not a valid ModelProviderName
                pass
            providers[key] = cls._get_system_instance(
                "openai_provider",
                agent_settings,
                logger,
                system_settings=provider_settings,
            )
        return providers

    @classmethod
    def _init_planners(
        cls,
        agent_settings: AgentSettings,
        logger: logging.Logger,
        model_providers: dict[str | ModelProviderName, ChatModelProvider],
    ) -> tuple[SimplePlanner, SimplePlanner | None]:
        """Initialize the planning systems.

        Parameters
        ----------
        agent_settings
            Settings containing planner configurations.
        logger
            Logger used by the agent.
        model_providers
            Mapping of model providers to be passed to planners.

        Returns
        -------
        tuple[SimplePlanner, SimplePlanner | None]
            A tuple containing the default planner and an optional creative
            planner.
        """

        if not model_providers:
            return None, None

        planning = cls._get_system_instance(
            "planning",
            agent_settings,
            logger,
            model_providers=model_providers,
        )
        creative_planning = cls._get_system_instance(
            "creative_planning",
            agent_settings,
            logger,
            model_providers=model_providers,
        )
        return planning, creative_planning

    @classmethod
    def _init_memory(
        cls,
        agent_settings: AgentSettings,
        logger: logging.Logger,
        workspace: SimpleWorkspace,
    ) -> SimpleMemory:
        """Load the memory backend.

        Parameters
        ----------
        agent_settings
            Settings with memory configuration.
        logger
            Logger used by the agent.
        workspace
            Workspace instance used to store memory data.

        Returns
        -------
        SimpleMemory
            Instantiated memory backend.
        """

        return cls._get_system_instance(
            "memory", agent_settings, logger, workspace=workspace
        )

    @classmethod
    def _init_ability_registry(
        cls,
        agent_settings: AgentSettings,
        logger: logging.Logger,
        workspace: SimpleWorkspace,
        memory: SimpleMemory,
        model_providers: dict[str | ModelProviderName, ChatModelProvider],
    ) -> SimpleAbilityRegistry:
        """Instantiate the ability registry.

        Parameters
        ----------
        agent_settings
            Settings with ability registry configuration.
        logger
            Logger used by the agent.
        workspace
            Workspace instance required by some abilities.
        memory
            Memory backend to store ability information.
        model_providers
            Mapping of model providers used by abilities.

        Returns
        -------
        SimpleAbilityRegistry
            The ability registry populated according to settings.
        """

        return cls._get_system_instance(
            "ability_registry",
            agent_settings,
            logger,
            workspace=workspace,
            memory=memory,
            model_providers=model_providers,
        )

    @classmethod
    def _init_cognition(
        cls, agent_settings: AgentSettings, logger: logging.Logger
    ) -> SimpleBrainAdapter | None:
        try:
            return cls._get_system_instance(
                "cognition", agent_settings, logger
            )
        except KeyError:
            return None

    @classmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
        optimize_abilities: bool = False,
        world_model: "WorldModel | None" = None,
        world_model_hook: Optional[Callable[[Callable[["WorldModel"], None]], None]] = None,
    ) -> "SimpleAgent":
        """Create an agent from settings stored in a workspace directory.

        Parameters
        ----------
        workspace_path
            Path to the workspace containing ``agent_settings.json``.
        logger
            Logger used by the agent.
        optimize_abilities
            Whether to track ability metrics for optimisation.

        Returns
        -------
        SimpleAgent
            Fully assembled agent instance.
        """

        agent_settings = SimpleWorkspace.load_agent_settings(workspace_path)

        # Initialize core systems from the workspace configuration
        workspace = cls._init_workspace(agent_settings, logger)
        model_providers = cls._init_model_providers(agent_settings, logger)
        planning, creative_planning = cls._init_planners(
            agent_settings, logger, model_providers
        )
        memory = cls._init_memory(agent_settings, logger, workspace)
        ability_registry = cls._init_ability_registry(
            agent_settings, logger, workspace, memory, model_providers
        )
        cognition = cls._init_cognition(agent_settings, logger)

        return cls(
            settings=agent_settings.agent,
            logger=logger,
            ability_registry=ability_registry,
            memory=memory,
            model_providers=model_providers,
            planning=planning,
            workspace=workspace,
            cognition=cognition,
            creative_planning=creative_planning,
            optimize_abilities=optimize_abilities,
            world_model=world_model,
            world_model_hook=world_model_hook,
        )

    def use_creative_planner(self, use_creative: bool = True) -> None:
        """Switch between the default planner and the creative planner."""
        if use_creative and self._creative_planning is not None:
            self._planning = self._creative_planning
        else:
            self._planning = self._default_planning

    def _evaluate_task_quality(self, task: Task) -> float:
        """Evaluate how well-defined a task is.

        A task is considered high quality when it contains both ready and
        acceptance criteria. The :class:`PerformanceEvaluator` is re-used here to
        produce a numeric score that can be compared against a threshold.
        """

        result = AbilityResult(
            ability_name="task_quality_evaluation",
            ability_args={},
            success=bool(task.ready_criteria and task.acceptance_criteria),
            message="",
        )
        # Cost and duration are unknown for a task that has not yet executed, so
        # we pass zero for those values.
        return self._performance_evaluator.score(result, cost=0.0, duration=0.0)

    async def build_initial_plan(self) -> dict:
        if self._use_neuromorphic_backend() and self._cognition is not None:
            plan_dict, metadata = await self._cognition.build_initial_plan(
                agent_name=self._configuration.name,
                agent_role=self._configuration.role,
                agent_goals=self._configuration.goals,
                ability_specs=self._ability_registry.list_abilities(),
            )
            self._cognition_metadata = metadata
        else:
            if self._planning is None:
                raise RuntimeError("Planner is not configured for GPT-based planning.")
            plan = await self._planning.make_initial_plan(
                agent_name=self._configuration.name,
                agent_role=self._configuration.role,
                agent_goals=self._configuration.goals,
                abilities=self._ability_registry.list_abilities(),
            )
            plan_dict = plan.parsed_result
            self._cognition_metadata = None
        if "plan_options" in plan_dict:
            plan_dict = plan_dict["plan_options"][0]
        tasks = [Task.parse_obj(task) for task in plan_dict["task_list"]]

        # Evaluate each task and only enqueue those that meet a minimum quality
        for task in tasks:
            score = self._evaluate_task_quality(task)
            if score < self._task_quality_threshold:
                self._logger.info(
                    f"Rejecting low-quality task '{task.objective}' with score {score:.2f}"
                )
                continue
            heapq.heappush(self._task_queue, (task.priority - score, task))
        if self._task_queue:
            self._task_queue[0][1].context.status = TaskStatus.READY
        return plan_dict

    def route_task(self, task: Task, *args, **kwargs):
        heapq.heappush(self._task_queue, (task.priority, task))
        self._task_queue[0][1].context.status = TaskStatus.READY
        return task

    async def determine_next_ability(self, *args, **kwargs):
        if not self._task_queue:
            return {"response": "I don't have any tasks to work on right now."}

        self._configuration.cycle_count += 1
        _, task = heapq.heappop(self._task_queue)
        self._logger.info(f"Working on task: {task}")

        task = await self._evaluate_task_and_add_context(task)
        ability_specs = self._ability_registry.dump_abilities()
        self._global_workspace.update_state(
            goal=self._configuration.goals[0] if self._configuration.goals else "",
            memory_pointer=self._memory_pointer,
        )
        ability_selection = await self._choose_next_ability(
            task, ability_specs, self._global_workspace.get_context()
        )
        self._current_task = task
        if hasattr(ability_selection, "parsed_result"):
            parsed_selection = ability_selection.parsed_result
        else:
            parsed_selection = ability_selection
        if isinstance(parsed_selection, dict):
            parsed_selection = dict(parsed_selection)
        else:
            raise TypeError("Ability selection did not provide a mapping response.")
        if hasattr(self, '_knowledge_acquisition') and self._knowledge_acquisition:
            metadata = self._cognition_metadata or {}
            override = self._knowledge_acquisition.maybe_acquire(
                metadata=metadata,
                ability_specs=ability_specs,
                task=task,
                current_selection=parsed_selection,
            )
            if override:
                parsed_selection = override
                parsed_selection.setdefault("backend", "knowledge_acquisition")
                if isinstance(self._cognition_metadata, dict):
                    self._cognition_metadata["knowledge_triggered"] = True
                elif self._cognition_metadata is not None:
                    self._cognition_metadata = {"value": self._cognition_metadata, "knowledge_triggered": True}
                else:
                    self._cognition_metadata = {"knowledge_triggered": True}
        backend = None
        if isinstance(parsed_selection, dict):
            backend = parsed_selection.get("backend")
        if not backend:
            backend = "whole_brain" if self._use_neuromorphic_backend() else "language_model"
        if isinstance(parsed_selection, dict) and "backend" not in parsed_selection:
            parsed_selection["backend"] = backend
        self._next_ability = parsed_selection
        self._global_workspace.update_state(
            action=self._next_ability["next_ability"]
        )
        task_desc = getattr(task, "description", task.objective)
        task_id = str(getattr(task, "id", id(task)))
        feedback = self._global_workspace.broadcast(
            parsed_selection,
            context={
                "task_objective": task.objective,
                "task_description": task_desc,
            },
        )
        for signal in feedback:
            source = getattr(signal, "source", "observer")
            message = getattr(signal, "message", str(signal))
            self._logger.info("Workspace feedback (%s): %s", source, message)
        log_payload = {
            "event": "action_selected",
            "agent": self._configuration.name,
            "task_id": task_id,
            "task_description": task_desc,
            "cycle_count": task.context.cycle_count,
            "ability_options": [spec.name for spec in ability_specs],
            "chosen_action": self._next_ability["next_ability"],
            "backend": backend,
        }
        if self._cognition_metadata:
            log_payload["cognition"] = self._cognition_metadata
        self._action_logger.log(log_payload)
        return self._current_task, self._next_ability

    async def execute_next_ability(self, user_input: str, *args, **kwargs):
        if user_input == "y":
            ability_name = self._next_ability["next_ability"]
            ability_args = self._next_ability["ability_arguments"]
            predicted_action = ability_name
            ability = self._ability_registry.get_ability(ability_name)

            filename = (
                ability_args.get("filename") if ability_name == "write_file" else None
            )
            knowledge_session_id = None
            if isinstance(self._next_ability, dict):
                knowledge_session_id = self._next_ability.get("knowledge_session_id")
            if knowledge_session_id and getattr(self, "_knowledge_acquisition", None):
                self._knowledge_acquisition.mark_session_started(knowledge_session_id)
            start_time = time.perf_counter()
            ability_response = await ability(**ability_args)
            duration = time.perf_counter() - start_time
            cost = float(ability_response.ability_args.get("cost", 0))
            self._ability_metrics.setdefault(ability_name, []).append(duration)
            hint = getattr(
                getattr(ability, "_configuration", None),
                "performance_hint",
                None,
            )
            if (
                self._optimize_abilities
                and hint is not None
                and duration > hint
            ):
                self._ability_registry.optimize_ability(
                    ability_name, {"duration": duration}
                )

            if ability_name == "write_file" and ability_response.success:
                lint_ability = self._ability_registry.get_ability("lint_code")
                lint_result = await lint_ability(file_path=filename)
                self._logger.info(
                    f"Static analysis for {filename}: {lint_result.message}"
                )
                self._memory.add(
                    f"Static analysis for {filename}: {lint_result.message}"
                )
                ability_response.message += f" Lint: {lint_result.message}"
                if not lint_result.success:
                    subprocess.run(["git", "checkout", "--", filename], check=False)
                    if hasattr(self._workspace, "refresh"):
                        self._workspace.refresh()
                    ability_response.message += (
                        " Static analysis failed. Changes reverted."
                    )
                else:
                    generate_tests = self._ability_registry.get_ability(
                        "generate_tests"
                    )
                    tests_code = await generate_tests(file_path=filename)
                    test_filename = None
                    if tests_code.success and tests_code.message:
                        test_filename = str(
                            Path("tests") / f"test_{Path(filename).stem}.py"
                        )
                        await ability(
                            filename=test_filename, contents=tests_code.message
                        )
                        ability_response.message += (
                            f" Test file generated at {test_filename}."
                        )
                    run_tests = self._ability_registry.get_ability("run_tests")
                    tests_result = await run_tests()
                    critique = await self._ability_registry.perform(
                        "query_language_model",
                        query=(
                            "Given these test results, provide feedback on "
                            "the change:\n"
                            + tests_result.message
                        ),
                    )
                    test_status = "passed" if tests_result.success else "failed"
                    if not tests_result.success:
                        self._memory.add(
                            f"Test failed for {filename}:\n{tests_result.message}\n"
                            f"Critique: {critique.message}"
                        )
                        subprocess.run(["git", "checkout", "--", filename], check=False)
                        if test_filename:
                            subprocess.run(
                                ["git", "checkout", "--", test_filename], check=False
                            )
                            if subprocess.run(
                                ["git", "ls-files", test_filename, "--error-unmatch"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            ).returncode:
                                Path(test_filename).unlink(missing_ok=True)
                        if hasattr(self._workspace, "refresh"):
                            self._workspace.refresh()
                        ability_response.message += " Tests failed. Changes reverted."
                    else:
                        ability_response.message += " Tests passed."
                        evaluate_metrics = self._ability_registry.get_ability(
                            "evaluate_metrics"
                        )
                        metrics_result = await evaluate_metrics(file_path=filename)
                        self._memory.add(
                            f"Metrics for {filename}: {metrics_result.message}"
                        )
                        ability_response.message += (
                            f" Metrics: {metrics_result.message}"
                        )
                        subprocess.run(["git", "add", filename], check=False)
                        if test_filename:
                            subprocess.run(["git", "add", test_filename], check=False)
                        diff = subprocess.check_output(
                            ["git", "diff", "--cached"], text=True
                        )
                        summary = await self._ability_registry.perform(
                            "query_language_model",
                            query=(
                                "Summarize the following changes for ",
                                "a commit message:\n"
                                + diff
                            ),
                        )
                        commit_message = (
                            f"Auto-update: {summary.message}"
                            if summary.success and summary.message
                            else f"Auto-update: modify {filename}"
                        )
                        subprocess.run(
                            ["git", "commit", "-m", commit_message], check=False
                        )
                        commit_hash = subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], text=True
                        ).strip()
                        self._memory.add(
                            f"Commit {commit_hash} - {commit_message} - ",
                            f"Test {test_status} for {filename}:\n",
                            f"{tests_result.message}\nCritique: {critique.message}",
                        )

            knowledge_log = None
            if knowledge_session_id and getattr(self, "_knowledge_acquisition", None):
                knowledge_log = self._knowledge_acquisition.complete_session(
                    knowledge_session_id,
                    ability_response,
                    metadata=self._cognition_metadata,
                    memory=self._memory,
                )
                if knowledge_log:
                    self._action_logger.log(dict(knowledge_log))

            await self._update_tasks_and_memory(ability_response)
            task_desc = getattr(
                self._current_task,
                "description",
                self._current_task.objective,
            )
            task_id = str(
                getattr(self._current_task, "id", id(self._current_task))
            )
            score = self._performance_evaluator.score(
                ability_response, cost=cost, duration=duration
            )
            self._memory.log_score(
                task_id=task_id,
                task_description=task_desc,
                ability=ability_name,
                score=score,
            )
            self._global_workspace.reflect(predicted_action, ability_name)
            self._memory_pointer += 1
            self._action_logger.log(
                {
                    "event": "action_executed",
                    "agent": self._configuration.name,
                    "task_id": task_id,
                    "task_description": task_desc,
                    "action": ability_name,
                    "metrics": {
                        "success": ability_response.success,
                        "cost": cost,
                        "duration": duration,
                        "score": score,
                    },
                }
            )
            try:
                if getattr(self, '_self_teacher', None) is not None:
                    await self._self_teacher.maybe_run(
                        ability_registry=self._ability_registry,
                        memory=self._memory,
                        knowledge_acquisition=self._knowledge_acquisition,
                        cognition_metadata=self._cognition_metadata,
                    )
            except Exception:
                self._logger.debug('Self-teacher cycle failed.', exc_info=True)
            if (
                self._configuration.cycle_count
                % self._configuration.self_assess_frequency
                == 0
            ):
                assessment = await self._ability_registry.perform(
                    "self_assess", limit=5
                )
                self._memory.add(f"Self-assessment: {assessment.message}")
            if self._current_task.context.status == TaskStatus.DONE:
                self._completed_tasks.append(self._current_task)
            else:
                heapq.heappush(
                    self._task_queue,
                    (self._current_task.priority, self._current_task),
                )
                self._task_queue[0][1].context.status = TaskStatus.READY
            self._current_task = None
            self._next_ability = None

            return ability_response.dict()
        else:
            raise NotImplementedError

    async def _evaluate_task_and_add_context(self, task: Task) -> Task:
        """Evaluate the task and add context to it."""
        if task.context.status == TaskStatus.IN_PROGRESS:
            # Nothing to do here
            return task
        else:
            self._logger.debug(f"Evaluating task {task} and adding relevant context.")
            query = getattr(task, "description", task.objective)
            k = 5
            try:
                config = Config()
                relevant_memories = self._memory.get_relevant(query, k, config)

                class _ContextMemory:
                    def __init__(self, item):
                        self._item = item

                    def summary(self) -> str:
                        return self._item.summary

                for memory in relevant_memories:
                    task.context.memories.append(_ContextMemory(memory.memory_item))
            except Exception as e:
                self._logger.debug(f"Failed to get relevant memories: {e}")

            task.context.enough_info = True
            task.context.status = TaskStatus.IN_PROGRESS
            return task

    async def _choose_next_ability(
        self,
        task: Task,
        ability_specs: list[CompletionModelFunction],
        state_context: dict[str, Any] | None = None,
    ):
        """Choose the next ability to use for the task."""
        self._logger.debug(f"Choosing next ability for task {task}.")

        degraded_modules = self._get_degraded_modules()
        if degraded_modules:
            module_path = degraded_modules[0]
            self._cognition_metadata = None
            return ChatModelResponse(
                response=AssistantChatMessage(
                    content="evaluate_metrics due to performance regression"
                ),
                parsed_result={
                    "next_ability": "evaluate_metrics",
                    "ability_arguments": {"file_path": module_path},
                },
            )

        if self._use_neuromorphic_backend() and self._cognition is not None:
            selection, metadata = await self._cognition.determine_next_ability(
                agent_name=self._configuration.name,
                agent_role=self._configuration.role,
                agent_goals=self._configuration.goals,
                task=task,
                ability_specs=ability_specs,
                cycle_index=self._configuration.cycle_count,
                backlog_size=len(self._task_queue),
                completed=len(self._completed_tasks),
                state_context=state_context or {},
            )
            self._cognition_metadata = metadata
            return selection

        if task.context.cycle_count > self._configuration.max_task_cycle_count:
            # Don't hit the LLM, just set the next action as "breakdown_task"
            #  with an appropriate reason
            raise NotImplementedError
        elif not task.context.enough_info:
            # Don't ask the LLM, just set the next action as "breakdown_task"
            #  with an appropriate reason
            raise NotImplementedError
        else:
            if self._planning is None:
                raise RuntimeError("Planner is not configured for GPT-based ability selection.")
            task_desc = getattr(task, "description", task.objective)
            ability_specs.sort(
                key=lambda spec: self._average_score(task_desc, spec.name),
                reverse=True,
            )
            next_ability = await self._planning.determine_next_ability(
                task, ability_specs, state_context=state_context
            )
            self._cognition_metadata = None
            return next_ability

    def _average_score(self, task_desc: str, ability_name: str) -> float:
        scores = self._memory.get_scores_for_task(task_desc, ability_name)
        return sum(scores) / len(scores) if scores else 0.0

    def _get_degraded_modules(self) -> list[str]:
        """Parse memory for performance metrics and return modules with regressions."""
        records: dict[str, list[tuple[float, float]]] = {}
        for msg in self._memory.get():
            match = re.match(
                r"Metrics for (.*): complexity=([0-9.]+), runtime=([0-9.]+)",
                msg,
            )
            if match:
                file = match.group(1)
                complexity = float(match.group(2))
                runtime = float(match.group(3))
                records.setdefault(file, []).append((complexity, runtime))

        degraded = []
        for file, values in records.items():
            if len(values) >= 2:
                prev_c, prev_r = values[-2]
                curr_c, curr_r = values[-1]
                if curr_c > prev_c or curr_r > prev_r:
                    degraded.append(file)
        return degraded

    async def _update_tasks_and_memory(self, ability_result: AbilityResult):
        self._current_task.context.cycle_count += 1
        self._current_task.context.prior_actions.append(ability_result)

        if ability_result.ability_name == "evaluate_metrics":
            file_path = ability_result.ability_args.get("file_path", "")
            self._memory.add(f"Metrics for {file_path}: {ability_result.message}")

        # --- Summarize and extract knowledge ---
        summary, knowledge_items = extract(ability_result.message or "")

        if summary:
            # store in global memory and task specific memories
            self._memory.add(summary)
            self._current_task.context.memories.append(summary)

        if knowledge_items:
            # attach structured knowledge to the current task for later reference
            self._current_task.context.supplementary_info.append(knowledge_items)

            kb_path = self._workspace.get_path("knowledge_base.json")
            if kb_path.exists():
                with kb_path.open("r") as f:
                    existing = json.load(f)
            else:
                existing = []
            updated = resolve_conflicts(existing, knowledge_items)
            with kb_path.open("w") as f:
                json.dump(updated, f, indent=2)

        # --- Determine if the task has been completed ---
        # First check if an explicit status was provided by the ability result
        status_arg = ability_result.ability_args.get("status")
        if status_arg:
            try:
                status = (
                    status_arg
                    if isinstance(status_arg, TaskStatus)
                    else TaskStatus(status_arg)
                )
            except Exception:
                status = None
            if status == TaskStatus.DONE:
                self._current_task.context.status = TaskStatus.DONE

        # If no explicit status, infer completion from acceptance criteria
        if (
            self._current_task.context.status != TaskStatus.DONE
            and self._current_task.acceptance_criteria
        ):
            message_text = ability_result.message or ""
            if all(
                criterion.lower() in message_text.lower()
                for criterion in self._current_task.acceptance_criteria
            ):
                self._current_task.context.status = TaskStatus.DONE

        # Move any tasks marked as done out of the queue
        remaining_queue: list[tuple[int, Task]] = []
        for priority, task in self._task_queue:
            if task.context.status == TaskStatus.DONE:
                self._completed_tasks.append(task)
            else:
                remaining_queue.append((priority, task))
        if len(remaining_queue) != len(self._task_queue):
            heapq.heapify(remaining_queue)
            self._task_queue = remaining_queue

    def __repr__(self):
        return "SimpleAgent()"

    ################################################################
    # Factory interface for agent bootstrapping and initialization #
    ################################################################

    @classmethod
    def build_user_configuration(cls) -> dict[str, Any]:
        """Build the user's configuration."""
        configuration_dict = {
            "agent": cls.get_user_config(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]
        for system_name, system_location in system_locations.items():
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.get_user_config()
        configuration_dict = _prune_empty_dicts(configuration_dict)
        return configuration_dict

    @classmethod
    def compile_settings(
        cls, logger: logging.Logger, user_configuration: dict
    ) -> AgentSettings:
        """Compile the user's configuration with the defaults."""
        logger.debug("Processing agent system configuration.")
        configuration_dict = {
            "agent": cls.build_agent_configuration(
                user_configuration.get("agent", {})
            ).dict(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]

        # Build up default configuration
        for system_name, system_location in system_locations.items():
            logger.debug(f"Compiling configuration for system {system_name}")
            system_class = SimplePluginService.get_plugin(system_location)
            if system_name == "openai_provider":
                providers_cfg = user_configuration.get(system_name, {})
                if providers_cfg:
                    configuration_dict["openai_providers"] = {
                        name: system_class.build_agent_configuration(cfg).dict()
                        for name, cfg in providers_cfg.items()
                    }
                else:
                    configuration_dict["openai_providers"] = {}
            else:
                configuration_dict[system_name] = system_class.build_agent_configuration(
                    user_configuration.get(system_name, {})
                ).dict()

        return AgentSettings.parse_obj(configuration_dict)

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ) -> dict:
        logger.debug("Loading model providers.")
        providers: dict[str | ModelProviderName, ChatModelProvider] = {}
        for name, provider_settings in agent_settings.openai_providers.items():
            key = name
            try:
                key = ModelProviderName(name)
            except Exception:
                pass
            providers[key] = cls._get_system_instance(
                "openai_provider",
                agent_settings,
                logger=logger,
                system_settings=provider_settings,
            )
        logger.debug("Loading agent planner.")
        if not providers:
            logger.debug("No GPT providers configured; using default agent identity.")
            configuration = agent_settings.agent.configuration
            return {
                "agent_name": configuration.name,
                "agent_role": configuration.role,
                "agent_goals": configuration.goals,
            }
        agent_planner: SimplePlanner = cls._get_system_instance(
            "planning",
            agent_settings,
            logger=logger,
            model_providers=providers,
        )
        logger.debug("determining agent name and goals.")
        model_response = await agent_planner.decide_name_and_goals(
            user_objective,
        )

        return model_response.parsed_result

    @classmethod
    def provision_agent(
        cls,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ):
        agent_settings.agent.configuration.creation_time = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        workspace: SimpleWorkspace = cls._get_system_instance(
            "workspace",
            agent_settings,
            logger=logger,
        )
        return workspace.setup_workspace(agent_settings, logger)

    @classmethod
    def _get_system_instance(
        cls,
        system_name: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
        *args,
        system_settings=None,
        **kwargs,
    ):
        system_locations = agent_settings.agent.configuration.systems.dict()
        if system_settings is None:
            system_settings = getattr(agent_settings, system_name)
        system_class = SimplePluginService.get_plugin(system_locations[system_name])
        system_instance = system_class(
            system_settings,
            *args,
            logger=logger.getChild(system_name),
            **kwargs,
        )
        return system_instance


def _prune_empty_dicts(d: dict) -> dict:
    """
    Prune branches from a nested dictionary if the branch only contains empty
    dictionaries at the leaves.

    Args:
        d: The dictionary to prune.

    Returns:
        The pruned dictionary.
    """
    pruned = {}
    for key, value in d.items():
        if isinstance(value, dict):
            pruned_value = _prune_empty_dicts(value)
            if (
                pruned_value
            ):  # if the pruned dictionary is not empty, add it to the result
                pruned[key] = pruned_value
        else:
            pruned[key] = value
    return pruned
