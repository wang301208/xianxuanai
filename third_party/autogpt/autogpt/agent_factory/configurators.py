from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from forge.sdk.model import Task

from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIDirectives, AIProfile, Config
from autogpt.core.resource.model_providers import ChatModelProvider
from autogpt.file_storage.base import FileStorage
from autogpt.logs.config import configure_chat_plugins
from autogpt.models.command_registry import CommandRegistry
from autogpt.plugins import scan_plugins
from autogpt.memory.vector import get_memory
from modules.brain.backends import BrainBackendProtocol

if TYPE_CHECKING:
    from knowledge import UnifiedKnowledgeBase
    from reasoning import DecisionEngine

    from autogpt.core.brain.transformer_brain import TransformerBrain


def create_agent(
    agent_id: str,
    task: Task,
    ai_profile: AIProfile,
    app_config: Config,
    file_storage: FileStorage,
    llm_provider: ChatModelProvider,
    directives: Optional[AIDirectives] = None,
    brain: "TransformerBrain" | None = None,
    whole_brain: BrainBackendProtocol | None = None,
    knowledge_base: "UnifiedKnowledgeBase" | None = None,
    decision_engine: "DecisionEngine" | None = None,
) -> Agent:
    if not task or not task.input:
        raise ValueError("No task specified for new agent")
    if not directives:
        directives = AIDirectives.from_file(app_config.prompt_settings_file)

    agent = _configure_agent(
        agent_id=agent_id,
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        app_config=app_config,
        file_storage=file_storage,
        llm_provider=llm_provider,
        brain=brain,
        whole_brain=whole_brain,
        knowledge_base=knowledge_base,
        decision_engine=decision_engine,
    )

    return agent


def configure_agent_with_state(
    state: AgentSettings,
    app_config: Config,
    file_storage: FileStorage,
    llm_provider: ChatModelProvider,
    brain: "TransformerBrain" | None = None,
    whole_brain: BrainBackendProtocol | None = None,
    knowledge_base: "UnifiedKnowledgeBase" | None = None,
    decision_engine: "DecisionEngine" | None = None,
) -> Agent:
    return _configure_agent(
        state=state,
        app_config=app_config,
        file_storage=file_storage,
        llm_provider=llm_provider,
        brain=brain,
        whole_brain=whole_brain,
        knowledge_base=knowledge_base,
        decision_engine=decision_engine,
    )


def _configure_agent(
    app_config: Config,
    llm_provider: ChatModelProvider,
    file_storage: FileStorage,
    agent_id: str = "",
    task: Task | None = None,
    ai_profile: Optional[AIProfile] = None,
    directives: Optional[AIDirectives] = None,
    state: Optional[AgentSettings] = None,
    brain: "TransformerBrain" | None = None,
    whole_brain: BrainBackendProtocol | None = None,
    knowledge_base: "UnifiedKnowledgeBase" | None = None,
    decision_engine: "DecisionEngine" | None = None,
) -> Agent:
    if not (state or (agent_id and task and ai_profile and directives)):
        raise TypeError(
            "Either (state) or (agent_id, task, ai_profile, directives)"
            " must be specified"
        )

    app_config.plugins = scan_plugins(app_config)
    configure_chat_plugins(app_config)

    # Create a CommandRegistry instance and scan default folder
    command_registry = CommandRegistry.with_command_modules(
        modules=COMMAND_CATEGORIES,
        config=app_config,
    )

    agent_state = state or create_agent_state(
        agent_id=agent_id,
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        app_config=app_config,
    )
    # Configure vector memory based on app configuration
    # Ensure the workspace path is available for memory backends
    object.__setattr__(
        app_config,
        "workspace_path",
        getattr(app_config, "workspace_path", file_storage.root),
    )
    memory = get_memory(app_config)

    agent = Agent(
        settings=agent_state,
        llm_provider=llm_provider,
        command_registry=command_registry,
        file_storage=file_storage,
        legacy_config=app_config,
        brain=brain,
        whole_brain=whole_brain,
        knowledge_base=knowledge_base,
        decision_engine=decision_engine,
    )

    # Attach memory to the agent instance
    agent.memory = memory

    return agent


def create_agent_state(
    agent_id: str,
    task: Task,
    ai_profile: AIProfile,
    directives: AIDirectives,
    app_config: Config,
) -> AgentSettings:
    agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
    agent_prompt_config.use_functions_api = app_config.openai_functions

    return AgentSettings(
        agent_id=agent_id,
        workspace_id=agent_id,
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        config=AgentConfiguration(
            fast_llm=app_config.fast_llm,
            smart_llm=app_config.smart_llm,
            allow_fs_access=not app_config.restrict_to_workspace,
            use_functions_api=app_config.openai_functions,
            plugins=app_config.plugins,
            use_transformer_brain=app_config.use_transformer_brain,
            brain_backend=app_config.brain_backend,
            whole_brain=app_config.whole_brain.copy(deep=True),
            brain_simulation=app_config.brain_simulation.copy(deep=True),
            use_knowledge_base=app_config.use_knowledge_base,
            use_decision_engine=app_config.use_decision_engine,
        ),
        prompt_config=agent_prompt_config,
        history=Agent.default_settings.history.copy(deep=True),
    )
