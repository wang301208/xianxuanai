import logging
from dataclasses import dataclass

from agent_protocol import StepHandler

from autogpt.agent_manager.agent_manager import AgentManager
from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.agents.utils.exceptions import AgentTerminated
from autogpt.app.main import UserFeedback, _configure_openai_provider
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIProfile, ConfigBuilder
from autogpt.file_storage import FileStorageBackendName, get_storage
from autogpt.logs.config import configure_logging
from autogpt.models.command_registry import CommandRegistry


@dataclass
class StepResult:
    output: dict | None
    is_last: bool = False


async def task_handler(task_input) -> StepHandler:
    task = task_input.__root__ if task_input else {}
    agent = bootstrap_agent(task.get("user_input"), False)

    next_command_name: str | None = None
    next_command_args: dict[str, str] | None = None

    async def step_handler(step_input) -> StepResult:
        step = step_input.__root__ if step_input else {}

        nonlocal next_command_name, next_command_args

        result = await interaction_step(
            agent,
            step.get("user_input"),
            step.get("user_feedback"),
            next_command_name,
            next_command_args,
        )

        next_command_name = result["next_step_command_name"] if result else None
        next_command_args = result["next_step_command_args"] if result else None

        if not result:
            return StepResult(output=None, is_last=True)
        return StepResult(output=result)

    return step_handler


async def interaction_step(
    agent: Agent,
    user_input,
    user_feedback: UserFeedback | None,
    command_name: str | None,
    command_args: dict[str, str] | None,
):
    """Run one step of the interaction loop."""
    if user_feedback == UserFeedback.EXIT:
        return
    if user_feedback == UserFeedback.TEXT:
        command_name = "human_feedback"

    result: str | None = None

    if command_name is not None:
        try:
            result_obj = await agent.execute(
                command_name, command_args or {}, user_input or ""
            )
            result = str(result_obj)
        except AgentTerminated:
            return

    (
        next_command_name,
        next_command_args,
        assistant_reply_dict,
    ) = await agent.propose_action()

    return {
        "config": agent.config,
        "ai_profile": agent.ai_profile,
        "result": result,
        "assistant_reply_dict": assistant_reply_dict,
        "next_step_command_name": next_command_name,
        "next_step_command_args": next_command_args,
    }


def bootstrap_agent(task, continuous_mode) -> Agent:
    configure_logging(level=logging.DEBUG, plain_console_output=True)

    config = ConfigBuilder.build_config_from_env()
    config.logging.level = logging.DEBUG
    config.logging.plain_console_output = True
    config.continuous_mode = continuous_mode
    config.temperature = 0
    config.memory_backend = "no_memory"

    command_registry = CommandRegistry.with_command_modules(COMMAND_CATEGORIES, config)

    ai_profile = AIProfile(
        ai_name="AutoGPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[task],
    )

    agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
    agent_prompt_config.use_functions_api = config.openai_functions
    agent_settings = AgentSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        agent_id=AgentManager.generate_id("AutoGPT-cli"),
        ai_profile=ai_profile,
        config=AgentConfiguration(
            fast_llm=config.fast_llm,
            smart_llm=config.smart_llm,
            allow_fs_access=not config.restrict_to_workspace,
            use_functions_api=config.openai_functions,
            plugins=config.plugins,
        ),
        prompt_config=agent_prompt_config,
        history=Agent.default_settings.history.copy(deep=True),
    )

    local = config.file_storage_backend == FileStorageBackendName.LOCAL
    restrict_to_root = not local or config.restrict_to_workspace
    file_storage = get_storage(
        config.file_storage_backend, root_path="data", restrict_to_root=restrict_to_root
    )
    file_storage.initialize()

    agent = Agent(
        settings=agent_settings,
        llm_provider=_configure_openai_provider(config),
        command_registry=command_registry,
        file_storage=file_storage,
        legacy_config=config,
    )
    return agent
