import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

import pytest


@dataclass
class Task:
    input: str
    additional_input: Optional[dict]
    created_at: datetime
    modified_at: datetime
    task_id: str
    artifacts: List[str]


from autogpt.agent_manager import AgentManager
from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config.ai_directives import AIDirectives
from autogpt.config.ai_profile import AIProfile
from autogpt.models.command_registry import CommandRegistry


@pytest.mark.asyncio
async def test_multiple_agents_shared_workspace(config, storage, llm_provider):
    directives = AIDirectives.from_file(config.prompt_settings_file)
    ai_profile = AIProfile(ai_name="Tester", ai_role="testing", ai_goals=[])

    def make_task():
        now = datetime.now()
        return Task(
            input="Test task",
            additional_input=None,
            created_at=now,
            modified_at=now,
            task_id=str(uuid4()),
            artifacts=[],
        )

    def build_agent(agent_id: str) -> Agent:
        task = make_task()
        agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
        agent_prompt_config.use_functions_api = config.openai_functions
        settings = AgentSettings(
            agent_id=agent_id,
            workspace_id="shared",
            name=Agent.default_settings.name,
            description=Agent.default_settings.description,
            task=task,
            ai_profile=ai_profile,
            directives=directives,
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
        command_registry = CommandRegistry.with_command_modules(
            modules=COMMAND_CATEGORIES, config=config
        )
        return Agent(
            settings=settings,
            llm_provider=llm_provider,
            command_registry=command_registry,
            file_storage=storage,
            legacy_config=config,
        )

    agent1 = build_agent("agent1")
    agent2 = build_agent("agent2")

    await asyncio.gather(
        agent1.workspace.write_file("file1.txt", "agent1"),
        agent2.workspace.write_file("file2.txt", "agent2"),
    )

    await asyncio.gather(agent1.save_state(), agent2.save_state())

    manager = AgentManager(storage)

    def restore(agent_id: str) -> Agent:
        settings = manager.load_agent_state(agent_id)
        registry = CommandRegistry.with_command_modules(
            modules=COMMAND_CATEGORIES, config=config
        )
        return Agent(
            settings=settings,
            llm_provider=llm_provider,
            command_registry=registry,
            file_storage=storage,
            legacy_config=config,
        )

    restored1 = restore("agent1")
    restored2 = restore("agent2")

    assert restored1.workspace.read_file("file1.txt") == "agent1"
    assert restored1.workspace.read_file("file2.txt") == "agent2"
    assert restored2.workspace.read_file("file1.txt") == "agent1"
    assert restored2.workspace.read_file("file2.txt") == "agent2"
