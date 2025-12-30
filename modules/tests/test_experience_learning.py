import asyncio
import pytest
from unittest.mock import AsyncMock

from third_party.autogpt.autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from third_party.autogpt.autogpt.config import AIProfile, Config
from third_party.autogpt.autogpt.models.command import Command
from third_party.autogpt.autogpt.models.command_registry import CommandRegistry
from third_party.autogpt.autogpt.file_storage.local import LocalFileStorage, FileStorageConfiguration
from third_party.autogpt.autogpt.models.action_history import Action


class DummyProvider:
    def count_tokens(self, text: str, model_name: str) -> int:
        return len(text)

    def get_token_limit(self, model_name: str) -> int:
        return 1000

    def get_tokenizer(self, model_name: str):
        class _Tok:
            def encode(self, text):
                return list(text)

            def decode(self, tokens):
                return "".join(tokens)

        return _Tok()

    async def create_chat_completion(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_learning_disables_failing_command(tmp_path):
    cfg = Config()
    cfg.fast_llm = "gpt-3.5-turbo"
    cfg.smart_llm = "gpt-3.5-turbo"

    storage = LocalFileStorage(
        FileStorageConfiguration(root=tmp_path, restrict_to_root=False)
    )
    storage.initialize()

    command_registry = CommandRegistry()

    ai_profile = AIProfile(ai_name="Test", ai_role="Test", ai_goals=[])
    agent_settings = AgentSettings(
        name="Agent",
        description="",
        agent_id="test-agent",
        ai_profile=ai_profile,
        config=AgentConfiguration(
            fast_llm=cfg.fast_llm,
            smart_llm=cfg.smart_llm,
            allow_fs_access=True,
            plugins=[],
        ),
        prompt_config=Agent.default_settings.prompt_config.copy(deep=True),
        history=Agent.default_settings.history.copy(deep=True),
    )

    agent = Agent(
        settings=agent_settings,
        llm_provider=DummyProvider(),
        command_registry=command_registry,
        file_storage=storage,
        legacy_config=cfg,
    )
    agent.config.learning.enabled = True
    agent.config.learning.learning_rate = 1.0

    # avoid network calls during compression
    object.__setattr__(agent.event_history, "handle_compression", AsyncMock())

    def fail_command(*, agent):
        raise RuntimeError("fail")

    cmd = Command(
        name="fail",
        description="failing command",
        method=fail_command,
        parameters=[],
    )
    agent.command_registry.register(cmd)

    assert any(c.name == "fail" for c in agent.command_registry.list_available_commands(agent))

    agent.event_history.register_action(Action(name="fail", args={}, reasoning=""))
    await agent.execute("fail")

    assert all(c.name != "fail" for c in agent.command_registry.list_available_commands(agent))
