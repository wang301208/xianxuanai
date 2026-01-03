import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "third_party/autogpt"))
sys.path.insert(0, str(ROOT / "modules"))

# Provide a lightweight stub for the plugin template dependency
import types

auto_plugin = types.SimpleNamespace(AutoGPTPluginTemplate=type("Plugin", (), {}))
sys.modules.setdefault("auto_gpt_plugin_template", auto_plugin)
sentry_stub = types.SimpleNamespace(capture_exception=lambda *args, **kwargs: None)
sys.modules.setdefault("sentry_sdk", sentry_stub)

forge_module = sys.modules.get("forge")
if forge_module is None:
    forge_module = types.ModuleType("forge")
    sys.modules["forge"] = forge_module
forge_sdk_module = sys.modules.get("forge.sdk")
if forge_sdk_module is None:
    forge_sdk_module = types.ModuleType("forge.sdk")
    sys.modules["forge.sdk"] = forge_sdk_module
forge_sdk_model_module = sys.modules.get("forge.sdk.model")
if forge_sdk_model_module is None:
    forge_sdk_model_module = types.ModuleType("forge.sdk.model")
    sys.modules["forge.sdk.model"] = forge_sdk_model_module

if not hasattr(forge_sdk_model_module, "Task"):
    class Task:  # pragma: no cover - simple test double
        def __init__(self, input: str = ""):
            self.input = input

    forge_sdk_model_module.Task = Task

if not hasattr(forge_sdk_module, "model"):
    forge_sdk_module.model = forge_sdk_model_module
if not hasattr(forge_module, "sdk"):
    forge_module.sdk = forge_sdk_module

if importlib.util.find_spec("pydantic") is None:  # pragma: no cover - optional dependency absent
    pytestmark = pytest.mark.skip(reason="pydantic not available for whole-brain integration test")
    Agent = AgentConfiguration = AgentSettings = None  # type: ignore
    BrainBackend = None  # type: ignore
    Action = ActionSuccessResult = Episode = None  # type: ignore
else:
    try:  # pragma: no cover - gracefully handle missing heavy dependencies
        from third_party.autogpt.autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
    except ModuleNotFoundError:  # e.g. dependency resolution failed at runtime
        Agent = AgentConfiguration = AgentSettings = None  # type: ignore
    from third_party.autogpt.autogpt.config import AIDirectives, AIProfile, ConfigBuilder
    try:  # pragma: no cover - configurator may have optional deps
        from third_party.autogpt.autogpt.agent_factory.configurators import create_agent_state
    except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
        create_agent_state = None  # type: ignore
    from third_party.autogpt.autogpt.core.brain.config import BrainBackend
    from third_party.autogpt.autogpt.models.action_history import Action, ActionSuccessResult, Episode


@pytest.mark.parametrize(
    ("explicit_backend", "expected_backend"),
    [
        (True, "whole_brain"),
        (False, "brain_simulation"),
    ],
    ids=["explicit-whole-brain", "default-brain-simulation"],
)
def test_agent_with_structured_brain_backend_proposes_internal_action(
    explicit_backend, expected_backend
):
    if Agent is None:
        pytest.skip("Agent dependencies not available in this test environment")

    async def _run_test() -> None:
        llm_provider = Mock()
        llm_provider.create_chat_completion = AsyncMock()
        llm_provider.count_tokens = Mock(return_value=0)

        command_registry = Mock()
        command_registry.list_available_commands.return_value = []
        command_registry.get_command.return_value = None

        file_storage = Mock()
        legacy_config = SimpleNamespace(
            event_bus_backend="inmemory",
            event_bus_redis_host="localhost",
            event_bus_redis_port=0,
            event_bus_redis_password="",
        )

        config_kwargs = {}
        if explicit_backend:
            config_kwargs["brain_backend"] = BrainBackend.WHOLE_BRAIN
        config = AgentConfiguration(**config_kwargs)
        assert config.brain_backend.value == expected_backend
        config.whole_brain.runtime.enable_self_learning = False
        config.whole_brain.runtime.metrics_enabled = True
        settings = AgentSettings(config=config)

        agent = Agent(
            settings=settings,
            llm_provider=llm_provider,
            command_registry=command_registry,
            file_storage=file_storage,
            legacy_config=legacy_config,
        )

        agent.event_history.episodes.extend(
            [
                Episode(
                    action=Action(name="bootstrap", args={}, reasoning="initialise"),
                    result=ActionSuccessResult(outputs="ok"),
                    summary="Boot sequence completed.",
                ),
                Episode(
                    action=Action(name="observe", args={}, reasoning="check status"),
                    result=ActionSuccessResult(outputs="environment stable"),
                    summary="Status is nominal.",
                ),
            ]
        )

        command, args, thoughts = await agent.propose_action()

        assert agent.config.brain_backend.value == expected_backend
        assert agent.whole_brain is not None
        assert command == "internal_brain_action"
        assert args["intention"]
        assert thoughts["backend"] == expected_backend
        assert isinstance(thoughts["plan"], list)
        assert isinstance(thoughts["metrics"], dict)
        llm_provider.create_chat_completion.assert_not_called()

    asyncio.run(_run_test())


@pytest.mark.skipif(
    Agent is None or "create_agent_state" not in globals() or create_agent_state is None,
    reason="Agent dependencies not available in this test environment",
)
def test_config_builder_defaults_to_brain_simulation(monkeypatch, tmp_path):
    monkeypatch.delenv("BRAIN_BACKEND", raising=False)
    monkeypatch.setenv("PLUGINS_CONFIG_FILE", str(tmp_path / "plugins_config.yaml"))
    monkeypatch.setenv("AI_SETTINGS_FILE", str(tmp_path / "ai_settings.yaml"))
    monkeypatch.setenv("PROMPT_SETTINGS_FILE", str(tmp_path / "prompt_settings.yaml"))
    monkeypatch.setenv("AZURE_CONFIG_FILE", str(tmp_path / "azure.yaml"))

    config = ConfigBuilder.build_config_from_env(project_root=tmp_path)

    assert config.brain_backend == BrainBackend.BRAIN_SIMULATION

    from forge.sdk.model import Task

    ai_profile = AIProfile(
        ai_name="Test Agent",
        ai_role="Tester",
        ai_goals=["ensure defaults"],
        api_budget=0.0,
    )
    directives = AIDirectives()
    task = Task(input="validate default brain backend")

    agent_state = create_agent_state(
        agent_id="agent-default-brain",
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        app_config=config,
    )

    assert agent_state.config.brain_backend == BrainBackend.BRAIN_SIMULATION
