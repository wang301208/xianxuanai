import os
import sys
import logging
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "autogpts", "autogpt")))

import types
sys.modules.setdefault("auto_gpt_plugin_template", types.SimpleNamespace(AutoGPTPluginTemplate=object))

class _DummyStructuredLogHandler(logging.Handler):
    pass

class _DummyCloudLoggingFilter(logging.Filter):
    pass

sys.modules.setdefault(
    "google.cloud.logging_v2.handlers",
    types.SimpleNamespace(
        CloudLoggingFilter=_DummyCloudLoggingFilter,
        StructuredLogHandler=_DummyStructuredLogHandler,
    ),
)
sys.modules.setdefault("playsound", types.SimpleNamespace(playsound=lambda *a, **k: None))
sys.modules.setdefault("gtts", types.SimpleNamespace(gTTS=object))
class _DummyActionLogger:
    def __init__(self, *args, **kwargs):
        pass

sys.modules.setdefault("monitoring", types.SimpleNamespace(ActionLogger=_DummyActionLogger))

pytest.importorskip("pydantic")
pytest.importorskip("inflection")

from third_party.autogpt.autogpt.core.agent.simple import AgentSettings, SimpleAgent
from third_party.autogpt.autogpt.core.ability import SimpleAbilityRegistry
from third_party.autogpt.autogpt.core.memory import SimpleMemory
from third_party.autogpt.autogpt.core.planning.simple import SimplePlanner
from third_party.autogpt.autogpt.core.workspace.simple import SimpleWorkspace
from third_party.autogpt.autogpt.core.resource.model_providers.openai import OpenAIProvider


class Dummy:  # simple placeholder objects
    pass


@pytest.mark.asyncio
async def test_from_workspace_supports_multiple_providers(tmp_path, monkeypatch):
    logger = logging.getLogger("test")

    provider_settings1 = OpenAIProvider.default_settings.copy(update={"name": "provider1"})
    provider_settings2 = OpenAIProvider.default_settings.copy(update={"name": "provider2"})

    agent_settings = AgentSettings(
        agent=SimpleAgent.default_settings,
        ability_registry=SimpleAbilityRegistry.default_settings,
        memory=SimpleMemory.default_settings,
        openai_providers={"openai": provider_settings1, "alt": provider_settings2},
        planning=SimplePlanner.default_settings,
        creative_planning=SimplePlanner.default_settings,
        workspace=SimpleWorkspace.default_settings,
    )

    (tmp_path / "agent_settings.json").write_text(agent_settings.json())

    providers_created = {}
    captured = {}

    def fake_get_system_instance(cls, system_name, agent_settings, logger, *args, **kwargs):
        if system_name == "openai_provider":
            settings = kwargs["system_settings"]
            obj = Dummy()
            providers_created[settings.name] = obj
            return obj
        if system_name in ("planning", "creative_planning", "ability_registry"):
            captured[system_name] = kwargs.get("model_providers")
            return Dummy()
        if system_name == "cognition":
            return Dummy()
        if system_name == "workspace":
            ws = Dummy()
            ws.root = tmp_path
            return ws
        return Dummy()

    monkeypatch.setattr(SimpleAgent, "_get_system_instance", classmethod(fake_get_system_instance))

    agent = SimpleAgent.from_workspace(tmp_path, logger)

    assert set(providers_created.keys()) == {"provider1", "provider2"}
    assert isinstance(agent, SimpleAgent)
    for mapping in captured.values():
        assert set(mapping.values()) == set(providers_created.values())
        assert len(mapping) == 2
