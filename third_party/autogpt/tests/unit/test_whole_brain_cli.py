from __future__ import annotations

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

if "forge" not in sys.modules:
    forge_module = types.ModuleType("forge")
    sdk_module = types.ModuleType("forge.sdk")
    db_module = types.ModuleType("forge.sdk.db")
    model_module = types.ModuleType("forge.sdk.model")

    class AgentDB:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs) -> None:
            pass

    @dataclass
    class _Task:
        input: str
        additional_input: str | None
        created_at: datetime
        modified_at: datetime
        task_id: str
        artifacts: list

    db_module.AgentDB = AgentDB
    model_module.Task = _Task
    sdk_module.db = db_module
    sdk_module.model = model_module
    forge_module.sdk = sdk_module
    sys.modules["forge"] = forge_module
    sys.modules["forge.sdk"] = sdk_module
    sys.modules["forge.sdk.db"] = db_module
    sys.modules["forge.sdk.model"] = model_module

import autogpt.agent_factory.configurators as configurators
from autogpt import commands as command_pkg
from autogpt.app.cli import cli
from autogpt.app.main import NullChatModelProvider, UserFeedback, run_interaction_loop
from autogpt.config import AIDirectives, AIProfile, ConfigBuilder
from autogpt.agent_factory.configurators import create_agent
from autogpt.file_storage import FileStorageBackendName, get_storage
from forge.sdk.model import Task
from modules.brain.whole_brain import WholeBrainSimulation


def test_cli_allows_whole_brain_without_openai(monkeypatch):
    """CLI should not require an OpenAI key when Whole Brain backend is selected."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BRAIN_BACKEND", "whole_brain")
    monkeypatch.setattr(configurators, "COMMAND_CATEGORIES", [], raising=False)
    assert configurators.COMMAND_CATEGORIES == []

    runner = CliRunner()
    called: dict[str, object] = {}

    def fake_run_auto_gpt(*args, **kwargs):
        called["args"] = (args, kwargs)

    monkeypatch.setattr("autogpt.app.main.run_auto_gpt", fake_run_auto_gpt)

    result = runner.invoke(cli, ["run", "--skip-reprompt"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "args" in called


def test_run_interaction_loop_completes_cycle_without_openai(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """AutoGPT should complete a thinking cycle without OpenAI credentials."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BRAIN_BACKEND", "whole_brain")

    config = ConfigBuilder.build_config_from_env(project_root=tmp_path)
    config.skip_news = True
    config.plugins = []
    config.app_data_dir = tmp_path / "data"
    config.app_data_dir.mkdir(parents=True, exist_ok=True)
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    config.plugins_dir = str(plugins_dir)
    config.disabled_command_categories = list(command_pkg.COMMAND_CATEGORIES)

    storage = get_storage(
        FileStorageBackendName.LOCAL, root_path=config.app_data_dir, restrict_to_root=False
    )
    storage.initialize()

    task = Task(
        input="diagnose system status",
        additional_input=None,
        created_at=datetime.now(),
        modified_at=datetime.now(),
        task_id="whole-brain",
        artifacts=[],
    )

    ai_profile = AIProfile(ai_name="WB-Agent", ai_role="Whole brain operator", ai_goals=[task.input])
    directives = AIDirectives()

    whole_brain = WholeBrainSimulation(**config.whole_brain.to_simulation_kwargs())

    agent = create_agent(
        agent_id="wb-agent",
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        app_config=config,
        file_storage=storage,
        llm_provider=NullChatModelProvider(),
        whole_brain=whole_brain,
    )

    assert agent.whole_brain is whole_brain
    assert getattr(agent, "_whole_brain_adapter", None) is not None

    async def fake_get_user_feedback(*args, **kwargs):  # type: ignore[override]
        return UserFeedback.EXIT, "", None

    monkeypatch.setattr("autogpt.app.main.get_user_feedback", fake_get_user_feedback)

    with pytest.raises(SystemExit):
        asyncio.run(run_interaction_loop(agent))

    assert agent.config.cycle_count >= 1
