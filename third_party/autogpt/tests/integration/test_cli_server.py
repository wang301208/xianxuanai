import asyncio
import sys
from types import SimpleNamespace

import pytest
from click.testing import CliRunner


@pytest.mark.asyncio
async def test_task_handler_minimal_execution(mocker):
    coordination_module = SimpleNamespace(TaskStatus=object())
    events_module = SimpleNamespace(coordination=coordination_module)
    mocker.patch.dict(
        sys.modules,
        {
            "spacy": mocker.Mock(),
            "events": events_module,
            "events.coordination": coordination_module,
        },
    )
    from autogpt.core.runner.cli_web_app.server import api
    mocker.patch(
        "autogpt.core.runner.cli_web_app.server.api._configure_openai_provider",
        return_value=mocker.Mock(),
    )
    mocker.patch(
        "autogpt.agents.agent.Agent.propose_action",
        return_value=(
            "finish",
            {"reason": "done"},
            {"command": {"name": "finish", "args": {"reason": "done"}}},
        ),
    )
    step_handler = await api.task_handler(SimpleNamespace(__root__={"user_input": "hi"}))

    first = await step_handler(SimpleNamespace(__root__={}))
    assert first.output["next_step_command_name"] == "finish"

    second = await step_handler(SimpleNamespace(__root__={}))
    assert second.is_last


def test_cli_server_startup(mocker):
    coordination_module = SimpleNamespace(TaskStatus=object())
    mocker.patch.dict(
        sys.modules,
        {
            "spacy": mocker.Mock(),
            "events": SimpleNamespace(coordination=coordination_module),
            "events.coordination": coordination_module,
            "autogpt.core.runner.cli_web_app.server.api": SimpleNamespace(
                task_handler=lambda *args, **kwargs: None
            ),
        },
    )
    from autogpt.core.runner.cli_web_app import cli

    runner = CliRunner()
    dummy_server = mocker.Mock()
    mocker.patch(
        "autogpt.core.runner.cli_web_app.cli.AgentProtocol.handle_task",
        return_value=dummy_server,
    )
    result = runner.invoke(cli.autogpt, ["server", "--port", "1234"])
    assert "Running AutoGPT runner httpserver..." in result.output
    dummy_server.start.assert_called_once_with(1234)
    assert result.exit_code == 0
