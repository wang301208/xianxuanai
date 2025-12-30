import pathlib
import logging

import click
import yaml
from agent_protocol import Agent as AgentProtocol
from agent_protocol_client import AgentApi, ApiClient, Configuration, TaskRequestBody

from autogpt.core.runner.cli_web_app.server.api import task_handler
from autogpt.core.runner.client_lib.shared_click_commands import (
    DEFAULT_SETTINGS_FILE,
    make_settings,
)
from autogpt.core.runner.client_lib.utils import coroutine


logger = logging.getLogger(__name__)

# Backward compatibility: provide handle_task if missing in agent_protocol
if not hasattr(AgentProtocol, "handle_task"):
    def _handle_task(task_handler):
        class _Server:
            def start(self, port: int):
                AgentProtocol.setup_agent(task_handler, lambda step: step).start(port)

        return _Server()

    AgentProtocol.handle_task = staticmethod(_handle_task)

@click.group()
def autogpt():
    """Temporary command group for v2 commands."""
    pass


autogpt.add_command(make_settings)


@autogpt.command()
@click.option(
    "port",
    "--port",
    default=8080,
    help="The port of the webserver.",
    type=click.INT,
)
def server(port: int) -> None:
    """Run the AutoGPT runner httpserver."""
    click.echo("Running AutoGPT runner httpserver...")
    AgentProtocol.handle_task(task_handler).start(port)


@autogpt.command()
@click.option(
    "--settings-file",
    type=click.Path(),
    default=DEFAULT_SETTINGS_FILE,
)
@coroutine
async def client(settings_file) -> None:
    """Run the AutoGPT runner client."""
    settings_file = pathlib.Path(settings_file)
    settings = {}
    if settings_file.exists():
        settings = yaml.safe_load(settings_file.read_text())
    task_input = click.prompt("Task", default="Hello", show_default=True)

    configuration = Configuration(host="http://localhost:8080")

    try:
        async with ApiClient(configuration) as api_client:
            api = AgentApi(api_client)
            body = TaskRequestBody(input=task_input, additional_input=settings)
            response = await api.create_agent_task(body)
        logger.info("Task created successfully: %s", response.task_id)
        click.echo(f"Task created successfully: {response.task_id}")
    except Exception as err:
        logger.exception("Failed to create task: %s", err)
        raise click.ClickException(str(err))


if __name__ == "__main__":
    autogpt()
