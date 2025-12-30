from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

from forge.sdk.db import AgentDB
from forge.sdk.errors import NotFoundError
from forge.sdk.model import StepRequestBody, TaskRequestBody
from mcp import types as mcp_types
from mcp.server import NotificationOptions, Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError

from autogpt.app.agent_protocol_server import AgentProtocolServer
from autogpt.config import Config, ConfigBuilder, assert_config_has_openai_api_key
from autogpt.core.resource.model_providers.openai import OpenAIModelName, OpenAIProvider
from autogpt.core.runner.client_lib.utils import coroutine
from autogpt.file_storage import FileStorageBackendName, get_storage
from autogpt.logs.config import configure_logging
from autogpt.plugins import scan_plugins

from .configurator import apply_overrides_to_config
from scripts.install_plugin_deps import install_plugin_dependencies

logger = logging.getLogger(__name__)

TASKS_ROOT_URI = "autogpt://tasks"
TASK_PAGE_SIZE = int(os.getenv("MCP_TASK_PAGE_SIZE", "50"))


def _configure_openai_provider(config: Config) -> OpenAIProvider:
    if config.openai_credentials is None:
        raise RuntimeError("OpenAI key is not configured")

    openai_settings = OpenAIProvider.default_settings.copy(deep=True)
    openai_settings.credentials = config.openai_credentials
    return OpenAIProvider(
        settings=openai_settings,
        logger=logging.getLogger("OpenAIProvider"),
    )


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _serialize_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model


def _serialize_optional(model: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model


def _task_to_resource(task_data: dict[str, Any]) -> mcp_types.Resource:
    task_id = task_data["task_id"]
    return mcp_types.Resource(
        name=f"tasks/{task_id}",
        title=f"Task {task_id}",
        uri=f"{TASKS_ROOT_URI}/{task_id}",
        description=task_data.get("input", "") or "AutoGPT task",
    )


def _mcp_error(code: int, message: str) -> McpError:
    return McpError(mcp_types.ErrorData(code=code, message=message))


def _format_step(step_data: dict[str, Any]) -> str:
    output = step_data.get("output") or "(no output)"
    return (
        f"Step {step_data.get('step_id')} "
        f"[{step_data.get('status')}]\n"
        f"Command: {step_data.get('name') or 'n/a'}\n"
        f"Output: {output}"
    )


def _format_task_list(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return "No AutoGPT tasks recorded."
    return "\n".join(
        f"- {task['task_id']}: {task.get('input', '').strip() or '(no description)'}"
        for task in tasks
    )


def _parse_task_uri(raw_uri: str) -> tuple[str | None, str | None]:
    parsed = urlparse(raw_uri)
    if parsed.scheme != "autogpt":
        raise _mcp_error(mcp_types.INVALID_PARAMS, "Unsupported URI scheme")
    host = parsed.netloc or None
    path = parsed.path.lstrip("/")
    task_id = path or None
    return host, task_id


async def _gather_task_payload(
    agent_server: AgentProtocolServer, task_id: str
) -> dict[str, Any]:
    try:
        task = await agent_server.get_task(task_id)
    except NotFoundError:
        raise _mcp_error(mcp_types.INVALID_PARAMS, f"Task {task_id} was not found") from None

    steps_response = await agent_server.list_steps(task_id, page=1, pageSize=TASK_PAGE_SIZE)
    artifacts_response = await agent_server.list_artifacts(
        task_id, page=1, pageSize=TASK_PAGE_SIZE
    )

    payload: dict[str, Any] = {
        "task": _serialize_model(task),
        "steps": [_serialize_model(step) for step in steps_response.steps or []],
        "artifacts": [_serialize_model(a) for a in artifacts_response.artifacts or []],
    }

    step_pagination = _serialize_optional(steps_response.pagination)
    if step_pagination:
        payload["steps_pagination"] = step_pagination

    artifact_pagination = _serialize_optional(artifacts_response.pagination)
    if artifact_pagination:
        payload["artifacts_pagination"] = artifact_pagination

    return payload


def _build_mcp_server(agent_server: AgentProtocolServer) -> Server:
    server = Server("AutoGPT MCP", version="0.5.1")

    @server.list_resources()
    async def list_resources() -> list[mcp_types.Resource]:
        tasks_response = await agent_server.list_tasks(page=1, pageSize=TASK_PAGE_SIZE)
        task_items = [_serialize_model(task) for task in tasks_response.tasks or []]

        resources = [
            mcp_types.Resource(
                name="tasks",
                title="AutoGPT Tasks",
                uri=TASKS_ROOT_URI,
                description="Browse tasks managed by the AutoGPT Agent Protocol layer.",
            )
        ]
        resources.extend(_task_to_resource(task) for task in task_items)
        return resources

    @server.read_resource()
    async def read_resource(uri: mcp_types.AnyUrl):
        host, task_id = _parse_task_uri(str(uri))
        if host != "tasks":
            raise _mcp_error(mcp_types.INVALID_PARAMS, "Unsupported resource host")

        if task_id is None:
            tasks_response = await agent_server.list_tasks(page=1, pageSize=TASK_PAGE_SIZE)
            payload: dict[str, Any] = {
                "tasks": [_serialize_model(task) for task in tasks_response.tasks or []],
            }
            pagination = _serialize_optional(tasks_response.pagination)
            if pagination:
                payload["pagination"] = pagination
            return [
                ReadResourceContents(
                    content=_json_dumps(payload),
                    mime_type="application/json",
                )
            ]

        payload = await _gather_task_payload(agent_server, task_id)
        return [
            ReadResourceContents(
                content=_json_dumps(payload),
                mime_type="application/json",
            )
        ]

    @server.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="list_tasks",
                title="List AutoGPT tasks",
                description="Return the tasks currently tracked by AutoGPT.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            mcp_types.Tool(
                name="create_task",
                title="Create a new AutoGPT task",
                description="Create a task through the Agent Protocol interface.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Primary instruction for the agent.",
                        },
                        "additional_input": {
                            "type": "object",
                            "description": "Optional metadata forwarded with the task.",
                        },
                    },
                    "required": ["input"],
                    "additionalProperties": False,
                },
            ),
            mcp_types.Tool(
                name="describe_task",
                title="Describe an AutoGPT task",
                description="Fetch details, steps, and artifacts for a given task.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Identifier returned from create_task or list_tasks.",
                        }
                    },
                    "required": ["task_id"],
                    "additionalProperties": False,
                },
            ),
            mcp_types.Tool(
                name="advance_task",
                title="Advance an AutoGPT task",
                description="Request the agent to execute the next step for the specified task.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Identifier of the task to advance.",
                        },
                        "input": {
                            "type": "string",
                            "description": "Optional feedback or command authorization for the agent.",
                        },
                        "additional_input": {
                            "type": "object",
                            "description": "Optional metadata forwarded with the step.",
                        },
                    },
                    "required": ["task_id"],
                    "additionalProperties": False,
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> mcp_types.CallToolResult:
        arguments = arguments or {}

        if name == "list_tasks":
            tasks_response = await agent_server.list_tasks(page=1, pageSize=TASK_PAGE_SIZE)
            tasks_data = [_serialize_model(task) for task in tasks_response.tasks or []]
            payload = {
                "tasks": tasks_data,
                "pagination": _serialize_optional(tasks_response.pagination),
            }
            return mcp_types.CallToolResult(
                content=[
                    mcp_types.TextContent(
                        type="text",
                        text=_format_task_list(tasks_data),
                    )
                ],
                structuredContent=payload,
            )

        if name == "create_task":
            task_input = arguments.get("input")
            if not isinstance(task_input, str) or not task_input:
                raise _mcp_error(mcp_types.INVALID_PARAMS, "'input' is required and must be a string")

            task_request = TaskRequestBody(
                input=task_input,
                additional_input=arguments.get("additional_input"),
            )
            task = await agent_server.create_task(task_request)
            task_data = _serialize_model(task)
            message = f"Created task {task_data['task_id']}. Call advance_task to start executing it."
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=message)],
                structuredContent=task_data,
            )

        if name == "describe_task":
            task_id = arguments.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise _mcp_error(mcp_types.INVALID_PARAMS, "'task_id' must be a non-empty string")

            payload = await _gather_task_payload(agent_server, task_id)
            steps = payload.get("steps", [])
            text = (
                f"Task {payload['task']['task_id']}: {payload['task'].get('input', '')}\n"
                f"Steps recorded: {len(steps)}"
            )
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=text)],
                structuredContent=payload,
            )

        if name == "advance_task":
            task_id = arguments.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise _mcp_error(mcp_types.INVALID_PARAMS, "'task_id' must be a non-empty string")

            step_request = StepRequestBody(
                input=arguments.get("input"),
                additional_input=arguments.get("additional_input"),
            )
            try:
                step = await agent_server.execute_step(task_id, step_request)
            except NotFoundError:
                raise _mcp_error(mcp_types.INVALID_PARAMS, f"Task {task_id} was not found") from None
            except Exception as exc:  # pragma: no cover - defensive fallback
                raise _mcp_error(mcp_types.INTERNAL_ERROR, str(exc)) from exc

            step_data = _serialize_model(step)
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=_format_step(step_data))],
                structuredContent=step_data,
            )

        raise _mcp_error(mcp_types.METHOD_NOT_FOUND, f"Unknown tool '{name}'")

    return server


@coroutine
async def run_auto_gpt_mcp_server(
    prompt_settings: str | os.PathLike[str] | None = None,
    debug: bool = False,
    log_level: str | None = None,
    log_format: str | None = None,
    log_file_format: str | None = None,
    gpt3only: bool = False,
    gpt4only: bool = False,
    browser_name: str | None = None,
    allow_downloads: bool = False,
    install_plugin_deps: bool = False,
) -> None:
    config = ConfigBuilder.build_config_from_env()

    local = config.file_storage_backend == FileStorageBackendName.LOCAL
    restrict_to_root = not local or config.restrict_to_workspace
    file_storage = get_storage(
        config.file_storage_backend,
        root_path="data",
        restrict_to_root=restrict_to_root,
    )
    file_storage.initialize()

    configure_logging(
        debug=debug,
        level=log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        config=config.logging,
        tts_config=config.tts_config,
    )

    config.fast_llm = config.fast_llm or OpenAIModelName.GPT3_16k
    config.smart_llm = config.smart_llm or OpenAIModelName.GPT4_TURBO
    config.temperature = config.temperature or 0
    assert_config_has_openai_api_key(config)

    await apply_overrides_to_config(
        config=config,
        prompt_settings_file=prompt_settings,
        gpt3only=gpt3only,
        gpt4only=gpt4only,
        browser_name=browser_name,
        allow_downloads=allow_downloads,
    )

    llm_provider = _configure_openai_provider(config)

    if install_plugin_deps:
        install_plugin_dependencies()

    config.plugins = scan_plugins(config)

    database = AgentDB(
        database_string=os.getenv("AP_SERVER_DB_URL", "sqlite:///data/ap_server.db"),
        debug_enabled=debug,
    )

    agent_server = AgentProtocolServer(
        app_config=config,
        database=database,
        file_storage=file_storage,
        llm_provider=llm_provider,
    )

    server = _build_mcp_server(agent_server)
    init_options = server.create_initialization_options(
        notification_options=NotificationOptions(),
        experimental_capabilities={},
    )

    logger.info("Starting AutoGPT MCP server over stdio")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)

    logger.info(
        "Total OpenAI session cost: $%s",
        round(sum(b.total_cost for b in agent_server._task_budgets.values()), 2),
    )
