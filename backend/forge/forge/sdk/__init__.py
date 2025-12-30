"""
The Forge SDK. This is the core of the Forge. It contains the agent protocol, which is the
core of the Forge.
"""
import logging
from typing import Any

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - fallback for optional dependencies during tests
    from ..llm import chat_completion_request, create_embedding_request, transcribe_audio
except ModuleNotFoundError as exc:
    LOG.warning(
        "LLM dependencies unavailable; falling back to stubbed SDK imports.",
        exc_info=exc,
    )

    async def chat_completion_request(*args: Any, **kwargs: Any):
        raise exc

    async def create_embedding_request(*args: Any, **kwargs: Any):
        raise exc

    async def transcribe_audio(*args: Any, **kwargs: Any):
        raise exc
try:  # pragma: no cover - optional runtime dependency (uvicorn)
    from .agent import Agent
except ModuleNotFoundError as exc:
    LOG.warning(
        "Agent utilities unavailable; ensure optional server dependencies are installed.",
        exc_info=exc,
    )
    class Agent:  # type: ignore
        def __init__(self, database: Any, workspace: Any):
            self.db = database
            self.workspace = workspace

        async def create_task(self, task_request):
            return await self.db.create_task(
                input=task_request.input, additional_input=task_request.additional_input
            )
try:  # pragma: no cover - optional dependency (sqlalchemy)
    from .db import AgentDB, Base
except ModuleNotFoundError as exc:
    LOG.warning(
        "Database dependencies unavailable; using in-memory AgentDB stub.",
        exc_info=exc,
    )
    import uuid
    from datetime import datetime

    from .model import Artifact, Status, Step, Task

    class AgentDB:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            self.tasks: dict[str, Task] = {}
            self.steps: dict[str, Step] = {}
            self.artifacts: dict[str, Artifact] = {}
            self.debug_enabled = False

        async def create_task(self, input: str, additional_input: dict | None = None) -> Task:
            task_id = str(uuid.uuid4())
            now = datetime.utcnow()
            task = Task(
                task_id=task_id,
                created_at=now,
                modified_at=now,
                input=input,
                additional_input=additional_input or {},
                artifacts=[],
            )
            self.tasks[task_id] = task
            return task

        async def get_task(self, task_id: str) -> Task:
            return self.tasks[task_id]

        async def create_step(
            self,
            task_id: str,
            input,
            is_last: bool = False,
            additional_input: dict | None = None,
        ) -> Step:
            now = datetime.utcnow()
            step_id = str(uuid.uuid4())
            step = Step(
                task_id=task_id,
                step_id=step_id,
                created_at=now,
                modified_at=now,
                name=input.input,
                input=input.input,
                status=Status.created,
                output=None,
                artifacts=[],
                is_last=is_last,
                additional_input=additional_input or {},
                additional_output={},
            )
            self.steps[step_id] = step
            return step

        async def update_step(
            self,
            task_id: str,
            step_id: str,
            status: str | None = None,
            output: str | None = None,
            additional_output: dict | None = None,
            additional_input: dict | None = None,
        ) -> Step:
            step = self.steps[step_id]
            if status:
                step.status = Status.completed if status == "completed" else Status.created
            if output is not None:
                step.output = output
            if additional_output is not None:
                step.additional_output = additional_output
            if additional_input is not None:
                step.additional_input = additional_input
            step.modified_at = datetime.utcnow()
            return step

        async def create_artifact(
            self,
            task_id: str,
            file_name: str,
            relative_path: str,
            agent_created: bool = False,
            step_id: str | None = None,
        ) -> Artifact:
            now = datetime.utcnow()
            artifact_id = str(uuid.uuid4())
            artifact = Artifact(
                artifact_id=artifact_id,
                created_at=now,
                modified_at=now,
                relative_path=relative_path,
                agent_created=agent_created,
                file_name=file_name,
                step_id=step_id,
                task_id=task_id,
            )
            self.artifacts[artifact_id] = artifact
            return artifact

        async def get_step(self, task_id: str, step_id: str) -> Step:
            return self.steps[step_id]

    Base = None  # type: ignore
from .errors import (
    AccessDeniedError,
    AgentException,
    CodeExecutionError,
    CommandExecutionError,
    ConfigurationError,
    DuplicateOperationError,
    InvalidAgentResponseError,
    InvalidArgumentError,
    NotFoundError,
    OperationNotAllowedError,
    TooMuchOutputError,
    UnknownCommandError,
)
from .forge_log import ForgeLogger
from .model import (
    Artifact,
    ArtifactUpload,
    Pagination,
    Status,
    Step,
    StepOutput,
    StepRequestBody,
    Task,
    TaskArtifactsListResponse,
    TaskListResponse,
    TaskRequestBody,
    TaskStepsListResponse,
)
try:  # pragma: no cover - optional dependency (jinja2)
    from .prompting import PromptEngine
except ModuleNotFoundError as exc:
    LOG.warning(
        "Prompt engine unavailable; install templating dependencies for full functionality.",
        exc_info=exc,
    )

    class PromptEngine:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            raise exc
try:  # pragma: no cover - optional dependency (google.cloud for GCS)
    from .workspace import LocalWorkspace, Workspace
except ModuleNotFoundError as exc:
    LOG.warning(
        "Workspace backends unavailable; defaulting to stub workspace.",
        exc_info=exc,
    )

    from pathlib import Path

    class Workspace:  # type: ignore
        def __init__(self, base_path: str):
            self.base_path = Path(base_path).resolve()

        def _resolve_path(self, task_id: str, path: str) -> Path:
            rel = Path(str(path).lstrip("/"))
            return (self.base_path / task_id / rel).resolve()

        def read(self, task_id: str, path: str) -> bytes:  # pragma: no cover - stub
            return self._resolve_path(task_id, path).read_bytes()

        def write(self, task_id: str, path: str, data: bytes) -> None:  # pragma: no cover - stub
            file_path = self._resolve_path(task_id, path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(data)

        def delete(
            self, task_id: str, path: str, directory: bool = False, recursive: bool = False
        ):
            target = self._resolve_path(task_id, path)
            if directory:
                if recursive:
                    shutil.rmtree(target)
                else:
                    target.rmdir()
            else:
                target.unlink(missing_ok=True)

        def exists(self, task_id: str, path: str) -> bool:
            return self._resolve_path(task_id, path).exists()

        def list(self, task_id: str, path: str):
            base = self._resolve_path(task_id, path)
            if not base.exists():
                return []
            return [str(p.relative_to(self.base_path / task_id)) for p in base.iterdir()]

    class LocalWorkspace(Workspace):  # type: ignore
        pass

__all__ = [
    "chat_completion_request",
    "create_embedding_request",
    "transcribe_audio",
    "Agent",
    "AgentDB",
    "Base",
    "AccessDeniedError",
    "AgentException",
    "CodeExecutionError",
    "CommandExecutionError",
    "ConfigurationError",
    "DuplicateOperationError",
    "InvalidAgentResponseError",
    "InvalidArgumentError",
    "NotFoundError",
    "OperationNotAllowedError",
    "TooMuchOutputError",
    "UnknownCommandError",
    "ForgeLogger",
    "Artifact",
    "ArtifactUpload",
    "Pagination",
    "Status",
    "Step",
    "StepOutput",
    "StepRequestBody",
    "Task",
    "TaskArtifactsListResponse",
    "TaskListResponse",
    "TaskRequestBody",
    "TaskStepsListResponse",
    "PromptEngine",
    "LocalWorkspace",
    "Workspace",
]
