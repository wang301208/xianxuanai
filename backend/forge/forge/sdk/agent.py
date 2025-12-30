import os
import pathlib
from io import BytesIO
from uuid import uuid4

import uvicorn
from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .db import AgentDB
from .errors import AutoGPTError, DatabaseError, NotFoundError
from .forge_log import ForgeLogger
from .middlewares import AgentMiddleware
from ..ui_api import mount_ui_api
from .model import (
    Artifact,
    Step,
    StepRequestBody,
    Task,
    TaskArtifactsListResponse,
    TaskListResponse,
    TaskRequestBody,
    TaskStepsListResponse,
)
from .routes.agent_protocol import base_router
from .workspace import Workspace

LOG = ForgeLogger(__name__)


class Agent:
    def __init__(self, database: AgentDB, workspace: Workspace):
        self.db = database
        self.workspace = workspace

    def get_agent_app(self, router: APIRouter = base_router):
        """
        Start the agent server.
        """

        app = FastAPI(
            title="AutoGPT Forge",
            description="Modified version of The Agent Protocol.",
            version="v0.4",
        )

        # Add CORS middleware
        origins = [
            "http://localhost:5000",
            "http://127.0.0.1:5000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            # Add any other origins you want to whitelist
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.include_router(router, prefix="/ap/v1")
        mount_ui_api(app)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        frontend_path = pathlib.Path(
            os.path.join(script_dir, "../../../../frontend/build/web")
        ).resolve()

        if os.path.exists(frontend_path):
            app.mount("/app", StaticFiles(directory=frontend_path), name="app")

            @app.get("/", include_in_schema=False)
            async def root():
                return RedirectResponse(url="/app/index.html", status_code=307)

        else:
            LOG.warning(
                f"Frontend not found. {frontend_path} does not exist. The frontend will not be served"
            )
        app.add_middleware(AgentMiddleware, agent=self)

        return app

    def start(self, port):
        uvicorn.run(
            "forge.app:app", host="localhost", port=port, log_level="error", reload=True
        )

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        Create a task for the agent.
        """
        try:
            task = await self.db.create_task(
                input=task_request.input,
                additional_input=task_request.additional_input,
            )
            return task
        except DatabaseError as err:
            LOG.error("Failed to create task", exc_info=err)
            raise DatabaseError("Failed to create task") from err

    async def list_tasks(self, page: int = 1, pageSize: int = 10) -> TaskListResponse:
        """
        List all tasks that the agent has created.
        """
        try:
            tasks, pagination = await self.db.list_tasks(page, pageSize)
            response = TaskListResponse(tasks=tasks, pagination=pagination)
            return response
        except DatabaseError as err:
            LOG.error("Failed to list tasks", exc_info=err)
            raise DatabaseError("Failed to list tasks") from err

    async def get_task(self, task_id: str) -> Task:
        """
        Get a task by ID.
        """
        try:
            task = await self.db.get_task(task_id)
            return task
        except NotFoundError as err:
            LOG.warning(f"Task {task_id} not found")
            raise NotFoundError(f"Task {task_id} not found") from err
        except DatabaseError as err:
            LOG.error(f"Failed to get task {task_id}", exc_info=err)
            raise DatabaseError(f"Failed to get task {task_id}") from err

    async def list_steps(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskStepsListResponse:
        """
        List the IDs of all steps that the task has created.
        """
        try:
            steps, pagination = await self.db.list_steps(task_id, page, pageSize)
            response = TaskStepsListResponse(steps=steps, pagination=pagination)
            return response
        except DatabaseError as err:
            LOG.error(
                f"Failed to list steps for task {task_id}", exc_info=err
            )
            raise DatabaseError(
                f"Failed to list steps for task {task_id}"
            ) from err

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        Create a step for the task.
        """
        LOG.info("Executing default step handler for task %s", task_id)
        task = await self.db.get_task(task_id)

        # Default behaviour: record the step, echo the request, and mark it completed.
        created_step = await self.db.create_step(
            task_id=task_id,
            input=step_request,
            is_last=True,
            additional_input=step_request.additional_input or {},
        )

        output = "\n".join(
            part
            for part in (
                f"Task: {task.input}",
                f"Step: {step_request.input}" if step_request.input else "Step: (empty)",
                "Result: default Agent.execute_step completed (override this in a subclass).",
            )
            if part
        )

        completed = await self.db.update_step(
            task_id=task_id,
            step_id=created_step.step_id,
            status="completed",
            output=output,
            additional_output={
                "handler": "forge.sdk.Agent.execute_step",
                "note": "Override this method to run real agent logic.",
            },
        )
        return completed

    async def get_step(self, task_id: str, step_id: str) -> Step:
        """
        Get a step by ID.
        """
        try:
            step = await self.db.get_step(task_id, step_id)
            return step
        except NotFoundError as err:
            LOG.warning(f"Step {step_id} for task {task_id} not found")
            raise NotFoundError(
                f"Step {step_id} for task {task_id} not found"
            ) from err
        except DatabaseError as err:
            LOG.error(
                f"Failed to get step {step_id} for task {task_id}", exc_info=err
            )
            raise DatabaseError(
                f"Failed to get step {step_id} for task {task_id}"
            ) from err

    async def list_artifacts(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskArtifactsListResponse:
        """
        List the artifacts that the task has created.
        """
        try:
            artifacts, pagination = await self.db.list_artifacts(
                task_id, page, pageSize
            )
            return TaskArtifactsListResponse(
                artifacts=artifacts, pagination=pagination
            )

        except DatabaseError as err:
            LOG.error(
                f"Failed to list artifacts for task {task_id}", exc_info=err
            )
            raise DatabaseError(
                f"Failed to list artifacts for task {task_id}"
            ) from err

    async def create_artifact(
        self, task_id: str, file: UploadFile, relative_path: str
    ) -> Artifact:
        """
        Create an artifact for the task.
        """
        data = None
        file_name = file.filename or str(uuid4())
        try:
            data = b""
            while contents := file.file.read(1024 * 1024):
                data += contents
            # Check if relative path ends with filename
            if relative_path.endswith(file_name):
                file_path = relative_path
            else:
                file_path = os.path.join(relative_path, file_name)

            self.workspace.write(task_id, file_path, data)

            artifact = await self.db.create_artifact(
                task_id=task_id,
                file_name=file_name,
                relative_path=relative_path,
                agent_created=False,
            )
            return artifact
        except OSError as err:
            LOG.error(
                f"Failed to write artifact {file_name} for task {task_id}", exc_info=err
            )
            raise AutoGPTError(
                f"Failed to write artifact {file_name} for task {task_id}"
            ) from err
        except DatabaseError as err:
            LOG.error("Failed to create artifact entry", exc_info=err)
            raise DatabaseError("Failed to create artifact") from err

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        """
        Get an artifact by ID.
        """
        try:
            artifact = await self.db.get_artifact(artifact_id)
            if artifact.file_name not in artifact.relative_path:
                file_path = os.path.join(artifact.relative_path, artifact.file_name)
            else:
                file_path = artifact.relative_path
            retrieved_artifact = self.workspace.read(task_id=task_id, path=file_path)
        except NotFoundError as err:
            LOG.warning(f"Artifact {artifact_id} not found")
            raise NotFoundError(f"Artifact {artifact_id} not found") from err
        except FileNotFoundError as err:
            LOG.error(
                f"Artifact file not found for task {task_id}: {err}" , exc_info=err
            )
            raise
        except OSError as err:
            LOG.error(
                f"Error reading artifact {artifact_id} for task {task_id}", exc_info=err
            )
            raise AutoGPTError(
                f"Error reading artifact {artifact_id} for task {task_id}"
            ) from err
        except DatabaseError as err:
            LOG.error(
                f"Failed to get artifact {artifact_id}", exc_info=err
            )
            raise DatabaseError(
                f"Failed to get artifact {artifact_id}"
            ) from err

        return StreamingResponse(
            BytesIO(retrieved_artifact),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={artifact.file_name}"
            },
        )
