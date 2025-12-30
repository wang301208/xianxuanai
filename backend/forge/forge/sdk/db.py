"""
This is an example implementation of the Agent Protocol DB for development Purposes
It uses SQLite as the database and file store backend.
IT IS NOT ADVISED TO USE THIS IN PRODUCTION!
"""

import asyncio
import datetime
import math
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar

T = TypeVar("T")

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    String,
    create_engine,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, joinedload, relationship, sessionmaker

from .errors import DatabaseError, NotFoundError
from .forge_log import ForgeLogger
from .model import Artifact, Pagination, Status, Step, StepRequestBody, Task

LOG = ForgeLogger(__name__)


class Base(DeclarativeBase):
    pass


class TaskModel(Base):
    __tablename__ = "tasks"

    task_id = Column(String, primary_key=True, index=True)
    input = Column(String)
    additional_input = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    artifacts = relationship("ArtifactModel", back_populates="task")


class StepModel(Base):
    __tablename__ = "steps"

    step_id = Column(String, primary_key=True, index=True)
    task_id = Column(String, ForeignKey("tasks.task_id"))
    name = Column(String)
    input = Column(String)
    status = Column(String)
    output = Column(String)
    is_last = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    additional_input = Column(JSON)
    additional_output = Column(JSON)
    artifacts = relationship("ArtifactModel", back_populates="step")


class ArtifactModel(Base):
    __tablename__ = "artifacts"

    artifact_id = Column(String, primary_key=True, index=True)
    task_id = Column(String, ForeignKey("tasks.task_id"))
    step_id = Column(String, ForeignKey("steps.step_id"))
    agent_created = Column(Boolean, default=False)
    file_name = Column(String)
    relative_path = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    step = relationship("StepModel", back_populates="artifacts")
    task = relationship("TaskModel", back_populates="artifacts")


def convert_to_task(task_obj: TaskModel, debug_enabled: bool = False) -> Task:
    if debug_enabled:
        LOG.debug(f"Converting TaskModel to Task for task_id: {task_obj.task_id}")
    task_artifacts = [convert_to_artifact(artifact) for artifact in task_obj.artifacts]
    return Task(
        task_id=task_obj.task_id,
        created_at=task_obj.created_at,
        modified_at=task_obj.modified_at,
        input=task_obj.input,
        additional_input=task_obj.additional_input,
        artifacts=task_artifacts,
    )


def convert_to_step(step_model: StepModel, debug_enabled: bool = False) -> Step:
    if debug_enabled:
        LOG.debug(f"Converting StepModel to Step for step_id: {step_model.step_id}")
    step_artifacts = [
        convert_to_artifact(artifact) for artifact in step_model.artifacts
    ]
    status = Status.completed if step_model.status == "completed" else Status.created
    return Step(
        task_id=step_model.task_id,
        step_id=step_model.step_id,
        created_at=step_model.created_at,
        modified_at=step_model.modified_at,
        name=step_model.name,
        input=step_model.input,
        status=status,
        output=step_model.output,
        artifacts=step_artifacts,
        is_last=step_model.is_last == 1,
        additional_input=step_model.additional_input,
        additional_output=step_model.additional_output,
    )


def convert_to_artifact(artifact_model: ArtifactModel) -> Artifact:
    return Artifact(
        artifact_id=artifact_model.artifact_id,
        created_at=artifact_model.created_at,
        modified_at=artifact_model.modified_at,
        agent_created=artifact_model.agent_created,
        relative_path=artifact_model.relative_path,
        file_name=artifact_model.file_name,
    )


# sqlite:///{database_name}
class AgentDB:
    def __init__(self, database_string, debug_enabled: bool = False) -> None:
        super().__init__()
        self.debug_enabled = debug_enabled
        if self.debug_enabled:
            LOG.debug(f"Initializing AgentDB with database_string: {database_string}")
        self.engine = create_engine(database_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    async def _run_in_thread(self, func, *args, **kwargs):
        """Execute blocking DB operations in a thread to avoid blocking the event loop."""
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _execute_with_session(
        self, operation: str, func: Callable[[Any], T]
    ) -> T:
        """Handle session scope and common SQLAlchemy exception logging."""

        def wrapper() -> T:
            with self.Session() as session:
                return func(session)

        try:
            return await self._run_in_thread(wrapper)
        except SQLAlchemyError as err:
            LOG.error(f"SQLAlchemy error while {operation}: {err}")
            raise DatabaseError(f"Error {operation}") from err

    async def create_task(
        self, input: Optional[str], additional_input: Optional[dict] = {}
    ) -> Task:
        if self.debug_enabled:
            LOG.debug("Creating new task")

        def _op(session) -> Task:
            new_task = TaskModel(
                task_id=str(uuid.uuid4()),
                input=input,
                additional_input=additional_input if additional_input else {},
            )
            session.add(new_task)
            session.commit()
            session.refresh(new_task)
            if self.debug_enabled:
                LOG.debug(f"Created new task with task_id: {new_task.task_id}")
            return convert_to_task(new_task, self.debug_enabled)

        return await self._execute_with_session("creating task", _op)

    async def create_step(
        self,
        task_id: str,
        input: StepRequestBody,
        is_last: bool = False,
        additional_input: Optional[Dict[str, Any]] = {},
    ) -> Step:
        if self.debug_enabled:
            LOG.debug(f"Creating new step for task_id: {task_id}")

        def _op(session) -> Step:
            new_step = StepModel(
                task_id=task_id,
                step_id=str(uuid.uuid4()),
                name=input.input,
                input=input.input,
                status="created",
                is_last=is_last,
                additional_input=additional_input,
            )
            session.add(new_step)
            session.commit()
            session.refresh(new_step)
            if self.debug_enabled:
                LOG.debug(f"Created new step with step_id: {new_step.step_id}")
            return convert_to_step(new_step, self.debug_enabled)

        return await self._execute_with_session("creating step", _op)

    async def create_artifact(
        self,
        task_id: str,
        file_name: str,
        relative_path: str,
        agent_created: bool = False,
        step_id: str | None = None,
    ) -> Artifact:
        if self.debug_enabled:
            LOG.debug(f"Creating new artifact for task_id: {task_id}")

        def _op(session) -> Artifact:
            existing_artifact = (
                session.query(ArtifactModel)
                .filter_by(
                    task_id=task_id,
                    file_name=file_name,
                    relative_path=relative_path,
                )
                .first()
            )
            if existing_artifact:
                if self.debug_enabled:
                    LOG.debug(
                        f"Artifact already exists with relative_path: {relative_path}"
                    )
                return convert_to_artifact(existing_artifact)

            new_artifact = ArtifactModel(
                artifact_id=str(uuid.uuid4()),
                task_id=task_id,
                step_id=step_id,
                agent_created=agent_created,
                file_name=file_name,
                relative_path=relative_path,
            )
            session.add(new_artifact)
            session.commit()
            session.refresh(new_artifact)
            if self.debug_enabled:
                LOG.debug(
                    f"Created new artifact with artifact_id: {new_artifact.artifact_id}"
                )
            return convert_to_artifact(new_artifact)

        return await self._execute_with_session("creating artifact", _op)

    async def get_task(self, task_id: str) -> Task:
        """Get a task by its id"""
        if self.debug_enabled:
            LOG.debug(f"Getting task with task_id: {task_id}")

        def _op(session) -> Task:
            task_obj = (
                session.query(TaskModel)
                .options(joinedload(TaskModel.artifacts))
                .filter_by(task_id=task_id)
                .first()
            )
            if task_obj:
                return convert_to_task(task_obj, self.debug_enabled)
            else:
                LOG.error(f"Task not found with task_id: {task_id}")
                raise NotFoundError("Task not found")

        return await self._execute_with_session("getting task", _op)

    async def get_step(self, task_id: str, step_id: str) -> Step:
        if self.debug_enabled:
            LOG.debug(f"Getting step with task_id: {task_id} and step_id: {step_id}")

        def _op(session) -> Step:
            step = (
                session.query(StepModel)
                .options(joinedload(StepModel.artifacts))
                .filter(StepModel.step_id == step_id)
                .first()
            )
            if step:
                return convert_to_step(step, self.debug_enabled)
            else:
                LOG.error(
                    f"Step not found with task_id: {task_id} and step_id: {step_id}"
                )
                raise NotFoundError("Step not found")

        return await self._execute_with_session("getting step", _op)

    async def get_artifact(self, artifact_id: str) -> Artifact:
        if self.debug_enabled:
            LOG.debug(f"Getting artifact with and artifact_id: {artifact_id}")

        def _op(session) -> Artifact:
            artifact_model = (
                session.query(ArtifactModel)
                .filter_by(artifact_id=artifact_id)
                .first()
            )
            if artifact_model:
                return convert_to_artifact(artifact_model)
            else:
                LOG.error(f"Artifact not found with and artifact_id: {artifact_id}")
                raise NotFoundError("Artifact not found")

        return await self._execute_with_session("getting artifact", _op)

    async def update_step(
        self,
        task_id: str,
        step_id: str,
        status: Optional[str] = None,
        output: Optional[str] = None,
        additional_input: Optional[Dict[str, Any]] = None,
        additional_output: Optional[Dict[str, Any]] = None,
    ) -> Step:
        if self.debug_enabled:
            LOG.debug(f"Updating step with task_id: {task_id} and step_id: {step_id}")

        def _op(session) -> Step:
            step = (
                session.query(StepModel)
                .filter_by(task_id=task_id, step_id=step_id)
                .first()
            )
            if step:
                if status is not None:
                    step.status = status
                if additional_input is not None:
                    step.additional_input = additional_input
                if output is not None:
                    step.output = output
                if additional_output is not None:
                    step.additional_output = additional_output
                session.commit()
                session.refresh(step)
                return convert_to_step(step, self.debug_enabled)
            else:
                LOG.error(
                    f"Step not found for update with task_id: {task_id} and step_id: {step_id}"
                )
                raise NotFoundError("Step not found")

        return await self._execute_with_session("updating step", _op)

    async def update_artifact(
        self,
        artifact_id: str,
        *,
        file_name: str = "",
        relative_path: str = "",
        agent_created: Optional[Literal[True]] = None,
    ) -> Artifact:
        LOG.debug(f"Updating artifact with artifact_id: {artifact_id}")

        def _op(session) -> Artifact:
            artifact = (
                session.query(ArtifactModel)
                .filter_by(artifact_id=artifact_id)
                .first()
            )
            if artifact:
                if file_name:
                    artifact.file_name = file_name
                if relative_path:
                    artifact.relative_path = relative_path
                if agent_created:
                    artifact.agent_created = agent_created
                session.commit()
                session.refresh(artifact)
                return convert_to_artifact(artifact)
            else:
                LOG.error(f"Artifact not found with artifact_id: {artifact_id}")
                raise NotFoundError("Artifact not found")

        return await self._execute_with_session("updating artifact", _op)

    async def list_tasks(
        self, page: int = 1, per_page: int = 10
    ) -> Tuple[List[Task], Pagination]:
        if self.debug_enabled:
            LOG.debug("Listing tasks")

        def _op(session) -> Tuple[List[Task], Pagination]:
            tasks = (
                session.query(TaskModel)
                .offset((page - 1) * per_page)
                .limit(per_page)
                .all()
            )
            total = session.query(TaskModel).count()
            pages = math.ceil(total / per_page)
            pagination = Pagination(
                total_items=total,
                total_pages=pages,
                current_page=page,
                page_size=per_page,
            )
            return [
                convert_to_task(task, self.debug_enabled) for task in tasks
            ], pagination

        return await self._execute_with_session("listing tasks", _op)

    async def list_steps(
        self, task_id: str, page: int = 1, per_page: int = 10
    ) -> Tuple[List[Step], Pagination]:
        if self.debug_enabled:
            LOG.debug(f"Listing steps for task_id: {task_id}")

        def _op(session) -> Tuple[List[Step], Pagination]:
            steps = (
                session.query(StepModel)
                .filter_by(task_id=task_id)
                .offset((page - 1) * per_page)
                .limit(per_page)
                .all()
            )
            total = session.query(StepModel).filter_by(task_id=task_id).count()
            pages = math.ceil(total / per_page)
            pagination = Pagination(
                total_items=total,
                total_pages=pages,
                current_page=page,
                page_size=per_page,
            )
            return [
                convert_to_step(step, self.debug_enabled) for step in steps
            ], pagination

        return await self._execute_with_session("listing steps", _op)

    async def list_artifacts(
        self, task_id: str, page: int = 1, per_page: int = 10
    ) -> Tuple[List[Artifact], Pagination]:
        if self.debug_enabled:
            LOG.debug(f"Listing artifacts for task_id: {task_id}")

        def _op(session) -> Tuple[List[Artifact], Pagination]:
            artifacts = (
                session.query(ArtifactModel)
                .filter_by(task_id=task_id)
                .offset((page - 1) * per_page)
                .limit(per_page)
                .all()
            )
            total = session.query(ArtifactModel).filter_by(task_id=task_id).count()
            pages = math.ceil(total / per_page)
            pagination = Pagination(
                total_items=total,
                total_pages=pages,
                current_page=page,
                page_size=per_page,
            )
            return [
                convert_to_artifact(artifact) for artifact in artifacts
            ], pagination

        return await self._execute_with_session("listing artifacts", _op)
