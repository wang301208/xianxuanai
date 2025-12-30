import enum
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, validator

try:  # pragma: no cover - fallback for lightweight test environments
    from autogpt.core.ability.schema import AbilityResult  # type: ignore
except Exception:  # pragma: no cover
    class AbilityResult(BaseModel):  # type: ignore
        """Fallback AbilityResult used when core ability module is unavailable."""
        pass


class TaskType(str, enum.Enum):
    RESEARCH = "research"
    WRITE = "write"
    EDIT = "edit"
    CODE = "code"
    DESIGN = "design"
    TEST = "test"
    PLAN = "plan"


class TaskStatus(str, enum.Enum):
    BACKLOG = "backlog"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class TaskContext(BaseModel):
    cycle_count: int = 0
    status: TaskStatus = TaskStatus.BACKLOG
    parent: Optional["Task"] = None
    prior_actions: list[AbilityResult] = Field(default_factory=list)
    memories: list = Field(default_factory=list)
    user_input: list[str] = Field(default_factory=list)
    supplementary_info: list[str] = Field(default_factory=list)
    enough_info: bool = False


class Task(BaseModel):
    objective: str
    type: TaskType
    priority: int
    ready_criteria: list[str]
    acceptance_criteria: list[str]
    context: TaskContext = Field(default_factory=TaskContext)

    @validator("type", pre=True)
    def _coerce_type(cls, value):
        if isinstance(value, TaskType):
            return value
        if isinstance(value, str):
            try:
                return TaskType(value.lower())
            except ValueError as e:  # pragma: no cover - defensive
                raise ValueError(f"Invalid task type: {value}") from e
        raise TypeError("task type must be a string or TaskType")


# Need to resolve the circular dependency between Task and TaskContext
# once both models are defined.
TaskContext.update_forward_refs()
