"""The planning system organizes the Agent's activities."""

from autogpt.core.planning.schema import Task, TaskStatus, TaskType

__all__ = [
    "PlannerSettings",
    "SimplePlanner",
    "CreativePlanner",
    "ReasoningPlanner",
    "PlanResult",
    "Task",
    "TaskStatus",
    "TaskType",
]


def __getattr__(name: str):
    if name in {"PlannerSettings", "SimplePlanner", "CreativePlanner"}:
        from autogpt.core.planning.simple import PlannerSettings, SimplePlanner
        from autogpt.core.planning.creative import CreativePlanner

        return {
            "PlannerSettings": PlannerSettings,
            "SimplePlanner": SimplePlanner,
            "CreativePlanner": CreativePlanner,
        }[name]
    if name in {"ReasoningPlanner", "PlanResult"}:
        from autogpt.core.planning.reasoner import PlanResult, ReasoningPlanner

        return {"ReasoningPlanner": ReasoningPlanner, "PlanResult": PlanResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
