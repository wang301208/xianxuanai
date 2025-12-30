from __future__ import annotations

"""Motor action abstractions and actuator interfaces."""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Protocol


@dataclass
class MotorCommand:
    """Concrete command dispatched to a toolchain/executor."""

    tool: str
    operation: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_updates(
        self,
        *,
        tool: str | None = None,
        operation: str | None = None,
        arguments: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MotorCommand":
        merged_args = dict(self.arguments)
        if arguments:
            merged_args.update(arguments)
        merged_meta = dict(self.metadata)
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, dict) and isinstance(merged_meta.get(key), dict):
                    merged_meta[key] = {**merged_meta[key], **value}
                else:
                    merged_meta[key] = value
        return MotorCommand(
            tool=tool or self.tool,
            operation=operation or self.operation,
            arguments=merged_args,
            metadata=merged_meta,
        )


@dataclass
class MotorPlan:
    """Structured movement plan produced by the motor cortex."""

    intention: str
    stages: list[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    command: MotorCommand | None = None

    def describe(self) -> str:
        return self.stages[-1] if self.stages else self.intention

    def summary(self) -> Dict[str, Any]:
        return {
            "intention": self.intention,
            "stages": list(self.stages),
            "parameters": dict(self.parameters),
            "metadata": dict(self.metadata),
        }

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return self.describe()


@dataclass
class MotorExecutionResult:
    """Outcome returned by an actuator."""

    success: bool
    output: Any
    telemetry: Dict[str, float] = field(default_factory=dict)
    error: str | None = None


class ActuatorInterface(Protocol):
    """Unified interface for dispatching motor commands."""

    def execute(self, command: MotorCommand) -> MotorExecutionResult:
        ...


@dataclass
class ActionMapping:
    """Declarative mapping from intention to actuator command."""

    tool: str
    operation: str
    argument_defaults: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_command(self, plan: MotorPlan) -> MotorCommand:
        arguments = dict(self.argument_defaults)
        arguments.update(plan.parameters)
        metadata = {"intention": plan.intention, "stages": list(plan.stages)}
        metadata.update(self.metadata)
        return MotorCommand(self.tool, self.operation, arguments, metadata)


class CallableActuator(ActuatorInterface):
    """Wrap a plain callable into an actuator interface."""

    def __init__(self, dispatcher: Callable[[MotorCommand], MotorExecutionResult]) -> None:
        self.dispatcher = dispatcher

    def execute(self, command: MotorCommand) -> MotorExecutionResult:
        return self.dispatcher(command)


__all__ = [
    "MotorCommand",
    "MotorPlan",
    "MotorExecutionResult",
    "ActuatorInterface",
    "ActionMapping",
    "CallableActuator",
]
