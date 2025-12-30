from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


class CallbackOrchestrator:
    """Thin orchestrator using user-provided callbacks for reset/step."""

    def __init__(
        self,
        reset_cb: Callable[[Dict[str, Any]], Dict[str, Any]],
        step_cb: Callable[[Dict[str, Any]], Dict[str, Any]],
    ):
        self.reset_cb = reset_cb
        self.step_cb = step_cb

    def reset(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        return self.reset_cb(task_spec)

    def step(self, policy_directive: Dict[str, Any]) -> Dict[str, Any]:
        return self.step_cb(policy_directive)


@dataclass
class AutoGPTTaskProxy:
    """Convenience wrapper around the AutoGPT agent protocol server."""

    client: Any
    task_template: Dict[str, Any]
    task_id: Optional[str] = None

    def create_task(self, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = dict(self.task_template)
        if overrides:
            payload.update(overrides)
        response = self.client.create_task(payload)
        self.task_id = response["task_id"]
        return response

    def next_step(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        if not self.task_id:
            raise RuntimeError("Task must be created before requesting next step.")
        return self.client.advance_task(self.task_id, directive)
