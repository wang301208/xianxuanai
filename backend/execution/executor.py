from __future__ import annotations

import ast
import hashlib
import re
import logging
from typing import Any, Dict, List, Sequence, Tuple

from third_party.autogpt.autogpt.core.errors import (
    SkillExecutionError,
    SkillSecurityError,
)
from capability.skill_library import SkillLibrary
from common.async_utils import run_async
from .task_graph import TaskGraph
from .scheduler import Scheduler

logger = logging.getLogger(__name__)


SAFE_BUILTINS: Dict[str, Any] = {
    "__import__": __import__,
    "len": len,
    "range": range,
    "print": print,
    "Exception": Exception,
    "RuntimeError": RuntimeError,
}


def _verify_skill(name: str, code: str, metadata: Dict[str, Any]) -> None:
    skill_type = metadata.get("type", "python")
    if skill_type != "python":
        return
    signature = metadata.get("signature")
    if not signature:
        raise SkillSecurityError(name, "missing signature")
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if signature != digest:
        raise SkillSecurityError(name, "invalid signature")


class Executor:
    """Very small executor that decomposes a goal into skill tasks."""

    def __init__(self, skill_library: SkillLibrary, scheduler: Scheduler | None = None) -> None:
        self.skill_library = skill_library
        self.scheduler = scheduler or Scheduler()
        if not self.scheduler._agents:
            # Ensure at least one local agent for execution
            self.scheduler.add_agent("local")

    # Goal decomposition
    def decompose_goal(self, goal: str) -> TaskGraph:
        """Split a goal string into sequential skill tasks.

        The goal is split on the words 'then' or 'and'. Each resulting token is
        treated as a skill name if it matches a skill in the library. Subsequent
        tasks depend on the previous task, forming a simple chain.
        """

        graph = TaskGraph()
        tokens = [t.strip() for t in re.split(r"then|and", goal) if t.strip()]
        previous: str | None = None
        available = set(self.skill_library.list_skills())
        for token in tokens:
            if token in available:
                deps: List[str] = [previous] if previous else []
                graph.add_task(
                    token,
                    description=f"Execute {token}",
                    skill=token,
                    dependencies=deps,
                )
                previous = token
        return graph

    # Task scheduling and execution
    async def execute(self, plans: Sequence[Tuple[str, float]] | str) -> Dict[str, Any]:
        """Execute the best plan from ``plans``.

        ``plans`` may be a single goal string or a sequence of ``(goal, score)``
        tuples. The plan with the highest score is selected for execution.
        """

        if isinstance(plans, str):
            goal = plans
        else:
            goal, _ = max(plans, key=lambda p: p[1]) if plans else ("", 0)
        graph = self.decompose_goal(goal)
        return await self.scheduler.submit(graph, self._call_skill)

    def execute_sync(self, plans: Sequence[Tuple[str, float]] | str) -> Dict[str, Any]:
        """Synchronous wrapper around :meth:`execute` for legacy callers."""
        return run_async(self.execute(plans))

    async def _call_skill(self, agent: str, name: str) -> Any:
        """Execute ``name`` skill for ``agent``.

        The basic executor only supports a local ``SkillLibrary`` and therefore
        ignores the ``agent`` argument, but the parameter allows alternative
        implementations to route tasks to remote agents or specialized
        resources.
        """
        code, meta = await self.skill_library.get_skill(name)
        skill_type = meta.get("type", "python")
        if skill_type != "python":
            func = self.skill_library.resolve_callable(name, meta)
            if func is None:
                err = SkillExecutionError(name, "callable target is unavailable")
                logger.error("Error executing skill %s: %s", name, err.cause)
                raise err
            try:
                return func()
            except Exception as err:  # noqa: BLE001
                logger.exception("Error executing skill %s: %s", name, err)
                raise SkillExecutionError(name, str(err)) from err

        try:
            _verify_skill(name, code, meta)
        except SkillSecurityError as err:
            logger.exception("Security violation for skill %s: %s", name, err.cause)
            raise
        namespace: Dict[str, Any] = {}
        parsed = ast.parse(code, mode="exec")
        exec(compile(parsed, filename=name, mode="exec"), {"__builtins__": SAFE_BUILTINS}, namespace)
        func = namespace.get(name)
        if not callable(func):
            err = SkillExecutionError(name, f"did not define a callable {name}()")
            logger.error("Error executing skill %s: %s", name, err.cause)
            raise err
        try:
            return func()
        except Exception as err:  # noqa: BLE001
            logger.exception("Error executing skill %s: %s", name, err)
            raise SkillExecutionError(name, str(err)) from err
