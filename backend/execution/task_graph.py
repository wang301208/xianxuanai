from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Task:
    """Represents a unit of work and its dependencies."""

    description: str
    skill: str | None = None
    dependencies: List[str] = field(default_factory=list)


class TaskGraph:
    """Simple task graph supporting dependency resolution."""

    def __init__(self) -> None:
        self.tasks: Dict[str, Task] = {}

    def add_task(
        self,
        task_id: str,
        description: str,
        *,
        skill: str | None = None,
        dependencies: List[str] | None = None,
    ) -> None:
        self.tasks[task_id] = Task(description, skill, dependencies or [])

    def execution_order(self) -> List[str]:
        """Return a list of task ids in executable order."""
        visited: set[str] = set()
        temp_mark: set[str] = set()
        order: List[str] = []

        def visit(node: str) -> None:
            if node in visited:
                return
            if node in temp_mark:
                raise ValueError("Cycle detected in task graph")
            temp_mark.add(node)
            for dep in self.tasks[node].dependencies:
                visit(dep)
            temp_mark.remove(node)
            visited.add(node)
            order.append(node)

        for node in list(self.tasks.keys()):
            visit(node)
        return order
