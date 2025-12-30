from __future__ import annotations

import importlib
import re
import textwrap
from typing import List, Sequence

from modules.skills.executor import SkillSandbox

from .registry import ActionRegister


class GeneratedActionHandler:
    """Handle runtime synthesis and registration of actions.

    The handler inspects requested goals, generates placeholder action modules
    for any missing abilities, validates them within a sandbox, and registers
    them with the :class:`ActionRegister` so they are available immediately.
    """

    def __init__(self, register: ActionRegister, *, sandbox: SkillSandbox | None = None):
        self.register = register
        self.sandbox = sandbox or SkillSandbox(max_workers=1, default_timeout=5)

    def detect_unfulfilled_goals(self, goals: Sequence[str]) -> List[str]:
        """Return goals that do not currently map to a registered action."""

        return [goal for goal in goals if goal not in self.register.abilities]

    def handle_unfulfilled_goals(self, goals: Sequence[str]) -> List[str]:
        """Generate, validate, and register actions for missing goals."""

        missing_goals = self.detect_unfulfilled_goals(goals)
        for goal in missing_goals:
            module_name, func_name, source = self._build_action_source(goal)
            self.register.register_generated_actions_from_source(
                module_name, source, category="generated"
            )
            self._validate_generated_action(module_name, func_name)
        return missing_goals

    def _build_action_source(self, goal: str) -> tuple[str, str, str]:
        module_name = self._sanitize_identifier(goal, suffix="module")
        func_name = self._sanitize_identifier(goal, suffix="action")

        description = (
            "Auto-generated action synthesized to close an unfulfilled goal. "
            "Update the generated module to replace this placeholder logic."
        )

        source = textwrap.dedent(
            f'''
            from forge.actions import action


            @action(
                name="{goal}",
                description="{description}",
                parameters=[
                    {{
                        "name": "context",
                        "description": "Optional context for the synthesized action",
                        "type": "string",
                        "required": False,
                    }}
                ],
                output_type="str",
            )
            async def {func_name}(agent, task_id: str, context: str | None = None) -> str:
                """Auto-generated ability for goal: {goal}."""
                note = f"with context: {context}" if context else "with no additional context"
                return f"Generated action for goal '{goal}' executed {note}"
            '''
        ).lstrip()

        return module_name, func_name, source

    def _sanitize_identifier(self, name: str, *, suffix: str) -> str:
        cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
        if not cleaned:
            cleaned = "generated"
        if cleaned[0].isdigit():
            cleaned = f"gen_{cleaned}"
        return f"{cleaned.lower()}_{suffix}"

    def _validate_generated_action(self, module_name: str, func_name: str) -> None:
        """Run a lightweight sandboxed import/callability check."""

        def _validator(payload, context=None):
            module = importlib.import_module(payload["module"])
            func = getattr(module, payload["func"], None)
            if not callable(func):
                raise RuntimeError(f"Generated action '{payload['func']}' is not callable")
            return True

        self.sandbox.run(
            _validator,
            {
                "module": f"{ActionRegister.GENERATED_PACKAGE}.{module_name}",
                "func": func_name,
            },
            metadata={"name": "generated-action-validation", "category": "generated"},
        )
