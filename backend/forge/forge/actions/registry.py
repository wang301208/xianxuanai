import glob
import importlib
import inspect
import os
import sys
from typing import Any, Callable, List, Sequence

import pydantic


class ActionParameter(pydantic.BaseModel):
    """
    This class represents a parameter for an action.

    Attributes:
        name (str): The name of the parameter.
        description (str): A brief description of what the parameter does.
        type (str): The type of the parameter.
        required (bool): A flag indicating whether the parameter is required or optional.
    """

    name: str
    description: str
    type: str
    required: bool


class Action(pydantic.BaseModel):
    """
    This class represents an action in the system.

    Attributes:
        name (str): The name of the action.
        description (str): A brief description of what the action does.
        method (Callable): The method that implements the action.
        parameters (List[ActionParameter]): A list of parameters that the action requires.
        output_type (str): The type of the output that the action returns.
    """

    name: str
    description: str
    method: Callable
    parameters: List[ActionParameter]
    output_type: str
    category: str | None = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        This method allows the class instance to be called as a function.

        Args:
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments.

        Returns:
            Any: The result of the method call.
        """
        return self.method(*args, **kwds)

    def __str__(self) -> str:
        """
        This method returns a string representation of the class instance.

        Returns:
            str: A string representation of the class instance.
        """
        func_summary = f"{self.name}("
        for param in self.parameters:
            func_summary += f"{param.name}: {param.type}, "
        func_summary = func_summary[:-2] + ")"
        func_summary += f" -> {self.output_type}. Usage: {self.description},"
        return func_summary


def action(
    name: str, description: str, parameters: List[ActionParameter], output_type: str
):
    def decorator(func):
        func_params = inspect.signature(func).parameters
        param_names = set(
            [ActionParameter.parse_obj(param).name for param in parameters]
        )
        param_names.add("agent")
        param_names.add("task_id")
        func_param_names = set(func_params.keys())
        if param_names != func_param_names:
            raise ValueError(
                f"Mismatch in parameter names. Action Annotation includes {param_names}, but function actually takes {func_param_names} in function {func.__name__} signature"
            )
        func.action = Action(
            name=name,
            description=description,
            parameters=parameters,
            method=func,
            output_type=output_type,
        )
        return func

    return decorator


class ActionRegister:
    GENERATED_PACKAGE = "forge.actions.generated"

    def __init__(self, agent) -> None:
        self.abilities = {}
        self.generated_dir = os.path.join(os.path.dirname(__file__), "generated")
        self.agent = agent
        self.register_abilities()

    def register_abilities(self) -> None:
        for action_path in glob.glob(
            os.path.join(os.path.dirname(__file__), "**/*.py"), recursive=True
        ):
            if not os.path.basename(action_path) in [
                "__init__.py",
                "registry.py",
            ]:
                action = os.path.relpath(
                    action_path, os.path.dirname(__file__)
                ).replace("/", ".")
                try:
                    module = importlib.import_module(
                        f".{action[:-3]}", package="forge.actions"
                    )
                    self._register_module_actions(
                        module, category=self._category_from_path(action)
                    )
                except Exception as e:
                    print(f"Error occurred while registering abilities: {str(e)}")

    def _register_module_actions(self, module, *, category: str | None = None) -> None:
        for attr in dir(module):
            func = getattr(module, attr)
            if hasattr(func, "action"):
                ab = func.action

                ab.category = category or ab.category or "general"
                self.abilities[func.action.name] = func.action

    def _category_from_path(self, action_path: str, *, fallback: str = "general") -> str:
        parts = action_path.split(".")
        if len(parts) > 1:
            return parts[0].lower().replace("_", " ")
        return fallback

    def register_generated_module(
        self, module_name: str, *, category: str | None = "generated"
    ) -> None:
        """
        Import a generated module and register any actions it exposes.

        This method can be called after new modules are written to disk to
        refresh the registry without rebuilding the agent.
        """

        module_path = f"{self.GENERATED_PACKAGE}.{module_name}"
        try:
            if module_path in sys.modules:
                module = importlib.reload(sys.modules[module_path])
            else:
                module = importlib.import_module(module_path)
            self._register_module_actions(module, category=category)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to register generated module '{module_name}': {exc}"
            ) from exc

    def register_generated_actions_from_source(
        self,
        module_name: str,
        source: str,
        *,
        category: str | None = "generated",
    ) -> None:
        """
        Persist generated action code and register it immediately.

        The source is written into ``forge/actions/generated/<module_name>.py``.
        The generated package is treated like any other action module and may
        contain multiple action definitions.
        """

        os.makedirs(self.generated_dir, exist_ok=True)
        init_path = os.path.join(self.generated_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w", encoding="utf-8"):
                pass

        module_path = os.path.join(self.generated_dir, f"{module_name}.py")
        with open(module_path, "w", encoding="utf-8") as module_file:
            module_file.write(source)

        self.register_generated_module(module_name, category=category)

    def ensure_generated_actions_for_goals(self, goals: Sequence[str]) -> List[str]:
        """
        Ensure actions exist for each goal, synthesizing placeholders when needed.

        This helper delegates to :class:`GeneratedActionHandler` so callers do not
        need to construct the handler manually.
        """

        from .generated_handler import GeneratedActionHandler

        handler = GeneratedActionHandler(self)
        return handler.handle_unfulfilled_goals(goals)

    def list_abilities(self) -> List[Action]:
        return self.abilities

    def list_abilities_for_prompt(self) -> List[str]:
        return [str(action) for action in self.abilities.values()]

    def abilities_description(self) -> str:
        abilities_by_category = {}
        for action in self.abilities.values():
            if action.category not in abilities_by_category:
                abilities_by_category[action.category] = []
            abilities_by_category[action.category].append(str(action))

        abilities_description = ""
        for category, abilities in abilities_by_category.items():
            if abilities_description != "":
                abilities_description += "\n"
            abilities_description += f"{category}:"
            for action in abilities:
                abilities_description += f"  {action}"

        return abilities_description

    async def run_action(
        self, task_id: str, action_name: str, *args: Any, **kwds: Any
    ) -> Any:
        """
        This method runs a specified action with the provided arguments and keyword arguments.

        The agent is passed as the first argument to the action. This allows the action to access and manipulate
        the agent's state as needed.

        Args:
            task_id (str): The ID of the task that the action is being run for.
            action_name (str): The name of the action to run.
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments.

        Returns:
            Any: The result of the action execution.

        Raises:
            Exception: If there is an error in running the action.
        """
        try:
            action = self.abilities[action_name]
            return await action(self.agent, task_id, *args, **kwds)
        except Exception:
            raise


if __name__ == "__main__":
    import sys

    sys.path.append("/Users/swifty/dev/forge/forge")
    register = ActionRegister(agent=None)
    print(register.abilities_description())
    print(register.run_action("abc", "list_files", "/Users/swifty/dev/forge/forge"))
