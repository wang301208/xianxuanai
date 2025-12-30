import importlib.util
from pathlib import Path

import pytest

from autogpt.core.ability.base import AbilityRegistry, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.planning.schema import Task, TaskType


def _load_capability_agent():
    """Load ``CapabilityAgent`` without importing the whole agents package."""

    base_path = Path(__file__).resolve().parents[1] / ".." / "autogpt"

    # Prepare stub for autogpt.core.agent to avoid heavy imports
    import types, sys

    base_module_spec = importlib.util.spec_from_file_location(
        "autogpt.core.agent.base", (base_path / "core" / "agent" / "base.py").resolve()
    )
    base_module = importlib.util.module_from_spec(base_module_spec)
    assert base_module_spec and base_module_spec.loader
    base_module_spec.loader.exec_module(base_module)
    sys.modules["autogpt.core.agent.base"] = base_module

    layer_path = base_path / "core" / "agent" / "layered.py"
    layer_spec = importlib.util.spec_from_file_location(
        "autogpt.core.agent.layered", layer_path.resolve()
    )
    layer_module = importlib.util.module_from_spec(layer_spec)
    assert layer_spec and layer_spec.loader
    core_agent_pkg = types.ModuleType("autogpt.core.agent")
    sys.modules["autogpt.core.agent"] = core_agent_pkg
    sys.modules["autogpt.core.agent.layered"] = layer_module
    layer_spec.loader.exec_module(layer_module)
    core_agent_pkg.layered = layer_module

    module_path = base_path / "agents" / "layers" / "capability.py"
    module_spec = importlib.util.spec_from_file_location(
        "autogpt.agents.layers.capability", module_path.resolve()
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec and module_spec.loader
    module_spec.loader.exec_module(module)
    return module.CapabilityAgent


CapabilityAgent = _load_capability_agent()


class DummyRegistry(AbilityRegistry):
    def __init__(self) -> None:
        self.performed = []

    def register_ability(self, ability_name: str, ability_configuration: AbilityConfiguration) -> None:
        pass

    def list_abilities(self) -> list[str]:
        return ["write_file", "search_web"]

    def dump_abilities(self):
        return []

    def get_ability(self, ability_name: str):
        raise NotImplementedError

    async def perform(self, ability_name: str, **kwargs) -> AbilityResult:
        self.performed.append((ability_name, kwargs))
        return AbilityResult(
            ability_name=ability_name,
            ability_args=kwargs,
            success=True,
            message="ok",
        )


@pytest.mark.asyncio
async def test_capability_agent_selects_matching_ability():
    registry = DummyRegistry()
    feedback = []
    agent = CapabilityAgent(
        registry, feedback_handler=lambda name, res: feedback.append((name, res.success))
    )

    task = Task(
        objective="Write some code",
        type=TaskType.WRITE,
        priority=1,
        ready_criteria=[],
        acceptance_criteria=[],
    )

    plan, result = await agent.determine_next_ability(task)

    assert plan["next_ability"] == "write_file"
    assert result.success
    assert feedback == [("write_file", True)]
