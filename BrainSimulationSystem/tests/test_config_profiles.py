"""Tests for configuration profiles and scalable simulation initialization."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.brain_simulation import BrainSimulation
from BrainSimulationSystem.config.default_config import get_config
import ast


def _get_brain_simulation_config_default(field_name: str) -> int:
    module_path = ROOT_DIR / "BrainSimulationSystem" / "core" / "complete_brain_system.py"
    module_ast = ast.parse(module_path.read_text(encoding="utf-8"))

    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == "BrainSimulationConfig":
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == field_name
                    and stmt.value is not None
                ):
                    return ast.literal_eval(stmt.value)
    raise AssertionError(f"Field '{field_name}' not found in BrainSimulationConfig")


def test_full_brain_profile_matches_large_scale_defaults() -> None:
    """The ``full_brain`` profile should align with the large-scale dataclass."""

    cfg = get_config(profile="full_brain")
    total_neurons_default = _get_brain_simulation_config_default("total_neurons")

    assert cfg["scope"]["total_neurons"] == total_neurons_default
    assert cfg["metadata"]["profile"] == "full_brain"


def test_ci_scalable_profile_initializes_simulation() -> None:
    """Ensure the CI-friendly profile loads without exhausting resources."""

    ci_profile = get_config(profile="ci_scalable")

    # 保持可扩展字段但验证资源管理选项可用
    resource_management = ci_profile["runtime"]["resource_management"]
    assert resource_management["backend_selection"]["preferred"] == "native"
    assert resource_management["partitioning"]["autoscale"] is True

    simulation = BrainSimulation({"profile": "ci_scalable"})
    assert simulation.backend.name == "native"
    assert simulation.config["scope"]["total_neurons"] == ci_profile["scope"]["total_neurons"]
    assert simulation.config["metadata"]["profile"] == "ci_scalable"
