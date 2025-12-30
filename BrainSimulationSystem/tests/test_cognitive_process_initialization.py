from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 为测试注入最小的 NeuralNetwork 桩对象，避免导入复杂依赖
import BrainSimulationSystem.core  # type: ignore  # noqa: E402

mock_network_module = ModuleType("BrainSimulationSystem.core.network")


class _StubNeuralNetwork:  # pragma: no cover - 简化的桩对象
    pass


mock_network_module.NeuralNetwork = _StubNeuralNetwork
sys.modules.setdefault("BrainSimulationSystem.core.network", mock_network_module)

from BrainSimulationSystem.models.attention import AttentionProcess, AttentionSystem
from BrainSimulationSystem.models.cognitive_controller import CognitiveController
from BrainSimulationSystem.models.decision import DecisionProcess
from BrainSimulationSystem.models.perception import PerceptionProcess
from BrainSimulationSystem.models.working_memory import (
    AChModulatedWorkingMemory,
    WorkingMemory,
)


class DummyNetwork:
    def __init__(self):
        self.layers = {}
        self.neurons = {}
        self.input_layer_name = None
        self.last_input = None

    def set_input(self, values):
        self.last_input = values


@pytest.mark.parametrize(
    "process_cls",
    [
        PerceptionProcess,
        AttentionSystem,
        AttentionProcess,
        DecisionProcess,
        WorkingMemory,
        AChModulatedWorkingMemory,
    ],
)
def test_cognitive_process_initialization(process_cls):
    network = DummyNetwork()
    params = {"example": process_cls.__name__}

    process = process_cls(network, params)

    assert process.network is network
    assert process.params["example"] == process_cls.__name__


def test_cognitive_controller_initialization():
    network = DummyNetwork()
    params = {
        "attention": {"ach_sensitivity": 0.75},
        "working_memory": {"capacity": 3},
    }

    controller = CognitiveController(network, params)

    assert controller.network is network
    assert controller.params == params
    assert "attention" in controller.components
    assert controller.components["attention"].network is network
    assert controller.components["working_memory"].params["capacity"] == 3
