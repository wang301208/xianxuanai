"""Smoke tests for the BrainSimulation entry point."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.brain_simulation import BrainSimulation
from BrainSimulationSystem.core.network import NeuralNetwork


def test_brain_simulation_initializes_with_default_config():
    simulation = BrainSimulation()

    assert isinstance(simulation.network, NeuralNetwork)
    assert simulation.backend.name == "native"
    assert simulation.config  # configuration should not be empty


def test_brain_simulation_step_propagates_sensory_input_to_network_buffer():
    simulation = BrainSimulation()
    sensory_data = [0.2, 0.5, 0.8]

    result = simulation.step({"sensory_data": sensory_data}, dt=0.1)

    perception_output = result["cognitive_state"]["perception"]["perception_output"]
    assert perception_output, "Perception output should not be empty for non-empty stimulus"
    assert simulation.network._input_buffer == perception_output
