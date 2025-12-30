import os
import sys
from unittest.mock import Mock

import pytest

# Ensure repository root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.quantum.quantum_cognition import QuantumCognition
from modules.brain.quantum.hardware_interface import QuantumHardwareInterface


class DummyCircuit:
    def __init__(self, counts=None):
        self._counts = counts or {"00": 1024}

    def simulate(self, shots=1024):
        # Return a copy so tests can mutate without affecting internal state
        return dict(self._counts)


def test_local_simulation_task_queue():
    qc = QuantumCognition()
    circuit = DummyCircuit({"00": 100})
    qc.add_task(circuit, shots=100)
    results = qc.run_tasks()
    assert results == [{"00": 100}]


def test_hardware_backend_task_queue():
    # Create a mocked hardware interface
    mock_iface = Mock(spec=QuantumHardwareInterface)
    raw_result = object()
    mock_iface.run.return_value = raw_result
    mock_iface.parse_counts.return_value = {"11": 7}

    qc = QuantumCognition(hardware=mock_iface, backend="qiskit")
    qc.add_task("circuit", shots=7)
    results = qc.run_tasks()

    mock_iface.run.assert_called_once_with("circuit", shots=7)
    mock_iface.parse_counts.assert_called_once_with(raw_result)
    assert results == [{"11": 7}]
