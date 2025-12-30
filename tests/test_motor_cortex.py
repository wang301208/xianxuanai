import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain import MotorCortex
from modules.brain.motor.actions import MotorPlan
from modules.brain.neuromorphic.spiking_network import SpikingNeuralNetwork


class StubBasalGanglia:
    def modulate(self, plan: str) -> str:
        return plan + " modulated"


class StubCerebellum:
    def fine_tune(self, command):
        if isinstance(command, str):
            return command + " tuned"
        return command.with_updates(metadata={"stub": True})


def test_motor_cortex_plan_execute():
    cortex = MotorCortex(basal_ganglia=StubBasalGanglia(), cerebellum=StubCerebellum())
    plan = cortex.plan_movement("wave")
    assert isinstance(plan, MotorPlan)
    assert plan.command is not None
    assert any("modulated" in stage for stage in plan.stages)

    result = cortex.execute_action(plan)
    assert isinstance(result, str)
    assert "executed" in result and "tuned" in result


def test_motor_cortex_spiking_backend():
    snn = SpikingNeuralNetwork(2, weights=[[0.0, 1.0], [0.0, 0.0]])
    initial = snn.synapses.weights[0][1]
    cortex = MotorCortex(spiking_backend=snn)
    cortex.execute_action([[1, 0], [0, 1]])
    assert snn.synapses.weights[0][1] != initial
