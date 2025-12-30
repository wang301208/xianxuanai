import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.motor.precision import PrecisionMotorSystem


def test_planning_and_execution_flow():
    system = PrecisionMotorSystem()
    plan = system.plan_movement("reach target")
    assert "[BG]" in plan
    assert "optimized for obstacles and forces" in plan
    result = system.execute_action(plan)
    assert result.startswith("executed")
    assert "fine-tuned" in result
    assert "[BG]" in result


def test_cerebellar_learning_updates():
    system = PrecisionMotorSystem()
    system.learn("offset")
    plan = system.plan_movement("move")
    result = system.execute_action(plan)
    assert "offset" in result


def test_basal_ganglia_modulation():
    system = PrecisionMotorSystem()
    plan = system.plan_movement("raise hand")
    system.execute_action(plan)
    assert len(system.basal_ganglia.gating_history) == 2


def test_precision_system_accepts_metric_feedback():
    system = PrecisionMotorSystem()
    metrics = {"velocity_error": 0.2, "stability_error": 0.1, "reward": 0.5}
    system.update_feedback(metrics)
    assert system.cerebellum.metric_history
    stored = system.cerebellum.metric_history[-1]
    assert stored["velocity_error"] == 0.2
    assert stored["stability_error"] == 0.1
