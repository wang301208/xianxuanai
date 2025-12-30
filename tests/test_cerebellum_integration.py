import os
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain import Cerebellum, MotorCortex
from modules.brain.motor.actions import MotorCommand, MotorExecutionResult


def _simulate_error_reduction(
    command: MotorCommand, tuned: MotorCommand, telemetry: Dict[str, float]
) -> Dict[str, float]:
    """Compute simulated post-adjustment error magnitudes."""

    errors_after: Dict[str, float] = {}
    for metric, error in telemetry.items():
        if not metric.endswith("_error"):
            continue
        arg_name = metric[: -len("_error")]
        original_value = command.arguments.get(arg_name)
        tuned_value = tuned.arguments.get(arg_name)
        if isinstance(original_value, (int, float)) and isinstance(tuned_value, (int, float)):
            applied_change = float(original_value) - float(tuned_value)
            errors_after[metric] = float(error) - applied_change
    return errors_after


def test_cerebellum_motor_learning_and_refinement_reduces_error():
    cerebellum = Cerebellum()
    command = MotorCommand(
        tool="arm",
        operation="reach",
        arguments={"position": 1.0, "velocity": 0.5},
        metadata={},
    )
    telemetry = {"position_error": 0.32, "velocity_error": -0.18}
    feedback = MotorExecutionResult(success=False, output=None, telemetry=telemetry, error="overshoot")

    learning_summary = cerebellum.motor_learning(feedback)
    assert learning_summary["telemetry"]["position_error"] == telemetry["position_error"]
    assert learning_summary["telemetry"]["velocity_error"] == telemetry["velocity_error"]
    assert learning_summary["filtered_error"]
    assert learning_summary["corrections"]

    tuned_command = cerebellum.fine_tune(command)
    assert isinstance(tuned_command, MotorCommand)
    errors_after = _simulate_error_reduction(command, tuned_command, telemetry)
    for metric, error_before in telemetry.items():
        if metric.endswith("_error") and metric in errors_after:
            assert abs(errors_after[metric]) < abs(error_before)
    cerebellum_hint = cerebellum.fine_tune("status")
    assert isinstance(cerebellum_hint, str)
    assert cerebellum_hint


def test_cerebellum_balance_control_structured_feedback():
    cerebellum = Cerebellum()
    sensory = {"tilt": 0.2, "acceleration": -0.05}
    response = cerebellum.balance_control(sensory)
    assert "adjustments" in response
    assert isinstance(response["filtered"], dict)
    assert set(response["adjustments"]).issubset({"tilt", "acceleration"})

    text_response = cerebellum.balance_control("tilt:0.1 sway:-0.02")
    assert "adjustments" in text_response


def test_motor_cortex_training_integrates_structured_feedback():
    cerebellum = Cerebellum()
    cortex = MotorCortex(cerebellum=cerebellum)
    feedback = MotorExecutionResult(
        success=False,
        output=None,
        telemetry={"position_error": 0.25, "stability_error": 0.15},
        error="drift",
    )
    report = cortex.train(feedback)
    assert isinstance(report, dict)
    assert report["telemetry"]["position_error"] == 0.25
    assert cerebellum.metric_history

    tuned = cortex.cerebellum.fine_tune(
        MotorCommand(tool="arm", operation="reach", arguments={"position": 0.8}, metadata={})
    )
    assert isinstance(tuned, MotorCommand)
    assert tuned.metadata["cerebellum"]["applied_corrections"]
