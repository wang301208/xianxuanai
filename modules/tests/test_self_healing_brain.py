import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.self_healing import SelfHealingBrain


def test_self_healing_normal_scenario():
    brain = SelfHealingBrain()

    spikes = [0.1, 0.2, 0.1]
    currents = [0.5, 0.5]
    latencies = [0.1, 0.2]
    input_data = {"value": 1, "text": "hello"}
    memory = {"x": 1}

    diagnosis = brain.comprehensive_diagnosis(
        spikes, currents, latencies, copy.deepcopy(input_data), copy.deepcopy(memory)
    )

    assert diagnosis["performance"]["suggestions"] == []
    assert not diagnosis["security"]["input_issues"]
    assert diagnosis["security"]["memory_issues"] == []

    plans = brain.generate_repair_plans(diagnosis)
    assert plans == []

    exec_result = brain.execute_repairs(spikes, currents, latencies, input_data, memory)

    assert exec_result["performance"]["suggestions"] == []
    assert exec_result["performance"]["adjustments"] == []
    assert exec_result["sanitized_input"] == input_data
    assert exec_result["memory_issues"] == []
    assert memory == {"x": 1}


def test_self_healing_detects_and_repairs_anomalies():
    brain = SelfHealingBrain()

    # Simulate previously recorded baseline for memory integrity
    brain.security_guard.memory_checker.baseline = {"a": 1, "b": 2}

    spikes = [0.1, 0.2, 0.3]
    currents = [5.0, 5.0, 5.0]
    latencies = [1.0, 0.8]
    input_data = {"value": 20, "text": "hello trigger"}
    memory = {"a": 1, "b": 3}

    diagnosis = brain.comprehensive_diagnosis(
        spikes, currents, latencies, copy.deepcopy(input_data), copy.deepcopy(memory)
    )

    suggestions = diagnosis["performance"]["suggestions"]
    assert any("energy" in s.lower() for s in suggestions)
    assert any("latency" in s.lower() for s in suggestions)
    assert diagnosis["security"]["input_issues"] is True
    assert diagnosis["security"]["memory_issues"] == ["b"]

    plans = brain.generate_repair_plans(diagnosis)
    assert any("Sanitize" in p for p in plans)
    assert any("Restore memory keys: b" in p for p in plans)

    exec_result = brain.execute_repairs(spikes, currents, latencies, input_data, memory)

    assert exec_result["sanitized_input"]["value"] == 10
    assert "trigger" not in exec_result["sanitized_input"]["text"]
    assert exec_result["memory_issues"] == ["b"]
    assert memory == {"a": 1, "b": 2}
    assert exec_result["performance"]["suggestions"]
    assert exec_result["performance"]["adjustments"]
