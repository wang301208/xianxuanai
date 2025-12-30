import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.security import NeuralSecurityGuard


def test_validate_neural_input_handles_threats():
    guard = NeuralSecurityGuard()
    data = {"values": [0.2, 99.0], "text": "hello trigger"}
    cleaned = guard.validate_neural_input(data)
    assert max(cleaned["values"]) <= guard.adversarial_detector.threshold
    assert "trigger" not in cleaned["text"]


def test_memory_checker_repairs_corruption():
    guard = NeuralSecurityGuard()
    memory = {"status": "ok"}
    assert guard.protect_neural_memory(memory) == []
    memory["status"] = "tampered"
    issues = guard.protect_neural_memory(memory)
    assert issues == ["status"]
    assert memory["status"] == "ok"


def test_monitor_and_harden_isolates_and_recovers():
    guard = NeuralSecurityGuard()
    memory = {"status": "ok"}
    data = {"values": [0.2, 99.0], "text": "hello trigger"}
    cleaned, issues = guard.monitor_and_harden(data, memory)
    assert issues["adversarial"]
    assert issues["backdoor"]
    assert issues["memory"] == []
    assert not guard.isolated
    assert max(cleaned["values"]) <= guard.adversarial_detector.threshold
    assert "trigger" not in cleaned["text"]


def test_monitor_and_harden_repairs_memory_and_recovers():
    guard = NeuralSecurityGuard()
    memory = {"status": "ok"}
    # establish baseline
    guard.monitor_and_harden({}, memory)
    memory["status"] = "tampered"
    cleaned, issues = guard.monitor_and_harden({}, memory)
    assert issues["memory"] == ["status"]
    assert not guard.isolated
    assert memory["status"] == "ok"
