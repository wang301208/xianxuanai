"""Tests for MetaCognitionController."""

from modules.meta_cognition import MetaCognitionController


def test_meta_controller_flags_repeated_failures():
    controller = MetaCognitionController(failure_threshold=2)
    controller.record_task_outcome("task-1", False)
    controller.record_task_outcome("task-1", False)

    signals = controller.analyse()

    kinds = {signal.kind for signal in signals}
    assert "replan_task" in kinds
    assert "schedule_skill_learning" in kinds


def test_meta_controller_flags_low_confidence():
    controller = MetaCognitionController(low_confidence_window=2, low_confidence_threshold=0.4)
    controller.record_task_outcome("task-1", True, metadata={"confidence": 0.1})
    controller.record_task_outcome("task-1", True, metadata={"confidence": 0.2})

    signals = controller.analyse()

    kinds = {signal.kind for signal in signals}
    assert "request_human_feedback" in kinds


def test_meta_controller_flags_knowledge_gaps():
    controller = MetaCognitionController(knowledge_gap_threshold=2)
    controller.record_task_outcome("task-1", False, metadata={"knowledge_gap": True})
    controller.record_task_outcome("task-1", False, metadata={"knowledge_gap": True})

    signals = controller.analyse()
    kinds = {signal.kind for signal in signals}
    assert "schedule_knowledge_learning" in kinds


def test_meta_controller_flags_regressions_trigger_evolution():
    controller = MetaCognitionController()
    controller.record_regressions([{"latest_version": 3, "reasons": ["resource_decline"]}])

    signals = controller.analyse()
    kinds = {signal.kind for signal in signals}
    assert "trigger_evolution" in kinds
