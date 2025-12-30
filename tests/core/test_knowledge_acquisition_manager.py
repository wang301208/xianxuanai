from types import SimpleNamespace

import pytest

from modules.knowledge.acquisition import KnowledgeAcquisitionManager


class DummyMemory:
    def __init__(self) -> None:
        self.entries: list[str] = []

    def add(self, item: str) -> None:
        self.entries.append(item)


def _build_ability_spec(name: str = "web_search"):
    return SimpleNamespace(name=name, parameters={"query": {}, "context": {}})


def _build_task(description: str = "Explain quantum spin ice.") -> SimpleNamespace:
    return SimpleNamespace(description=description, objective=description)


def _build_result(
    ability_name: str,
    ability_args: dict,
    *,
    success: bool = True,
    message: str = "",
):
    return SimpleNamespace(
        ability_name=ability_name,
        ability_args=ability_args,
        success=success,
        message=message,
        new_knowledge=None,
    )


def test_maybe_acquire_creates_session_and_plan() -> None:
    manager = KnowledgeAcquisitionManager(confidence_threshold=0.6)
    ability_spec = _build_ability_spec()
    metadata = {"confidence": 0.3, "reason": "Initial response below threshold."}
    task = _build_task("Outline the basics of reinforcement learning.")

    override = manager.maybe_acquire(
        metadata=metadata,
        ability_specs=[ability_spec],
        task=task,
        current_selection=None,
    )

    assert override is not None, "Knowledge acquisition should trigger on low confidence."
    assert override["next_ability"] == ability_spec.name
    assert override["knowledge_acquisition"] is True
    session_id = override["knowledge_session_id"]
    assert session_id

    plan = override["knowledge_plan"]
    assert plan["query"] == "Outline the basics of reinforcement learning."
    assert plan["steps"], "Plan should include at least one search step."
    assert plan["steps"][0]["status"] == "pending"
    assert plan["metadata"]["confidence"] == pytest.approx(0.3)


def test_complete_session_records_history_and_memory() -> None:
    manager = KnowledgeAcquisitionManager(confidence_threshold=0.8)
    ability_spec = _build_ability_spec()
    metadata = {"confidence": 0.2}
    task = _build_task("Define artificial intelligence.")

    override = manager.maybe_acquire(
        metadata=metadata,
        ability_specs=[ability_spec],
        task=task,
        current_selection=None,
    )
    assert override is not None
    session_id = override["knowledge_session_id"]

    manager.mark_session_started(session_id)

    ability_result = _build_result(
        ability_spec.name,
        override["ability_arguments"],
        message="Artificial intelligence refers to computational systems that perform tasks requiring human-like cognition.",
    )

    memory = DummyMemory()
    log_entry = manager.complete_session(
        session_id,
        ability_result,
        metadata={"confidence": 0.2},
        memory=memory,
    )

    assert log_entry is not None
    assert log_entry["success"] is True
    assert log_entry["plan"]["steps"][0]["status"] == "completed"
    assert session_id == log_entry["session_id"]
    assert manager.history()[-1]["session_id"] == session_id
    assert any("Knowledge acquisition succeeded" in entry for entry in memory.entries)
