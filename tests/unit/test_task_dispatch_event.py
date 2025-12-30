from modules.events.coordination import TaskDispatchEvent


def test_task_dispatch_event_to_dict_omits_absent_optional_fields():
    event = TaskDispatchEvent(task_id="task-123", payload={"work": "do"})

    serialized = event.to_dict()

    assert serialized["task_id"] == "task-123"
    assert serialized["payload"] == {"work": "do"}
    assert "metadata" not in serialized
    assert "routed" not in serialized
    # assigned_to remains explicit so downstream consumers can detect
    # coordinator expectations about ownership.
    assert "assigned_to" in serialized and serialized["assigned_to"] is None


def test_task_dispatch_event_to_dict_retains_metadata_and_routed_flags():
    event = TaskDispatchEvent(
        task_id="task-456",
        payload={"work": "redo"},
        assigned_to="agent-7",
        metadata={"attempt": 2},
        routed=True,
    )

    serialized = event.to_dict()

    assert serialized["metadata"] == {"attempt": 2}
    assert serialized["routed"] is True
    assert serialized["assigned_to"] == "agent-7"
