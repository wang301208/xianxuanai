import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.brain.message_bus import (
    CircuitBreaker,
    publish_neural_event,
    reset_message_bus,
    subscribe_to_brain_region,
)


def test_publish_and_subscribe() -> None:
    reset_message_bus()
    received: list[str] = []

    subscribe_to_brain_region("motor", lambda e: received.append(e["payload"]))
    publish_neural_event({"target": "motor", "payload": "step"})

    assert received == ["step"]


def test_circuit_breaker_isolates_failures() -> None:
    reset_message_bus()
    calls: list[int] = []

    def failing_handler(event):
        calls.append(1)
        raise ValueError("boom")

    subscribe_to_brain_region("visual", failing_handler)
    limit = CircuitBreaker().max_failures

    for _ in range(limit + 1):
        publish_neural_event({"target": "visual", "payload": None})

    assert len(calls) == limit
