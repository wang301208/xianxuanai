"""
Standardised interfaces and message schemas for intra-module communication.

This module defines a lightweight publish-subscribe message bus together with
common signal descriptors that individual subsystems can use to exchange data
without tightly coupling to each other's internal structures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


class ModuleTopic(Enum):
    """Canonical topics used by the brain simulation subsystems."""

    SENSORY_VISUAL = "sensory.visual"
    SENSORY_SOMATOSENSORY = "sensory.somatosensory"
    SENSORY_AUDITORY = "sensory.auditory"
    MICROCIRCUIT_EVENT = "microcircuit.event"
    MICROCIRCUIT_COMMAND = "microcircuit.command"
    COGNITIVE_STATE = "cognitive.state"
    MEMORY_EVENT = "memory.event"
    EMOTION_STATE = "affect.state"
    MOTOR_PLAN = "motor.plan"
    MOTOR_COMMAND = "motor.command"
    MOTOR_FEEDBACK = "motor.feedback"
    CONTROL_TOP_DOWN = "control.top_down"
    CONTROL_BOTTOM_UP = "control.bottom_up"
    ENVIRONMENT_EVENT = "environment.event"


@dataclass
class ModuleSignal:
    """Container describing a single message exchanged between modules."""

    topic: ModuleTopic
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=lambda: time.time())
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a serialisable dictionary."""
        data = asdict(self)
        data["topic"] = self.topic.value
        return data


@dataclass
class ModulePortDescriptor:
    """Descriptor for module input/output interfaces."""

    topic: ModuleTopic
    description: str


class ModuleInterface:
    """
    Base class for modules that communicate via the standardised interface.

    Subclasses should override :meth:`process` to consume incoming signals and
    produce a list of outgoing signals that will be published on the bus.
    """

    inputs: Iterable[ModulePortDescriptor] = ()
    outputs: Iterable[ModulePortDescriptor] = ()

    def process(self, signals: Iterable[ModuleSignal]) -> List[ModuleSignal]:
        raise NotImplementedError


class ModuleBus:
    """Publish/subscribe message bus for module communication."""

    def __init__(self) -> None:
        self._topics: Dict[ModuleTopic, List[ModuleSignal]] = {}
        self._subscribers: Dict[ModuleTopic, List[Callable[[ModuleSignal], None]]] = {}
        self._current_time: float = 0.0

    def register_topic(self, topic: ModuleTopic) -> None:
        """Ensure internal bookkeeping exists for a topic."""
        self._topics.setdefault(topic, [])
        self._subscribers.setdefault(topic, [])

    def reset_cycle(self, current_time: float) -> None:
        """Clear previous signals in preparation for a new simulation step."""
        self._current_time = current_time
        for topic in self._topics:
            self._topics[topic] = []

    def publish(self, signal: ModuleSignal) -> None:
        """Publish a signal and notify subscribers."""
        if signal.topic not in self._topics:
            self.register_topic(signal.topic)
        self._topics[signal.topic].append(signal)
        for callback in self._subscribers.get(signal.topic, []):
            try:
                callback(signal)
            except Exception:
                # Subscribers are auxiliary; errors should not propagate.
                continue

    def subscribe(self, topic: ModuleTopic, callback: Callable[[ModuleSignal], None]) -> None:
        """Register a callback that will receive signals on the given topic."""
        if topic not in self._topics:
            self.register_topic(topic)
        self._subscribers[topic].append(callback)

    def get_signals(self, topic: ModuleTopic) -> List[ModuleSignal]:
        """Return all signals published for the given topic in the current cycle."""
        return list(self._topics.get(topic, []))

    def export_cycle(self) -> Dict[str, Any]:
        """Export the current cycle's signal log as a serialisable dictionary."""
        return {
            "timestamp": self._current_time,
            "topics": {
                topic.value: [signal.to_dict() for signal in signals]
                for topic, signals in self._topics.items()
            },
        }

