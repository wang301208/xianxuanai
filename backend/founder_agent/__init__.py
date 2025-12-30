"""Monitoring loop for the founder agent.

This package collects system metrics from the event bus, performs
trend analysis, and periodically generates blueprint proposals.
"""
from __future__ import annotations

import time
from typing import Sequence

from events import EventBus

from .analytics import Analytics
from .proposal_generator import generate_proposal


DEFAULT_TOPICS: Sequence[str] = ("system.metrics",)


def run_monitoring_loop(
    event_bus: EventBus,
    topics: Sequence[str] = DEFAULT_TOPICS,
    interval: float = 60.0,
) -> None:
    """Subscribe to metric *topics* and periodically emit proposals.

    Args:
        topics: Event bus topics that publish metric dictionaries.
        interval: Seconds between proposal generations.
    """
    analytics = Analytics()
    subscriptions = [
        event_bus.subscribe(topic, analytics.handle_event) for topic in topics
    ]
    try:
        while True:
            time.sleep(interval)
            trends = analytics.get_trends()
            if trends:
                generate_proposal(trends)
    finally:
        for cancel in subscriptions:
            cancel()
