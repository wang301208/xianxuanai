"""Benchmark publish latency for InMemoryEventBus."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from events import InMemoryEventBus, publish, subscribe


def _handler(event):
    time.sleep(0.001)


def measure(bus: InMemoryEventBus, n: int = 100) -> float:
    unsub = subscribe(bus, "topic", _handler)
    start = time.perf_counter()
    for i in range(n):
        publish(bus, "topic", {"i": i})
    end = time.perf_counter()
    if hasattr(bus, "join"):
        bus.join()
    unsub()
    return (end - start) / n


if __name__ == "__main__":
    latency = measure(InMemoryEventBus())
    print(f"Average publish latency: {latency*1000:.3f} ms")
