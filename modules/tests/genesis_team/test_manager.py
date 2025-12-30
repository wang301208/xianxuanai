import asyncio
import sys
import time
from pathlib import Path

import os

sys.path.insert(0, os.path.abspath(os.getcwd()))

from evolution import Agent
from evolution.genesis_team.manager import GenesisTeamManager


class DummyAgent(Agent):
    def __init__(self, output: str = "done", delay: float = 0.1):
        self.output = output
        self.delay = delay

    def perform(self) -> str:  # type: ignore[override]
        time.sleep(self.delay)
        return self.output


def test_async_concurrency():
    """Manager should execute agents concurrently to reduce runtime."""

    delay = 0.3
    agents = [DummyAgent(delay=delay) for _ in range(4)]
    manager = GenesisTeamManager(
        sentinel=agents[0],
        archaeologist=agents[1],
        tdd_dev=agents[2],
        qa=agents[3],
        max_workers=4,
    )
    start = time.perf_counter()
    asyncio.run(manager.run())
    duration = time.perf_counter() - start
    assert duration < delay * len(agents)  # would be serial runtime


def test_conflict_resolution():
    """ConflictResolver should roll back when conflicts detected."""

    sentinel = DummyAgent(output="version conflict detected")
    archaeologist = DummyAgent()
    tdd_dev = DummyAgent()
    qa = DummyAgent()
    manager = GenesisTeamManager(
        sentinel=sentinel,
        archaeologist=archaeologist,
        tdd_dev=tdd_dev,
        qa=qa,
        max_workers=1,
    )
    asyncio.run(manager.run())
    assert any("rollback" in d for d in manager.decision_log)
