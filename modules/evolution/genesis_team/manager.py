"""Manager coordinating Genesis team agents."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List

from .sentinel import Sentinel
from .archaeologist import Archaeologist
from .tdd_dev import TDDDeveloper
from .qa import QA
from .conflict import ConflictResolver


@dataclass
class _AgentInfo:
    """Container holding agent execution metadata."""

    agent: object
    capability: int = 1
    load: int = 0


@dataclass
class GenesisTeamManager:
    """Orchestrates the Genesis team agents.

    The manager instantiates each agent and provides an asynchronous
    :meth:`run` method which schedules agents on a capability-aware task
    queue. Agents are executed concurrently and, after each completes, a
    conflict resolution step is triggered to merge or roll back results.
    Decision summaries are stored in :attr:`decision_log`.
    """

    sentinel: Sentinel = field(default_factory=Sentinel)
    archaeologist: Archaeologist = field(default_factory=Archaeologist)
    tdd_dev: TDDDeveloper = field(default_factory=TDDDeveloper)
    qa: QA = field(default_factory=QA)
    max_workers: int = 4
    decision_log: List[str] = field(default_factory=list)
    conflict: ConflictResolver = field(default_factory=ConflictResolver)

    def __post_init__(self) -> None:
        self._agents: Dict[str, _AgentInfo] = {
            "sentinel": _AgentInfo(
                self.sentinel,
                getattr(self.sentinel, "capability", 1),
                getattr(self.sentinel, "load", 0),
            ),
            "archaeologist": _AgentInfo(
                self.archaeologist,
                getattr(self.archaeologist, "capability", 1),
                getattr(self.archaeologist, "load", 0),
            ),
            "tdd_developer": _AgentInfo(
                self.tdd_dev,
                getattr(self.tdd_dev, "capability", 1),
                getattr(self.tdd_dev, "load", 0),
            ),
            "qa": _AgentInfo(
                self.qa,
                getattr(self.qa, "capability", 1),
                getattr(self.qa, "load", 0),
            ),
        }
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def _priority(self, info: _AgentInfo) -> float:
        """Compute scheduling priority based on capability and load."""

        return info.load / info.capability

    async def run(self) -> Dict[str, str]:
        """Execute all agents concurrently and return their logs."""

        logs: Dict[str, str] = {}
        queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        for name, info in self._agents.items():
            queue.put_nowait((self._priority(info), name, info))

        async def worker() -> None:
            while True:
                priority, agent_name, info = await queue.get()
                if agent_name is None:
                    queue.task_done()
                    break
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor, info.agent.perform
                )
                logs[agent_name] = result
                decision = self.conflict.resolve(agent_name, logs)
                self.decision_log.append(decision)
                queue.task_done()

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(self.max_workers, len(self._agents)))
        ]
        await queue.join()
        for _ in workers:
            await queue.put((float("inf"), None, None))
        await asyncio.gather(*workers)
        return logs

    def shutdown(self) -> None:
        """Shut down the shared thread pool executor."""

        self._executor.shutdown(wait=True)
