import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))

import asyncio

from evolution.genesis_team import GenesisTeamManager, Sentinel, Archaeologist, TDDDeveloper, QA


def test_manager_runs_agents_in_order():
    call_order: list[str] = []

    class DummySentinel(Sentinel):
        def perform(self) -> str:  # type: ignore[override]
            call_order.append("sentinel")
            return "sentinel done"

    class DummyArchaeologist(Archaeologist):
        def perform(self, max_entries: int = 5) -> str:  # type: ignore[override]
            call_order.append("archaeologist")
            return "archaeologist done"

    class DummyTDD(TDDDeveloper):
        def perform(self, test_cmd: str = "pytest") -> str:  # type: ignore[override]
            call_order.append("tdd_developer")
            return "tdd developer done"

    class DummyQA(QA):
        def perform(self) -> str:  # type: ignore[override]
            call_order.append("qa")
            return "qa done"

    manager = GenesisTeamManager(
        sentinel=DummySentinel(),
        archaeologist=DummyArchaeologist(),
        tdd_dev=DummyTDD(),
        qa=DummyQA(),
        max_workers=1,
    )
    logs = asyncio.run(manager.run())
    manager.shutdown()

    assert set(call_order) == {"sentinel", "archaeologist", "tdd_developer", "qa"}
    assert set(logs.keys()) == {"sentinel", "archaeologist", "tdd_developer", "qa"}
    assert logs["qa"] == "qa done"
