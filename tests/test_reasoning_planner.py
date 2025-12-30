import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.reasoning.planner import ReasoningPlanner
from backend.reasoning.solvers import RuleProbabilisticSolver


class DummySource:
    def __init__(self):
        self.calls = 0

    def query(self, statement: str):
        self.calls += 1
        return [f"{statement}_evidence"]


class SlowSource(DummySource):
    def __init__(self, delay: float = 0.1):
        super().__init__()
        self.delay = delay

    def query(self, statement: str):
        self.calls += 1
        time.sleep(self.delay)
        return [f"{statement}_evidence"]


def test_planner_caching_and_explanation():
    source = DummySource()
    solver = RuleProbabilisticSolver({"A": [("B", 0.7)]})
    planner = ReasoningPlanner([source], solver)

    first = planner.infer("A")
    second = planner.infer("A")

    assert first == ("B", 0.7)
    assert second == ("B", 0.7)
    assert source.calls == 1
    explanation = planner.explain()
    assert explanation[0]["statement"] == "A"
    assert explanation[0]["conclusion"] == "B"
    assert explanation[0]["probability"] == 0.7


def test_planner_chain():
    source = SlowSource(0.1)
    solver = RuleProbabilisticSolver({"A": [("B", 0.7)], "B": [("C", 0.5)]})
    planner = ReasoningPlanner([source], solver)

    statements = ["A", "B"]

    start = time.perf_counter()
    results = planner.chain(statements)
    duration = time.perf_counter() - start

    assert results == [("B", 0.7), ("C", 0.5)]
    assert source.calls == 2
    assert duration < 0.19

    source.calls = 0
    repeat = planner.chain(statements)
    assert repeat == results
    assert source.calls == 0


def test_infer_batch_performance_and_cache():
    statements = ["A", "B", "C"]
    rules = {"A": [("B", 0.7)], "B": [("C", 0.5)], "C": [("D", 0.4)]}

    seq_source = SlowSource(0.1)
    seq_planner = ReasoningPlanner([seq_source], RuleProbabilisticSolver(rules))
    start = time.perf_counter()
    seq_results = [seq_planner.infer(s) for s in statements]
    seq_duration = time.perf_counter() - start

    batch_source = SlowSource(0.1)
    batch_planner = ReasoningPlanner([batch_source], RuleProbabilisticSolver(rules))
    start = time.perf_counter()
    batch_results = batch_planner.infer_batch(statements)
    batch_duration = time.perf_counter() - start

    assert batch_results == seq_results == [("B", 0.7), ("C", 0.5), ("D", 0.4)]
    assert batch_duration < seq_duration
    assert batch_source.calls == len(statements)

    batch_source.calls = 0
    repeat = batch_planner.infer_batch(statements)
    assert repeat == batch_results
    assert batch_source.calls == 0
