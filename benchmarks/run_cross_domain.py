"""Cross-domain reasoning benchmark suite.

This module defines a simple benchmark that runs tasks from two domains:
logic puzzles and knowledge retrieval. It demonstrates basic self-reflection
between tasks by adapting its strategy based on earlier results and records
performance metrics for each task.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List


@dataclass
class TaskResult:
    """Container for benchmark task outcomes."""

    domain: str
    success: bool
    duration: float
    strategy: str


class CrossDomainBenchmark:
    """Run reasoning tasks from multiple domains with simple self-reflection."""

    def __init__(self) -> None:
        # default strategy for knowledge retrieval
        self.strategy = {"retrieval": "case-sensitive"}
        self.results: List[TaskResult] = []

    # --- Domain specific tasks -------------------------------------------------
    def _solve_logic_puzzle(self) -> None:
        start = time.perf_counter()

        # A tiny syllogism puzzle
        premise1 = "All men are mortal"
        premise2 = "Socrates is a man"
        conclusion = "Socrates is mortal"

        # naive deduction
        deduced = "Socrates" in premise2 and "men" in premise1
        success = deduced and conclusion.endswith("mortal")

        duration = time.perf_counter() - start
        self.results.append(
            TaskResult("logic", success, duration, "deduction")
        )

        # Reflect on the task and adjust retrieval strategy for the next task
        self.strategy["retrieval"] = "case-insensitive"

    def _knowledge_retrieval(self, query: str) -> None:
        start = time.perf_counter()
        data = {"paris": "France", "tokyo": "Japan"}
        strategy = self.strategy["retrieval"]

        if strategy == "case-sensitive":
            result = data.get(query)
        else:
            result = data.get(query.lower())
        success = result == "France"

        duration = time.perf_counter() - start
        self.results.append(
            TaskResult("knowledge", success, duration, strategy)
        )

        # Reflect: if retrieval failed, fall back to case-insensitive for future
        if not success:
            self.strategy["retrieval"] = "case-insensitive"

    # --- Public API -----------------------------------------------------------
    def run(self) -> List[TaskResult]:
        """Execute the benchmark across domains and return performance data."""

        self.results.clear()
        self._solve_logic_puzzle()
        self._knowledge_retrieval("Paris")
        return self.results


def main() -> None:
    benchmark = CrossDomainBenchmark()
    results = benchmark.run()

    for r in results:
        print(
            f"{r.domain:9s} | success={r.success!s:5s} | "
            f"strategy={r.strategy:15s} | duration={r.duration:.4f}s"
        )


if __name__ == "__main__":
    main()
