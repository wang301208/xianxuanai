import os
import sys

# Ensure repository root on path for direct module imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from backend.memory.long_term import LongTermMemory
from backend.reflection import (
    ReflectionModule,
    ReflectionResult,
    load_histories,
    save_history,
)


def test_history_round_trip(tmp_path):
    db = tmp_path / "mem.db"
    memory = LongTermMemory(db)

    evaluations = [
        ReflectionResult(0.1, "negative", raw="bad"),
        ReflectionResult(0.9, "positive", raw="good"),
    ]

    def eval_fn(text: str) -> ReflectionResult:
        return evaluations[0] if "revised" not in text else evaluations[1]

    def rewrite_fn(text: str) -> str:
        return text + " revised"

    captured: list[tuple[ReflectionResult, str]] = []

    module = ReflectionModule(
        evaluate=eval_fn,
        rewrite=rewrite_fn,
        max_passes=2,
        quality_threshold=0.5,
        callback=lambda ev, rev: captured.append((ev, rev)),
    )

    module.reflect("initial")
    assert captured == module.history

    save_history(memory, module.history)
    loaded = list(load_histories(memory))
    assert loaded and loaded[0] == module.history

    memory.close()
