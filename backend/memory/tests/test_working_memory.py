import sys
from pathlib import Path

# Ensure repository root on path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

from backend.memory.working_memory import WorkingMemory


def test_search_returns_matches_in_reverse_recency():
    memory = WorkingMemory(capacity=5)
    memory.store("alpha")
    memory.store("beta")
    memory.store("alphabet")

    results = list(memory.search("alp"))

    assert results == ["alphabet", "alpha"]


def test_search_handles_no_matches():
    memory = WorkingMemory(capacity=3)
    memory.store("cat")
    memory.store("dog")

    results = list(memory.search("bird"))

    assert results == []


def test_search_respects_limit():
    memory = WorkingMemory(capacity=4)
    memory.store("one")
    memory.store("two")
    memory.store("tone")
    memory.store("stone")

    results = list(memory.search("one", limit=2))

    assert results == ["stone", "tone"]


def test_search_normalizes_non_string_items():
    memory = WorkingMemory(capacity=4)
    memory.store(123)
    memory.store({"number": 123})
    memory.store("abc123")

    results = list(memory.search("123"))

    assert results == ["abc123", {"number": 123}, 123]
