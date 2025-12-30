import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evolution.genesis_team.conflict import (
    ConflictResolver,
    StructuredDataConflictStrategy,
)


def test_overlapping_edits_detection():
    """Conflicts arise when agents edit the same file differently."""

    logs = {
        "agent1": json.dumps({"file": "module.py", "version": 1, "content": "a = 1"}),
        "agent2": json.dumps({"file": "module.py", "version": 1, "content": "a = 2"}),
    }
    resolver = ConflictResolver(strategy=StructuredDataConflictStrategy())
    assert resolver.detect("agent2", logs)
    decision = resolver.resolve("agent2", logs)
    assert "rollback" in decision


def test_version_mismatch_detection():
    """Conflicts arise when agents disagree on version numbers."""

    logs = {
        "agent1": json.dumps({"file": "module.py", "version": 1, "content": "a = 1"}),
        "agent2": json.dumps({"file": "module.py", "version": 2, "content": "a = 1"}),
    }
    resolver = ConflictResolver(strategy=StructuredDataConflictStrategy())
    assert resolver.detect("agent2", logs)
    decision = resolver.resolve("agent2", logs)
    assert "rollback" in decision

