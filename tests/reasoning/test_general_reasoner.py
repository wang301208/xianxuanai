import importlib
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

modules_pkg = sys.modules.get("modules")
if modules_pkg is None:
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = [str(REPO_ROOT / "modules")]
    sys.modules["modules"] = modules_pkg

brain_pkg = sys.modules.get("modules.brain")
if brain_pkg is None:
    brain_pkg = types.ModuleType("modules.brain")
    brain_pkg.__path__ = [str(REPO_ROOT / "modules" / "brain")]
    sys.modules["modules.brain"] = brain_pkg

reasoning_pkg = sys.modules.get("modules.brain.reasoning")
if reasoning_pkg is None:
    reasoning_pkg = types.ModuleType("modules.brain.reasoning")
    reasoning_pkg.__path__ = [str(REPO_ROOT / "modules" / "brain" / "reasoning")]
    sys.modules["modules.brain.reasoning"] = reasoning_pkg

GeneralReasoner = importlib.import_module("modules.brain.reasoning.general_reasoner").GeneralReasoner

def _make_reasoner(tmp_path: Path) -> GeneralReasoner:
    return GeneralReasoner()


def test_reason_about_unknown_produces_hypotheses(tmp_path: Path):
    reasoner = _make_reasoner(tmp_path)
    steps = reasoner.reason_about_unknown(
        "Investigate flarblax anomalies", task_id="test-flarblax", max_steps=2
    )
    assert steps, "should return at least one step"
    for step in steps:
        assert step["hypothesis"]
        assert step["verification"]
        assert "confidence" in step
        assert "method" in step


def test_reason_about_unknown_uses_available_knowledge(tmp_path: Path):
    reasoner = _make_reasoner(tmp_path)
    reasoner.add_concept_relation("acid", "base")
    reasoner.analogical.add_knowledge(
        "default", {"subject": "acid", "object": "base"}, "acid neutralizes base"
    )
    reasoner.add_example("mixing acid with base", "produces salt")

    steps = reasoner.reason_about_unknown(
        "acid reaction with unknown", task_id="acid-reaction", max_steps=3
    )

    assert any("acid relates to base" in s["hypothesis"] for s in steps)
    assert any("analogy" in s["hypothesis"] for s in steps)
    assert any("produces salt" in s["verification"] for s in steps)


def test_reason_about_unknown_resumes_from_memory(tmp_path: Path):
    reasoner = _make_reasoner(tmp_path)
    reasoner.add_concept_relation("acid", "base")
    task_description = "acid reaction with unknown"
    task_id = "resume-acid"

    initial = reasoner.reason_about_unknown(task_description, task_id=task_id, max_steps=2)
    assert initial

    recovered = reasoner.resume_reasoning(task_id, task_description)
    assert recovered
    assert recovered[0]["hypothesis"] == initial[0]["hypothesis"]

    # ensure a follow-up call appends without losing trace
    follow_up = reasoner.reason_about_unknown(task_description, task_id=task_id, max_steps=3)
    assert len(follow_up) >= len(initial)


def test_reason_about_unknown_extends_trace_until_confident(tmp_path: Path):
    reasoner = _make_reasoner(tmp_path)
    reasoner.add_example("explore enigmatic signals", "compare spectral fingerprints")

    steps = reasoner.reason_about_unknown(
        "enigmatic signal correlation", task_id="long-trace", max_steps=6, confidence_threshold=0.9
    )

    assert len(steps) >= 4, "should accumulate more than baseline heuristics"
    assert any(step["method"] == "reflection" for step in steps), "reflection should contribute"
    assert steps[-1]["confidence"] >= 0.75, "confidence should improve across iterations"

    stored = reasoner.resume_reasoning("long-trace")
    assert len(stored) >= len(steps), "persisted trace should include extended iterations"


def test_reason_about_unknown_dynamic_limit(tmp_path: Path):
    reasoner = _make_reasoner(tmp_path)

    steps = reasoner.reason_about_unknown(
        "mysterious apparatus behaviour", task_id="dynamic-trace", max_steps=None, confidence_threshold=0.8
    )

    assert len(steps) >= 4
    assert steps[-1]["confidence"] >= 0.8
