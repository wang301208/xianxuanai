"""Regression tests for dialogue state tracking and semantic fallback parser."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.models.attention_diffuser import AttentionDiffuser
from BrainSimulationSystem.models.dialogue_state import DialogueStateTracker
from BrainSimulationSystem.models.language_processing import SemanticNetwork
from BrainSimulationSystem.models.semantic_parser import SemanticFallbackParser


def test_dialogue_state_tracker_accumulates_turns():
    tracker = DialogueStateTracker({"max_turns": 3})
    comprehension = {
        "intent": "question",
        "key_terms": ["report", "timeline"],
        "entities": ["Project Alpha"],
        "summary": "When will Project Alpha finish?",
        "action_items": ["gather_timeline"],
    }

    snapshot = tracker.update(comprehension, speaker="user")

    assert snapshot["turns"][-1]["intent"] == "question"
    assert "Project Alpha" in snapshot["entities"]


def test_semantic_fallback_parser_returns_heuristic_payload():
    parser = SemanticFallbackParser(llm_service=None, config={"enabled": True})
    payload = parser.parse(
        "We postponed the launch window because the platform needed repairs.",
        context={"dialogue_state": {"last_intent": "statement"}},
    )

    assert payload is not None
    assert payload["summary"]
    assert payload["key_terms"]
    assert payload["source"] == "heuristic"


def test_attention_diffuser_applies_top_down_directives():
    diffuser = AttentionDiffuser()
    network = SemanticNetwork({})
    memory_state = {"key_terms": []}

    directives = {
        "semantic_focus": ["alpha", "beta"],
        "goal_snapshot": ["inspect system"],
    }
    diffuser.apply(network, memory_state, ["status"], directives=directives)

    assert "alpha" in network.nodes
    assert "inspect system" in network.nodes
