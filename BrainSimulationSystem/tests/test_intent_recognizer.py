"""Tests for adaptive intent recognizer behaviour."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.models.intent_recognizer import IntentRecognizer
from BrainSimulationSystem.models.language_processing import SemanticNetwork, SyntaxProcessor


def _build_processors():
    network = SemanticNetwork({})
    syntax = SyntaxProcessor({})
    return network, syntax


def test_intent_recognizer_learns_terms_and_reuses_rules():
    recognizer = IntentRecognizer(
        {
            "auto_expand_terms": True,
            "auto_expand_threshold": 0.3,
            "max_learned_terms": 4,
        }
    )
    network, syntax = _build_processors()

    # Initial classification with explicit cue should learn the novel verb.
    recognizer.classify(
        "Please audit the quarterly finances",
        ["please", "audit", "the", "quarterly", "finances"],
        network,
        syntax,
    )
    assert "audit" in recognizer.learned_terms["command"]

    # Follow-up utterance relies on the learned keyword (no question mark or command trigger).
    result = recognizer.classify(
        "audit financial status",
        ["audit", "financial", "status"],
        network,
        syntax,
    )
    assert result.label == "command"
    assert result.source == "rule"
    assert result.details.get("evidence", {}).get("pattern") == "learned_keyword"
