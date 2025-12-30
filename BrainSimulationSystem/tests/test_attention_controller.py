"""Tests for the goal-driven attention controller."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.models.attention_controller import GoalDrivenAttentionController


def test_attention_controller_generates_focus_and_weights():
    controller = GoalDrivenAttentionController(
        {
            "vision_weight": 0.4,
            "language_weight": 0.6,
            "max_focus_terms": 4,
        }
    )
    directives = controller.compute(
        goals=["inspect visual layout"],
        planner={"candidates": [{"action": "scan_room", "justification": ["check_light"]}]},
        dialogue_state={"topics": ["safety"], "entities": ["Lab-42"]},
        working_memory={"key_terms": ["status"], "pending_actions": ["report_anomaly"]},
        motivation={"inspect visual layout": 0.9},
    )

    assert directives["semantic_focus"], "semantic focus terms should not be empty"
    assert directives["workspace_attention"]["goal"] == "inspect visual layout"
    weights = directives["modality_weights"]
    assert weights["vision"] > 0.4  # goal biases vision
    assert "workspace_focus" in directives
