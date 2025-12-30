"""Tests for the enhanced action planner."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.models.action_planner import ActionPlanner
from BrainSimulationSystem.models.knowledge_graph import KnowledgeGraph


class _StubReasoner:
    """Minimal stand-in for the GeneralReasoner API used during tests."""

    def reason_about_unknown(
        self,
        task_description: str,
        max_steps: int | None = None,
        confidence_threshold: float = 0.5,
        **_: object,
    ) -> list[dict[str, object]]:
        steps = [
            {
                "hypothesis": f"analyse_{task_description}",
                "verification": "needs context alignment",
                "confidence": 0.55,
                "method": "analysis",
                "iteration": 1,
            },
            {
                "hypothesis": f"execute_{task_description}",
                "verification": "final action",
                "confidence": 0.7,
                "method": "planner",
                "iteration": 2,
            },
        ]
        if max_steps is not None:
            return steps[:max_steps]
        return steps


def _build_knowledge_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add("deploy_model", "requires", "train_model")
    graph.add("train_model", "subtask_of", "prepare_dataset")
    graph.add("prepare_dataset", "requires", "collect_data")
    graph.add("collect_data", "precedes", "clean_data")
    return graph


def test_action_planner_generates_hierarchical_steps_from_knowledge_graph():
    planner = ActionPlanner({"goal_depth": 3, "max_subtasks": 4, "max_plan_branches": 3})
    graph = _build_knowledge_graph()

    plan = planner.plan(
        intent="command",
        suggested_actions=[],
        semantic_relations=[],
        affect_tone=[],
        context={"goals": ["deploy_model"], "knowledge_graph": graph},
    )

    assert plan.steps, "hierarchical plan should include steps derived from the knowledge graph"
    first_step = plan.steps[0]
    assert first_step["goal"] == "deploy_model"
    assert first_step["subtasks"], "each plan step should contain subtask descriptions"
    assert any("collect_data" in action for action in plan.actions), "graph-derived actions should join plan"


def test_action_planner_uses_reasoner_when_graph_missing_information():
    planner = ActionPlanner({})
    plan = planner.plan(
        intent="command",
        suggested_actions=[],
        semantic_relations=[],
        affect_tone=[],
        context={
            "goals": ["explore_target"],
            "general_reasoner": _StubReasoner(),
        },
    )

    assert plan.steps, "reasoner fallback should provide multi-step plan hints"
    assert plan.steps[0]["source"] == "general_reasoner"
    assert any("explore_target" in action for action in plan.actions)
