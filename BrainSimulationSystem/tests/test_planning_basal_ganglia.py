"""Tests for basal ganglia action selection + plan-following decision modes."""

from __future__ import annotations

from ..models.decision import DecisionProcess
from ..models.hybrid_planner import HybridPlanner
from ..models.knowledge_graph import KnowledgeGraph
from ..models.symbolic_reasoner import SymbolicReasoner


def test_decision_process_planner_prefers_plan_next_action():
    process = DecisionProcess(None, {"decision_type": "planner"})
    result = process.process({"options": ["a", "b"], "context": {"plan_next_action": "b"}})
    assert result["decision"] == "b"
    assert result["confidence"] >= 0.75


def test_decision_process_basal_ganglia_picks_highest_value():
    process = DecisionProcess(None, {"decision_type": "basal_ganglia", "basal_ganglia": {"deterministic": True}})
    context = {}
    context_key = str(hash(str(context)))
    process.action_values[(context_key, "b")] = 1.0
    result = process.process({"options": ["a", "b", "c"], "context": context})
    assert result["decision"] == "b"


def test_hybrid_planner_builds_sequence_from_knowledge_graph():
    graph = KnowledgeGraph()
    graph.add("deploy_model", "requires", "train_model")
    graph.add("train_model", "subtask_of", "prepare_dataset")
    graph.add("prepare_dataset", "requires", "collect_data")
    graph.add("collect_data", "precedes", "clean_data")

    reasoner = SymbolicReasoner(graph)
    planner = HybridPlanner(
        graph,
        reasoner,
        {"sequence_enabled": True, "sequence_max_depth": 4, "sequence_max_actions": 16},
    )
    plan = planner.generate_plan({}, goals=["deploy_model"], options=[])
    sequence = plan.get("sequence") or []

    assert "collect_data" in sequence
    assert "deploy_model" in sequence
    assert sequence.index("collect_data") < sequence.index("deploy_model")

