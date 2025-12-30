"""Tests for the neuro-symbolic reasoning enhancements."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
backend_autogpt = os.path.join(ROOT, "backend", "autogpt")
if backend_autogpt not in sys.path:
    sys.path.append(backend_autogpt)

from backend.reasoning.logic import LogicProgram, LogicRule
from backend.reasoning.symbolic import SymbolicReasoner
from backend.reasoning.causal import KnowledgeGraphCausalReasoner
from backend.reasoning.commonsense import CommonsenseKnowledge, CommonsenseValidator
from backend.reasoning.workspace import (
    LogicConstraintObserver,
    CausalObserver,
    CommonsenseObserver,
)
from backend.autogpt.autogpt.core.global_workspace import GlobalWorkspace


def test_logic_program_forward_chaining() -> None:
    program = LogicProgram()
    program.add_fact("rain")
    program.add_rule(LogicRule(head="wet", body=("rain",), description="rain implies wet"))
    holds, proof = program.query("wet")
    assert holds
    assert any("rain" in step for step in proof)


def test_symbolic_reasoner_prove_with_constraints() -> None:
    reasoner = SymbolicReasoner({})
    rule = LogicRule(head="goal", body=("premise",), description="premise => goal")
    holds, proof = reasoner.prove("goal", premises=["premise"], rules=[rule])
    assert holds
    assert proof
    satisfied, missing = reasoner.validate_constraints(["goal"], premises=["premise"], rules=[rule])
    assert satisfied
    assert not missing


def test_causal_reasoner_predicts_paths() -> None:
    graph = {"earthquake": ["power_outage"], "power_outage": ["service_disruption"]}
    reasoner = KnowledgeGraphCausalReasoner(graph)
    exists, path = reasoner.check_causality("earthquake", "service_disruption")
    assert exists
    assert path == ["earthquake", "power_outage", "service_disruption"]
    effects = reasoner.predict_effects("earthquake", depth=2)
    assert ("power_outage", 1.0) in effects
    intervention = reasoner.intervention("earthquake", "service_disruption")
    assert "reducing" in intervention


def test_commonsense_validator_detects_contradiction() -> None:
    knowledge = CommonsenseKnowledge()
    knowledge.add_fact("cat", "can", "purr", truth=True)
    knowledge.add_fact("cat", "can", "bark", truth=False)
    validator = CommonsenseValidator(knowledge)

    consistent = validator.validate("cat", "can", "purr")
    assert consistent.status == "consistent"

    contradiction = validator.validate("cat", "can", "bark")
    assert contradiction.status == "contradiction"
    assert contradiction.suggestions


def test_workspace_observers_surface_feedback() -> None:
    graph = {"policy": ["effect"], "effect": ["outcome"]}
    reasoner = SymbolicReasoner(graph)
    reasoner.register_fact("premise")
    rule = LogicRule(head="goal", body=("premise",))
    validator = CommonsenseValidator(CommonsenseKnowledge())
    workspace = GlobalWorkspace()
    workspace.register_observer(LogicConstraintObserver(reasoner))
    workspace.register_observer(CausalObserver(reasoner.causal))
    workspace.register_observer(CommonsenseObserver(validator))

    logic_feedback = workspace.broadcast(
        {
            "next_ability": "symbolic_reason",
            "ability_arguments": {
                "goal": "goal",
                "premises": ["premise"],
                "rules": [{"head": "goal", "body": ["premise"]}],
            },
        },
        context={"task_objective": "prove constraint"},
    )
    assert any(signal.source == "logic" for signal in logic_feedback)

    causal_feedback = workspace.broadcast(
        {
            "next_ability": "causal_reason",
            "ability_arguments": {"cause": "policy", "effect": "outcome"},
        },
        context={"task_objective": "evaluate causal impact"},
    )
    assert any(signal.source == "causal" for signal in causal_feedback)

    commonsense_feedback = workspace.broadcast(
        {
            "next_ability": "commonsense_validate",
            "ability_arguments": {
                "subject": "cat",
                "relation": "can",
                "object": "purr",
            },
        },
        context={"task_objective": "check commonsense"},
    )
    assert any(signal.source == "commonsense" for signal in commonsense_feedback)
    assert workspace.feedback_history()
