import os
import sys
import random
from typing import Any, Dict, Optional, Sequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.state import (
    CuriosityState,
    EmotionSnapshot,
    PerceptionSnapshot,
    PersonalityProfile,
)
from modules.brain.whole_brain import (
    CognitiveModule,
    CognitivePolicy,
    HeuristicCognitivePolicy,
    ProductionCognitivePolicy,
    ReinforcementCognitivePolicy,
    StructuredPlanner,
    default_plan_for_intention,
)
from schemas.emotion import EmotionType


def _perception(modalities=None):
    modalities = modalities or {
        "vision": {"spike_counts": [3.0, 2.0, 1.0]},
        "auditory": {"spike_counts": [1.0, 0.5]},
        "somatosensory": {"spike_counts": [0.2, 0.1]},
    }
    return PerceptionSnapshot(modalities=modalities)


def _emotion(primary: EmotionType, *, valence: float, arousal: float, dominance: float, intent_bias):
    return EmotionSnapshot(
        primary=primary,
        intensity=0.6,
        mood=0.2,
        dimensions={"valence": valence, "arousal": arousal, "dominance": dominance},
        context={},
        decay=0.1,
        intent_bias=intent_bias,
    )


def test_production_policy_prefers_approach_under_positive_context():
    policy = ProductionCognitivePolicy()
    perception = _perception()
    summary = {"vision": 0.5, "auditory": 0.3, "somatosensory": 0.2}
    emotion = _emotion(
        EmotionType.HAPPY,
        valence=0.65,
        arousal=0.45,
        dominance=0.25,
        intent_bias={"approach": 0.5, "withdraw": 0.15, "explore": 0.25, "observe": 0.1},
    )
    personality = PersonalityProfile(
        openness=0.7,
        conscientiousness=0.6,
        extraversion=0.75,
        agreeableness=0.7,
        neuroticism=0.2,
    )
    curiosity = CuriosityState(drive=0.65, novelty_preference=0.7, fatigue=0.1, last_novelty=0.55)
    decision = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        {"safety": 0.7, "social": 0.6},
        history=[{"intention": "approach", "confidence": 0.65}],
    )

    assert decision.intention == "approach"
    assert decision.confidence > 0.35
    assert decision.metadata["policy"] == "production"
    assert len(decision.plan) >= 3
    assert decision.weights["approach"] > decision.weights["withdraw"]


def test_production_policy_prioritises_withdraw_with_high_threat():
    policy = ProductionCognitivePolicy()
    perception = _perception()
    summary = {"vision": 0.4, "auditory": 0.4, "somatosensory": 0.2}
    emotion = _emotion(
        EmotionType.ANGRY,
        valence=-0.55,
        arousal=0.65,
        dominance=-0.25,
        intent_bias={"approach": 0.15, "withdraw": 0.5, "explore": 0.2, "observe": 0.15},
    )
    personality = PersonalityProfile(
        openness=0.4,
        conscientiousness=0.55,
        extraversion=0.35,
        agreeableness=0.45,
        neuroticism=0.7,
    )
    curiosity = CuriosityState(drive=0.35, novelty_preference=0.4, fatigue=0.25, last_novelty=0.4)
    decision = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        {"threat": 0.9, "safety": 0.1},
        history=[{"intention": "withdraw", "confidence": 0.6}],
    )

    assert decision.intention == "withdraw"
    assert decision.weights["withdraw"] > decision.weights["approach"]
    assert decision.metadata["policy"] == "production"
    assert decision.plan[0] in {"elevate_alert_state", "assess_risk_vectors"}


def test_cognitive_module_uses_production_policy_default():
    module = CognitiveModule()
    assert isinstance(module.policy, ProductionCognitivePolicy)

    perception = _perception()
    emotion = _emotion(
        EmotionType.HAPPY,
        valence=0.45,
        arousal=0.35,
        dominance=0.1,
        intent_bias={"approach": 0.4, "withdraw": 0.2, "explore": 0.25, "observe": 0.15},
    )
    personality = PersonalityProfile()
    curiosity = CuriosityState()
    decision = module.decide(
        perception,
        emotion,
        personality,
        curiosity,
        context={"safety": 0.5, "novelty": 0.4},
    )

    assert decision["policy_metadata"]["policy"] == "production"
    assert decision["plan"]
    assert 0.0 <= decision["confidence"] <= 1.0


def test_default_plan_for_intention_handles_threat():
    plan = default_plan_for_intention("withdraw", None, {"threat": 0.9, "safety": 0.1})
    assert plan[0] == "raise_alert"
    assert "seek_support" in plan


def test_structured_planner_produces_unique_padded_plan():
    planner = StructuredPlanner(min_steps=5)
    perception = _perception()
    summary = {"vision": 0.6, "auditory": 0.3, "somatosensory": 0.1}
    emotion = _emotion(
        EmotionType.NEUTRAL,
        valence=0.3,
        arousal=0.4,
        dominance=0.2,
        intent_bias={"observe": 0.5, "approach": 0.2, "withdraw": 0.1, "explore": 0.2},
    )
    curiosity = CuriosityState(drive=0.45, novelty_preference=0.5, fatigue=0.2, last_novelty=0.4)
    plan = planner.generate(
        "observe",
        "vision",
        {"novelty": 0.3, "safety": 0.4},
        perception,
        summary,
        emotion,
        curiosity,
        history=[{"intention": "observe", "confidence": 0.5} for _ in range(3)],
        learning_prediction={"cpu": 0.2, "memory": 0.3},
    )
    assert "archive_cognitive_trace" in plan
    assert len(plan) >= 5
    assert len(plan) == len(set(plan))


def test_structured_planner_and_policy_use_memory_context():
    planner = StructuredPlanner(min_steps=4)
    perception = _perception()
    summary = {"vision": 0.7, "auditory": 0.2, "somatosensory": 0.1}
    emotion = _emotion(
        EmotionType.NEUTRAL,
        valence=0.2,
        arousal=0.3,
        dominance=0.1,
        intent_bias={"observe": 0.5, "approach": 0.2, "withdraw": 0.15, "explore": 0.15},
    )
    curiosity = CuriosityState(drive=0.4, novelty_preference=0.4, fatigue=0.1, last_novelty=0.35)
    memory_records = [
        {
            "id": "rec-1",
            "text": "A remembered observation about the target area.",
            "score": 0.91,
            "metadata": {"source": "unit-test", "timestamp": "2024-01-01T00:00:00Z"},
        }
    ]
    memory_known_facts = [
        {
            "subject": "agent",
            "predicate": "should_review",
            "object": "retrieved memory findings",
        }
    ]
    base_context = {
        "memory_query": "test memory query",
        "memory_records": memory_records,
        "memory_known_facts": memory_known_facts,
        "memory_retrieval": {
            "query": "test memory query",
            "records": memory_records,
            "known_facts": memory_known_facts,
        },
    }

    plan = planner.generate(
        "observe",
        "vision",
        dict(base_context),
        perception,
        summary,
        emotion,
        curiosity,
    )

    assert plan[0] == "consult_retrieved_memory"
    assert "integrate_known_facts" in plan

    policy = ProductionCognitivePolicy(planner=planner)
    personality = PersonalityProfile()
    decision = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        context=dict(base_context),
    )

    assert decision.plan[0] == "consult_retrieved_memory"
    assert "integrate_known_facts" in decision.plan
    assert "memory-informed" in decision.tags
    memory_payload = decision.metadata.get("memory_retrieval")
    assert memory_payload
    assert memory_payload["query"] == "test memory query"
    assert memory_payload["records"]
    assert memory_payload["known_facts"]
    record = memory_payload["records"][0]
    assert record["id"] == "rec-1"
    assert record["metadata"]["source"] == "unit-test"


def test_structured_planner_and_policy_use_causal_context():
    planner = StructuredPlanner(min_steps=4)
    perception = _perception()
    summary = {"vision": 0.6, "auditory": 0.25, "somatosensory": 0.15}
    emotion = _emotion(
        EmotionType.NEUTRAL,
        valence=0.15,
        arousal=0.35,
        dominance=0.05,
        intent_bias={"observe": 0.45, "approach": 0.25, "withdraw": 0.15, "explore": 0.15},
    )
    curiosity = CuriosityState(drive=0.42, novelty_preference=0.45, fatigue=0.12, last_novelty=0.33)
    causal_relations = [
        {
            "cause": "power outage",
            "effect": "service disruption",
            "weight": 0.87,
            "metadata": {"source": "unit-test", "confidence": 0.76},
        }
    ]
    causal_paths = [
        {
            "cause": "power outage",
            "effect": "service disruption",
            "path": ["power outage", "network failure", "service disruption"],
        }
    ]
    base_context = {
        "causal_relations": causal_relations,
        "causal_paths": causal_paths,
        "causal_focus": "service disruption",
        "causal_query": "power outage impact",
    }

    plan = planner.generate(
        "observe",
        "vision",
        dict(base_context),
        perception,
        summary,
        emotion,
        curiosity,
    )

    assert plan[0] == "evaluate_causal_relations"
    assert "simulate_causal_outcomes" in plan
    assert "trace_causal_chain" in plan

    policy = ProductionCognitivePolicy(planner=planner)
    personality = PersonalityProfile()
    decision = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        context=dict(base_context),
    )

    assert decision.plan[0] == "evaluate_causal_relations"
    assert "simulate_causal_outcomes" in decision.plan
    assert "causal-informed" in decision.tags
    causal_payload = decision.metadata.get("causal_relations")
    assert causal_payload
    assert causal_payload[0]["cause"] == "power outage"
    assert causal_payload[0]["effect"] == "service disruption"
    assert "causal_paths" in decision.metadata
    assert decision.metadata["causal_paths"][0]["path"][0] == "power outage"


def test_reinforcement_policy_updates_q_values():
    policy = ReinforcementCognitivePolicy(
        learning_rate=0.5,
        discount=0.9,
        exploration=0.0,
        exploration_decay=1.0,
        min_exploration=0.0,
    )
    perception = _perception()
    summary = {"vision": 0.5, "auditory": 0.3, "somatosensory": 0.2}
    emotion = _emotion(
        EmotionType.HAPPY,
        valence=0.4,
        arousal=0.5,
        dominance=0.1,
        intent_bias={"approach": 0.3, "withdraw": 0.2, "explore": 0.3, "observe": 0.2},
    )
    personality = PersonalityProfile()
    curiosity = CuriosityState(drive=0.5, novelty_preference=0.4, fatigue=0.1, last_novelty=0.5)

    decision1 = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        context={},
    )
    state1 = policy.last_state
    value_before = policy.q_table[state1][decision1.intention]

    policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        context={"reward": 0.8},
    )
    value_after = policy.q_table[state1][decision1.intention]
    assert value_after > value_before



def test_default_plan_for_intention_explore_branch():
    plan = default_plan_for_intention("explore", "signal", {"novelty": 0.1, "safety": 0.9})
    assert plan[0] == "scan_environment"
    assert "focus_signal" in plan
    assert "expand_search_radius" in plan


def test_heuristic_policy_generates_high_confidence_tags():
    policy = HeuristicCognitivePolicy()
    perception = _perception()
    summary = {"vision": 0.6, "auditory": 0.25, "somatosensory": 0.15}
    emotion = _emotion(
        EmotionType.HAPPY,
        valence=0.85,
        arousal=0.6,
        dominance=0.4,
        intent_bias={"approach": 0.6, "withdraw": 0.1, "explore": 0.2, "observe": 0.1},
    )
    personality = PersonalityProfile(extraversion=0.8, neuroticism=0.1)
    curiosity = CuriosityState(drive=0.8, novelty_preference=0.7, fatigue=0.1, last_novelty=0.7)
    decision = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        {"safety": 0.8, "novelty": 0.7},
        history=[{"intention": "approach", "confidence": 0.7} for _ in range(2)],
    )
    assert decision.metadata["policy"] == "heuristic"
    assert "novelty-driven" in decision.tags
    assert decision.plan


def test_reinforcement_policy_marks_exploration():
    random.seed(0)
    policy = ReinforcementCognitivePolicy(
        exploration=1.0,
        exploration_decay=1.0,
        min_exploration=0.5,
    )
    perception = _perception()
    summary = {"vision": 0.5, "auditory": 0.3, "somatosensory": 0.2}
    emotion = _emotion(
        EmotionType.HAPPY,
        valence=0.2,
        arousal=0.4,
        dominance=0.1,
        intent_bias={"approach": 0.25, "withdraw": 0.25, "explore": 0.25, "observe": 0.25},
    )
    personality = PersonalityProfile()
    curiosity = CuriosityState(drive=0.4, novelty_preference=0.4, fatigue=0.2, last_novelty=0.3)
    decision = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        context={},
        learning_prediction={"cpu": 0.7, "memory": 0.5},
        history=[{"intention": "observe"}],
    )
    assert "exploring" in decision.tags
    assert any(trace.startswith("cpu=") for trace in decision.thought_trace)


def test_production_policy_fallback_on_invalid_weights():
    policy = ProductionCognitivePolicy(weight_matrix=[[0.1, 0.2], [0.3, 0.4]])
    perception = _perception()
    summary = {"vision": 0.5, "auditory": 0.3, "somatosensory": 0.2}
    emotion = _emotion(
        EmotionType.SAD,
        valence=-0.3,
        arousal=0.5,
        dominance=-0.1,
        intent_bias={"approach": 0.2, "withdraw": 0.4, "explore": 0.2, "observe": 0.2},
    )
    personality = PersonalityProfile()
    curiosity = CuriosityState()
    decision = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        context={"threat": 0.6},
    )
    assert decision.metadata.get("policy_error") is not None
    assert "policy_error" in decision.metadata


def test_cognitive_module_fallback_when_policy_raises():
    class ExplodingPolicy(CognitivePolicy):
        def select_intention(
            self,
            perception,
            summary,
            emotion,
            personality,
            curiosity,
            context,
            learning_prediction=None,
            history=None,
        ):
            raise RuntimeError("boom")
    module = CognitiveModule(policy=ExplodingPolicy())
    perception = _perception()
    emotion = _emotion(
        EmotionType.NEUTRAL,
        valence=0.0,
        arousal=0.3,
        dominance=0.0,
        intent_bias={"observe": 0.5, "approach": 0.2, "withdraw": 0.2, "explore": 0.1},
    )
    personality = PersonalityProfile()
    curiosity = CuriosityState()
    decision = module.decide(
        perception,
        emotion,
        personality,
        curiosity,
        context={"fallback_intention": "observe"},
    )
    assert decision["policy_metadata"]["policy"] == "fallback"
    assert decision["plan"]
    assert module.recall(limit=1)



def test_reinforcement_policy_adapts_exploration_downward():
    policy = ReinforcementCognitivePolicy(
        exploration=0.4,
        exploration_decay=1.0,
        min_exploration=0.05,
        exploration_smoothing=0.2,
    )
    perception = _perception()
    summary = {"vision": 0.5, "auditory": 0.3, "somatosensory": 0.2}
    emotion = _emotion(
        EmotionType.NEUTRAL,
        valence=0.1,
        arousal=0.4,
        dominance=0.1,
        intent_bias={"approach": 0.3, "withdraw": 0.2, "explore": 0.3, "observe": 0.2},
    )
    personality = PersonalityProfile()
    curiosity = CuriosityState(drive=0.4, novelty_preference=0.4, fatigue=0.1, last_novelty=0.3)

    baseline = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        context={"reward": 0.0},
    )
    positive = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        context={"reward": 0.9},
    )
    assert positive.metadata["adaptive_exploration"] < baseline.metadata["adaptive_exploration"]


def test_reinforcement_policy_adapts_exploration_upward():
    policy = ReinforcementCognitivePolicy(
        exploration=0.3,
        exploration_decay=1.0,
        min_exploration=0.05,
        exploration_smoothing=0.2,
    )
    perception = _perception()
    summary = {"vision": 0.5, "auditory": 0.3, "somatosensory": 0.2}
    emotion = _emotion(
        EmotionType.NEUTRAL,
        valence=0.0,
        arousal=0.3,
        dominance=0.0,
        intent_bias={"approach": 0.25, "withdraw": 0.25, "explore": 0.25, "observe": 0.25},
    )
    personality = PersonalityProfile()
    curiosity = CuriosityState(drive=0.3, novelty_preference=0.3, fatigue=0.2, last_novelty=0.2)

    baseline = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        context={"reward": 0.0},
    )
    negative = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        context={"reward": -0.8},
    )
    assert negative.metadata["adaptive_exploration"] > baseline.metadata["adaptive_exploration"]
