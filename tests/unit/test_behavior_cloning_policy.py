from __future__ import annotations

from modules.learning.behavior_cloning import BehaviorCloningConfig, BehaviorCloningPolicy


def test_behavior_cloning_policy_learns_state_action_mapping() -> None:
    policy = BehaviorCloningPolicy(
        BehaviorCloningConfig(
            state_dim=2,
            lr=0.2,
            weight_decay=0.0,
            label_smoothing=0.0,
            entropy_bonus=0.0,
            inference_uniform_mix=0.0,
            seed=123,
        )
    )

    state_a = {"fused_embedding": [1.0, 0.0]}
    state_b = {"fused_embedding": [0.0, 1.0]}

    for _ in range(40):
        policy.observe(state_a, "action_a")
        policy.observe(state_b, "action_b")

    assert policy.suggest_actions(state_a, top_k=1) == ["action_a"]
    assert policy.suggest_actions(state_b, top_k=1) == ["action_b"]

