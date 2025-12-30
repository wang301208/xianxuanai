from __future__ import annotations

from pathlib import Path

from modules.learning.meta_retrieval_policy import MetaRetrievalPolicy, MetaRetrievalPolicyConfig


def test_meta_retrieval_policy_suggests_web_search_for_algorithm_tasks(tmp_path: Path) -> None:
    policy = MetaRetrievalPolicy(
        MetaRetrievalPolicyConfig(state_path=tmp_path / "state.json", save_on_update=False, top_channels=2)
    )
    suggestion = policy.suggest(
        task_text="Explain Monte Carlo Tree Search algorithm and provide a Python implementation.",
        enabled_actions={"code_index_search": True, "web_search": True, "documentation_tool": True},
        has_local_roots=True,
    )

    assert suggestion["domain"] == "algorithm"
    assert "web_search" in suggestion["channels"]
    assert suggestion["config_patch"].get("web_search") is True


def test_meta_retrieval_policy_learns_channel_preference_and_persists(tmp_path: Path) -> None:
    state = tmp_path / "meta_policy.json"
    policy = MetaRetrievalPolicy(MetaRetrievalPolicyConfig(state_path=state, save_on_update=False, top_channels=2))

    for _ in range(6):
        policy.observe(domain="api", channels=["documentation_tool"], success=False)
    for _ in range(6):
        policy.observe(domain="api", channels=["web_search"], success=True)
    policy.save()

    reloaded = MetaRetrievalPolicy(MetaRetrievalPolicyConfig(state_path=state, save_on_update=False, top_channels=2))
    suggestion = reloaded.suggest(
        task_text="How to use the OpenAI API client? What parameters does chat.completions accept?",
        enabled_actions={"code_index_search": True, "web_search": True, "documentation_tool": True},
        has_local_roots=True,
    )

    assert suggestion["domain"] == "api"
    assert suggestion["channels"] == ["code_index", "web_search"]


def test_meta_retrieval_policy_supports_rated_reward_updates(tmp_path: Path) -> None:
    state = tmp_path / "meta_policy_reward.json"
    policy = MetaRetrievalPolicy(MetaRetrievalPolicyConfig(state_path=state, save_on_update=False, top_channels=2))

    # Prefer channels that correlate with higher human feedback scores.
    for _ in range(6):
        policy.observe(domain="api", channels=["documentation_tool"], reward=2.0)  # 2/5
    for _ in range(6):
        policy.observe(domain="api", channels=["web_search"], reward=5.0)  # 5/5
    policy.save()

    reloaded = MetaRetrievalPolicy(MetaRetrievalPolicyConfig(state_path=state, save_on_update=False, top_channels=2))
    suggestion = reloaded.suggest(
        task_text="How to use the OpenAI API client? What parameters does chat.completions accept?",
        enabled_actions={"code_index_search": True, "web_search": True, "documentation_tool": True},
        has_local_roots=True,
    )

    assert suggestion["domain"] == "api"
    assert suggestion["channels"] == ["code_index", "web_search"]
