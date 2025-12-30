import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.getcwd()))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "modules")))

from evolution.policy_trainer import PolicyTrainer


def _custom_map(obj: dict) -> tuple[str, str, float]:
    exp = obj.get("experience", obj)
    return exp.get("s", exp.get("state")), exp.get("a", exp.get("action")), float(
        exp.get("r", exp.get("reward"))
    )


@pytest.mark.parametrize(
    "ext,use_custom",
    [(".csv", False), (".json", True), (".jsonl", True)],
)
def test_policy_trainer_updates_from_rewards(tmp_path: Path, ext: str, use_custom: bool) -> None:
    dataset = tmp_path / f"dataset{ext}"
    if ext == ".csv":
        dataset.write_text("state,action,reward\nS,A,1\nS,B,-1\n")
    elif ext == ".json":
        data = [
            {"experience": {"s": "S", "a": "A", "r": 1}},
            {"experience": {"s": "S", "a": "B", "r": -1}},
        ]
        dataset.write_text(json.dumps(data))
    else:  # jsonl
        lines = [
            json.dumps({"experience": {"s": "S", "a": "A", "r": 1}}),
            json.dumps({"experience": {"s": "S", "a": "B", "r": -1}}),
        ]
        dataset.write_text("\n".join(lines))

    kwargs = {"dataset_path": dataset, "learning_rate": 0.1}
    if use_custom:
        kwargs["experience_transform"] = _custom_map
    trainer = PolicyTrainer(**kwargs)
    updated = trainer.update_policy()

    assert "S" in updated
    assert updated["S"]["A"] > updated["S"]["B"]

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(updated))
    loaded = json.loads(policy_path.read_text())
    assert loaded["S"]["A"] == updated["S"]["A"]


def test_multi_agent_training() -> None:
    trainer = PolicyTrainer(dataset_path=Path("nonexistent"), num_agents=2)
    trainer.push_experience("S", "A", 1.0, agent_id=0)
    trainer.push_experience("S", "B", -1.0, agent_id=0)
    trainer.push_experience("S", "A", 1.0, agent_id=1)
    trainer.push_experience("S", "B", -1.0, agent_id=1)
    policies = trainer.update_policy()
    assert isinstance(policies, dict)
    assert policies[0]["S"]["A"] > policies[0]["S"]["B"]
    assert policies[1]["S"]["B"] < policies[1]["S"]["A"]
