from __future__ import annotations

import logging
from pathlib import Path

from autogpt.core.configuration.learning import LearningConfiguration
from autogpt.core.learning.experience_store import ExperienceLogStore
from autogpt.core.self_improvement.auto_tuner import SelfImprovementEngine


def _make_learning_config(tmp_path: Path) -> LearningConfiguration:
    base = {
        "log_path": str(tmp_path / "log.jsonl"),
        "improvement_state_path": str(tmp_path / "state.json"),
        "plan_output_path": str(tmp_path / "plan.json"),
        "baseline_success_path": str(tmp_path / "baseline.json"),
        "validation_reports_dir": str(tmp_path / "validation"),
        "replay_reports_dir": str(tmp_path / "replay_reports"),
        "replay_scenarios_dir": str(tmp_path / "replay_scenarios"),
        "prompt_candidates_dir": str(tmp_path / "prompts"),
        "ability_history_path": str(tmp_path / "ability.json"),
        "ability_history_limit": 5,
        "ability_low_score_streak": 2,
    }
    return LearningConfiguration(**base)


def test_build_plan_includes_targeted_actions(tmp_path):
    config = _make_learning_config(tmp_path)
    store = ExperienceLogStore(Path(config.log_path))
    engine = SelfImprovementEngine(config=config, store=store, logger=logging.getLogger("test"))

    metrics = {"overall_success": 0.4, "command_stats": {}, "total": 0}
    ability_report = {
        "scores": {"creativity": 0.3},
        "weak_abilities": [
            {"name": "creativity", "streak": 2, "latest_score": 0.3},
        ],
        "history": [],
    }

    plan = engine._build_plan(metrics, [], ability_report)
    assert plan is not None
    assert plan["ability_scores"]["creativity"] == 0.3
    assert any(action["ability"] == "creativity" for action in plan["targeted_actions"])
