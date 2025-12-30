from __future__ import annotations

import json
from pathlib import Path

from autogpt.core.self_improvement.ability_tracker import AbilityScoreTracker


def test_ability_tracker_records_history_and_flags_weak_spots(tmp_path):
    history_path = Path(tmp_path) / "ability_history.json"
    tracker = AbilityScoreTracker(
        history_path=history_path,
        low_score_threshold=0.7,
        streak_length=2,
        history_limit=5,
        latency_normaliser=1.0,
    )

    evaluation_summary = {"precision": 0.55, "recall": 0.5, "latency": 0.2, "fairness": 0.1}
    benchmark_payload = {
        "tests": {
            "benchmarks/test_creativity_benchmark.py::test_creativity_feedback": {
                "outcome": "failed"
            },
            "benchmarks/tests/test_reasoning.py::test_symbolic_reasoning": {
                "outcome": "failed"
            },
        }
    }

    first_report = tracker.update(evaluation_summary, benchmark_payload)
    assert first_report is not None
    assert "logic_reasoning" in first_report["scores"]
    assert not first_report["weak_abilities"]

    # Second low-scoring update should trigger a weak ability streak.
    second_summary = {"precision": 0.45, "recall": 0.4, "latency": 0.3, "fairness": 0.15}
    second_report = tracker.update(second_summary, benchmark_payload)

    assert second_report is not None
    weak_abilities = {entry["name"]: entry for entry in second_report["weak_abilities"]}
    assert "creativity" in weak_abilities
    assert weak_abilities["creativity"]["streak"] >= 2

    # Ensure the persisted history matches the expected length and structure.
    stored = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(stored) == 2
    assert stored[-1]["scores"]["creativity"] <= 0.5

