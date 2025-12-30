"""Unit tests for the prototype SelfAwarenessFramework metacognition hook."""

from __future__ import annotations

from BrainSimulationSystem.core.cognition.self_awareness import SelfAwarenessFramework
from pathlib import Path


def test_metacognition_detects_repeated_failed_actions() -> None:
    framework = SelfAwarenessFramework()

    snapshot = {}
    for _ in range(3):
        snapshot = framework.observe_reasoning_step(
            thoughts={"text": "Try searching again."},
            action={"name": "web_search", "args": {"query": "test"}},
            observation={"error": "rate_limited"},
            success=False,
            error="rate_limited",
        )

    kinds = [issue.get("kind") for issue in snapshot.get("issues", [])]
    assert "repeated_action_failure" in kinds


def test_metacognition_detects_plan_contradiction() -> None:
    framework = SelfAwarenessFramework()

    snapshot = framework.observe_reasoning_step(
        thoughts={"text": "Plan the next steps.", "plan": ["Download the file", "Do not download the file"]},
        action="noop",
        observation="planned",
        success=True,
    )

    kinds = [issue.get("kind") for issue in snapshot.get("issues", [])]
    assert "plan_contradiction" in kinds


def test_self_awareness_records_metacognition_episode() -> None:
    framework = SelfAwarenessFramework()

    framework.observe_reasoning_step(
        thoughts="Attempt 1",
        action="web_search",
        observation="no results",
        success=False,
        error="empty",
    )

    assert any(ep.get("type") == "metacognition" for ep in framework.memory.episodes)


def test_metacognition_emits_self_warning_on_high_uncertainty() -> None:
    framework = SelfAwarenessFramework()

    snapshot = framework.observe_reasoning_step(
        thoughts={"text": "I'm not sure about the answer."},
        action="respond",
        observation={"result": "guess"},
        success=True,
        confidence=0.05,
    )

    assert snapshot.get("self_warning") == "我不确定这样做是否正确"
    suggestion_kinds = [s.get("kind") for s in snapshot.get("suggestions", [])]
    assert "agent.self_warning" in suggestion_kinds


def test_metacognition_detects_knowledge_gap() -> None:
    framework = SelfAwarenessFramework()

    snapshot = {}
    for _ in range(2):
        snapshot = framework.observe_reasoning_step(
            thoughts={"text": "Need supporting info."},
            action="answer",
            observation={"knowledge_hits": 0},
            success=True,
            confidence=0.6,
            metadata={"knowledge_hit": False, "query": "example query"},
        )

    kinds = [issue.get("kind") for issue in snapshot.get("issues", [])]
    assert "knowledge_gap" in kinds
    assert snapshot.get("self_warning") == "我目前缺乏这方面知识"
    suggestion_kinds = [s.get("kind") for s in snapshot.get("suggestions", [])]
    assert "agent.self_warning" in suggestion_kinds


def test_metacognition_suggests_rest_on_high_fatigue() -> None:
    framework = SelfAwarenessFramework()

    snapshot = {}
    for _ in range(3):
        snapshot = framework.observe_reasoning_step(
            thoughts="keep going",
            action="noop",
            observation={},
            success=True,
            confidence=0.9,
            metadata={"fatigue": 0.95},
        )

    kinds = [issue.get("kind") for issue in snapshot.get("issues", [])]
    assert "fatigue_high" in kinds
    suggestion_kinds = [s.get("kind") for s in snapshot.get("suggestions", [])]
    assert "agent.rest" in suggestion_kinds


def test_self_awareness_stores_and_retrieves_self_reflection_notes() -> None:
    framework = SelfAwarenessFramework()

    def llm_stub(prompt: str) -> str:
        assert "请回顾刚才的任务执行过程" in prompt
        return (
            '{'
            '"reflection":"主要问题是缺少输入有效性检查。",'
            '"lessons":["对关键输入先做校验","失败先做最小复现"],'
            '"checklist":["检查输入是否为空/格式是否正确"],'
            '"tags":["data_validation"]'
            '}'
        )

    record = framework.reflect_after_task(
        task="处理一批传感器数据并生成报告",
        outcome="失败：输入数据为空导致下游崩溃",
        llm=llm_stub,
    )

    assert record["type"] == "self_reflection"
    assert "输入有效性检查" in record["reflection"]

    hints = framework.retrieve_reflection_hints("传感器数据 输入 校验", top_k=3, min_similarity=0.05)
    assert hints
    assert hints[0].get("type") == "self_reflection"


def test_self_concept_defaults_and_updates_with_experience() -> None:
    framework = SelfAwarenessFramework()

    concept = framework.memory.self_concept
    for key in ("agency", "identity", "competence_confidence", "curiosity", "frustration", "achievement"):
        assert key in concept

    baseline_competence = float(concept["competence_confidence"])
    baseline_curiosity = float(concept["curiosity"])
    baseline_frustration = float(concept["frustration"])
    baseline_achievement = float(concept["achievement"])

    for _ in range(5):
        framework.observe_reasoning_step(
            thoughts="try",
            action="noop",
            observation={},
            success=True,
            confidence=0.8,
            metadata={"curiosity_drive": 0.9},
        )

    after_success = dict(framework.memory.self_concept)
    assert after_success["competence_confidence"] > baseline_competence
    assert after_success["curiosity"] > baseline_curiosity
    assert after_success["frustration"] < baseline_frustration
    assert after_success["achievement"] > baseline_achievement

    for _ in range(5):
        framework.observe_reasoning_step(
            thoughts="try",
            action="noop",
            observation={},
            success=False,
            confidence=0.2,
            metadata={"curiosity_drive": 0.1},
        )

    after_failure = dict(framework.memory.self_concept)
    assert after_failure["competence_confidence"] < after_success["competence_confidence"]
    assert after_failure["frustration"] > after_success["frustration"]


def test_self_concept_is_recorded_in_autobiographical_snapshots() -> None:
    framework = SelfAwarenessFramework()
    framework.update_self_state(sensor_data={"arm": 1.0})
    episode = framework.memory.episodes[-1]
    assert "self_concept" in episode
    assert "competence_confidence" in episode["self_concept"]


def test_task_outcome_updates_and_records_self_concept() -> None:
    framework = SelfAwarenessFramework()
    baseline = float(framework.memory.self_concept["competence_confidence"])

    record = framework.observe_task_outcome(
        task="demo_task",
        success=True,
        confidence=0.9,
        curiosity_drive=0.8,
    )

    assert record["type"] == "task_outcome"
    assert record["self_concept"]["competence_confidence"] >= baseline
    assert any(ep.get("type") == "task_outcome" for ep in framework.memory.episodes)


def test_daily_self_check_summarizes_and_exports(tmp_path: Path) -> None:
    framework = SelfAwarenessFramework()
    now = 1_700_000_000.0

    framework.observe_task_outcome(task="t1", success=True, confidence=0.8, curiosity_drive=0.6, timestamp=now - 3600)
    framework.observe_task_outcome(task="t2", success=False, confidence=0.3, curiosity_drive=0.2, timestamp=now - 1800)

    framework.apply_human_feedback(
        rating=0.6,
        notes="整体尚可，但对输入校验需要更严格。",
        calibration={"competence_confidence": 0.55},
        learning_rate=1.0,
        timestamp=now - 1200,
    )

    out = tmp_path / "daily_review.json"
    report = framework.daily_self_check(now=now, export_path=out)
    assert report["type"] == "daily_self_check"
    assert report["task_stats"]["total"] == 2
    assert report["task_stats"]["success"] == 1
    assert report["task_stats"]["failure"] == 1
    assert out.exists()


def test_human_feedback_calibrates_self_concept() -> None:
    framework = SelfAwarenessFramework()
    before = float(framework.memory.self_concept["competence_confidence"])

    record = framework.apply_human_feedback(
        rating=0.2,
        notes="你对自己的把握偏高，请更保守一些。",
        calibration={"competence_confidence": 0.3},
        learning_rate=1.0,
    )

    assert record["type"] == "human_feedback"
    assert float(framework.memory.self_concept["competence_confidence"]) != before
    assert abs(float(framework.memory.self_concept["competence_confidence"]) - 0.3) < 1e-6
