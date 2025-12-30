from __future__ import annotations

import time
from pathlib import Path

from modules.learning.continual_learning import ContinualLearningCoordinator, LearningLoopConfig
from modules.learning.experience_hub import EpisodeRecord, ExperienceHub
from modules.monitoring.collector import MetricEvent, RealTimeMetricsCollector
from modules.monitoring.performance_diagnoser import PerformanceDiagnoser
from modules.evolution.strategy_adjuster import StrategyAdjuster


def test_continual_learning_reflection_payload_enriched(tmp_path: Path) -> None:
    hub = ExperienceHub(tmp_path / "experience")
    hub.append(
        EpisodeRecord(
            task_id="t",
            policy_version="v",
            total_reward=0.0,
            steps=1,
            success=False,
            metadata={"source": "test"},
        )
    )

    collector = RealTimeMetricsCollector(monitor=None)
    # Inject a deterministic event so diagnoser can flag an issue.
    collector._events.append(
        MetricEvent(
            module="m",
            latency=2.0,
            energy=0.0,
            throughput=0.1,
            timestamp=time.time(),
            status="failure",
        )
    )

    diagnoser = PerformanceDiagnoser(max_latency_s=0.5, min_success_rate=1.0, min_throughput=1.0)
    adjuster = StrategyAdjuster()

    seen: list[dict] = []

    def _cb(payload: dict) -> None:
        seen.append(payload)

    cfg = LearningLoopConfig(reflection_interval=0.0, background_interval=9999.0)
    coordinator = ContinualLearningCoordinator(
        experience_hub=hub,
        performance_diagnoser=diagnoser,
        strategy_adjuster=adjuster,
        reflection_callback=_cb,
        config=cfg,
    )
    coordinator.set_collector(collector)

    coordinator.run_maintenance_cycle(force=True)

    assert seen
    payload = seen[-1]
    assert payload.get("episodes")
    assert payload.get("metrics_summary")
    assert payload.get("issues")
    assert payload.get("strategy_suggestions")


def test_register_human_feedback_emits_metric_event(tmp_path: Path) -> None:
    hub = ExperienceHub(tmp_path / "experience")
    collector = RealTimeMetricsCollector(monitor=None)
    coordinator = ContinualLearningCoordinator(experience_hub=hub, config=LearningLoopConfig(background_interval=9999.0))
    coordinator.set_collector(collector)

    coordinator.register_human_feedback(
        task_id="t1",
        prompt="p",
        agent_response="a",
        correct_response="c",
        rating=4.0,
        metadata={"source": "test"},
    )

    events = collector.events()
    assert any(event.module == "human_feedback" for event in events)
    feedback = next(event for event in events if event.module == "human_feedback")
    assert feedback.stage == "t1"
    assert feedback.metadata.get("rating") == 4.0
    # rating=4 on a 0-5 scale -> ~0.8 confidence
    assert feedback.confidence is not None
    assert feedback.confidence > 0.7
