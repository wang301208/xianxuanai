from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: str):
    module_path = ROOT / path
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Provide a lightweight package structure for backend modules
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)

backend_pkg.memory = _load("backend.memory", "backend/memory/__init__.py")
backend_pkg.reflection = _load("backend.reflection", "backend/reflection/__init__.py")
backend_pkg.self_monitoring = _load("backend.self_monitoring", "backend/self_monitoring/__init__.py")

monitor_module = _load("backend.self_monitoring.monitor", "backend/self_monitoring/monitor.py")
memory_module = _load("backend.memory.long_term", "backend/memory/long_term.py")
reflection_module = _load("backend.reflection.reflection", "backend/reflection/reflection.py")

StepReport = monitor_module.StepReport
SelfMonitoringSystem = monitor_module.SelfMonitoringSystem
LongTermMemory = memory_module.LongTermMemory
ReflectionModule = reflection_module.ReflectionModule
ReflectionResult = reflection_module.ReflectionResult


def test_step_report_summary_contains_details():
    report = StepReport(
        action="deploy",
        observation="deployment completed",
        status="success",
        metrics={"latency": 1.2345, "accuracy": 0.99},
        retries=1,
        metadata={"task_id": "task-123"},
    )
    summary = report.to_summary()
    assert "action=deploy" in summary
    assert "metrics=accuracy=0.99" in summary
    assert "retries=1" in summary
    assert '"task_id": "task-123"' in summary


def test_self_monitoring_triggers_recovery_and_persists(tmp_path):
    events: list[tuple[StepReport, ReflectionResult]] = []

    def evaluation(text: str) -> ReflectionResult:
        sentiment = "negative" if "failed" in text else "positive"
        score = 0.2 if "failed" in text else 0.9
        return ReflectionResult(score, sentiment, raw=text)

    def rewrite(text: str) -> str:
        return text + " :: revised"

    def recovery(report: StepReport, evaluation: ReflectionResult) -> None:
        events.append((report, evaluation))

    reflection = ReflectionModule(evaluate=evaluation, rewrite=rewrite, max_passes=1)
    memory = LongTermMemory(tmp_path / "monitoring.db")
    monitor = SelfMonitoringSystem(
        reflection=reflection,
        memory=memory,
        quality_threshold=0.5,
        recovery_hook=recovery,
    )

    report = StepReport(
        action="deploy",
        observation="deployment failed due to timeout",
        status="failed",
        metrics={"latency": 12.5},
        error="timeout",
        retries=0,
        metadata={"task_id": "release-42", "agent_id": "ops"},
    )

    decision = monitor.assess_step(report)
    assert decision.should_retry
    assert decision.adjustments is not None
    assert "severity" in decision.adjustments
    assert events, "Recovery hook should be invoked for low confidence outcomes"

    stored = list(memory.get("self_monitoring"))
    assert stored, "Self-monitoring history should be persisted"
    record = json.loads(stored[-1])
    assert record["should_retry"] is True
    assert record["evaluation"]["sentiment"] == "negative"

    memory.close()
