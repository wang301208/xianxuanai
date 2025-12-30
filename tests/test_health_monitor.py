from __future__ import annotations

import importlib.util
import shutil
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


# Expose backend package for relative imports
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)

# Load required backend modules
backend_pkg.memory = _load("backend.memory", "backend/memory/__init__.py")
backend_pkg.reflection = _load("backend.reflection", "backend/reflection/__init__.py")
backend_pkg.self_monitoring = _load("backend.self_monitoring", "backend/self_monitoring/__init__.py")

health_module = _load("backend.self_monitoring.health", "backend/self_monitoring/health.py")
diagnostics_module = _load("backend.self_monitoring.diagnostics", "backend/self_monitoring/diagnostics.py")
monitor_module = _load("backend.self_monitoring.monitor", "backend/self_monitoring/monitor.py")
reflection_module = _load("backend.reflection.reflection", "backend/reflection/reflection.py")
memory_module = _load("backend.memory.long_term", "backend/memory/long_term.py")

SensorReading = health_module.SensorReading
HealthMonitor = health_module.HealthMonitor
SystemDiagnostics = diagnostics_module.SystemDiagnostics
DiagnosticReport = diagnostics_module.DiagnosticReport
SelfMonitoringSystem = monitor_module.SelfMonitoringSystem
ReflectionModule = reflection_module.ReflectionModule
ReflectionResult = reflection_module.ReflectionResult
LongTermMemory = memory_module.LongTermMemory


def test_health_monitor_triggers_memory_repair(tmp_path):
    db_path = tmp_path / "memory.db"
    backup_path = tmp_path / "memory_backup.db"

    memory = LongTermMemory(db_path)
    memory.add("init", "ok")
    memory.conn.commit()
    memory.close()
    shutil.copy2(db_path, backup_path)

    broken_memory = LongTermMemory(db_path)
    broken_memory.close()

    def evaluation(text: str) -> ReflectionResult:
        score = 0.2 if "sensor" in text else 0.9
        sentiment = "negative" if score < 0.5 else "positive"
        return ReflectionResult(score, sentiment, raw=text)

    reflection = ReflectionModule(evaluate=evaluation, rewrite=lambda t: t + " :: revised", max_passes=1)
    monitor = SelfMonitoringSystem(reflection=reflection, quality_threshold=0.5)
    diagnostics = SystemDiagnostics()
    health_monitor = HealthMonitor(self_monitor=monitor, diagnostics=diagnostics)

    def memory_sensor() -> SensorReading:
        return SensorReading(
            name="long_term_memory",
            value=1.0,
            threshold=0.1,
            status="error",
            category="memory",
            message="Unable to write to long-term memory store",
            metadata={
                "memory": broken_memory,
                "path": db_path,
                "backup_path": backup_path,
            },
        )

    health_monitor.register_sensor("long_term_memory", memory_sensor)
    alerts = health_monitor.evaluate()
    assert alerts, "Expected an alert for the broken memory store"

    alert = alerts[0]
    assert isinstance(alert.diagnostic, DiagnosticReport)
    restored_memory = alert.diagnostic.metadata.get("memory")
    assert restored_memory is not None
    assert getattr(restored_memory, "path", None) == db_path

    restored_memory.add("health_check", "restored")
    restored_memory.close()


def test_system_diagnostics_restores_model_snapshot(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "snapshot_a.bin").write_text("old")
    newer = snapshots / "snapshot_b.bin"
    newer.write_text("newer")

    model_path = tmp_path / "deployed.bin"
    model_path.write_text("stale")

    diagnostics = SystemDiagnostics()
    restored, target, actions = diagnostics.restore_model_snapshot(
        model_path=model_path,
        snapshot_dir=snapshots,
        snapshot_name="snapshot_b.bin",
    )

    assert restored is True
    assert target == model_path
    assert model_path.read_text() == "newer"
    assert any(action.startswith("model_restored") for action in actions)
