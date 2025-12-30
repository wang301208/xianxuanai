from __future__ import annotations

import importlib.util
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


backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)

remediation_module = _load("backend.self_monitoring.remediation", "backend/self_monitoring/remediation.py")
health_module = _load("backend.self_monitoring.health", "backend/self_monitoring/health.py")
monitor_module = _load("backend.self_monitoring.monitor", "backend/self_monitoring/monitor.py")
guardian_module = _load("backend.self_monitoring.guardian", "backend/self_monitoring/guardian.py")

RemediationManager = remediation_module.RemediationManager
RemediationPatch = remediation_module.RemediationPatch
RemediationResult = remediation_module.RemediationResult
RemediationGuardian = guardian_module.RemediationGuardian
HealthMonitor = health_module.HealthMonitor
SensorReading = health_module.SensorReading
SelfMonitoringSystem = monitor_module.SelfMonitoringSystem
ReflectionModule = _load("backend.reflection.reflection", "backend/reflection/reflection.py").ReflectionModule


def test_remediation_manager_restart_and_fallback():
    manager = RemediationManager()
    state = {"restarted": False, "fallback": False}

    def restart(_ctx):
        state["restarted"] = True
        return True

    def fallback(_ctx):
        state["fallback"] = True
        return True

    manager.register_module("planner", restart=restart, fallback=fallback)
    result = manager.attempt("planner", {})
    assert isinstance(result, RemediationResult)
    assert result.success is True
    assert result.method == "restart"
    assert state["restarted"] is True
    assert state["fallback"] is False


def test_remediation_manager_fallback_when_restart_fails():
    manager = RemediationManager()

    def restart(_ctx):
        return False

    def fallback(_ctx):
        return True

    manager.register_module("navigator", restart=restart, fallback=fallback)
    result = manager.attempt("navigator", {})
    assert result.success
    assert result.method == "fallback"
    assert "fallback" in result.steps


def test_remediation_manager_respects_protected_patch():
    manager = RemediationManager()

    def patch(_ctx):
        return RemediationPatch(description="update safeguard", applied=True)

    manager.register_module("safety", patch=patch, protected=True)
    result = manager.attempt("safety", {}, allow_code_patch=True)
    assert not result.success
    assert result.patch is None
    assert "patch_denied_protected" in result.steps


def test_guardian_blocks_patch_without_approval():
    guardian = RemediationGuardian()

    def deny_sensitive(module: str, action: str, _ctx):
        if action == "patch":
            return False, "requires_dual_control"
        return True, None

    guardian.add_rule("dual_control", deny_sensitive)

    manager = RemediationManager(guardian=guardian)

    def patch(_ctx):
        return RemediationPatch(description="fix bug", applied=True)

    manager.register_module("llm_core", patch=patch)
    result = manager.attempt("llm_core", {}, allow_code_patch=True)
    assert not result.success
    assert result.patch is None
    assert result.error == "requires_dual_control"
    assert any(step.startswith("patch_denied:") for step in result.steps)


def test_guardian_allows_patch_with_clearance():
    guardian = RemediationGuardian()

    def require_flag(module: str, action: str, ctx):
        if action == "patch":
            allowed = bool(ctx.get("clearance"))
            return allowed, None if allowed else "missing_clearance"
        return True, None

    guardian.add_rule("safety_flag", require_flag)

    manager = RemediationManager(guardian=guardian)

    def patch(_ctx):
        return RemediationPatch(description="hotfix", applied=True, tests_run=["pytest"])

    manager.register_module("planner", patch=patch)
    result = manager.attempt("planner", {"clearance": True}, allow_code_patch=True)
    assert result.success
    assert result.method == "patch"
    assert result.patch is not None
    assert result.patch.applied is True


def test_health_monitor_invokes_remediation(tmp_path):
    restart_invoked = {"value": False}

    manager = RemediationManager()

    def restart(_ctx):
        restart_invoked["value"] = True
        return True

    manager.register_module("long_term_memory", restart=restart)

    reflection = ReflectionModule(
        evaluate=lambda text: monitor_module.ReflectionResult(0.1, "negative", raw=text),
        rewrite=lambda text: text + " :: revised",
        max_passes=1,
    )
    monitor = SelfMonitoringSystem(reflection=reflection)
    health_monitor = HealthMonitor(self_monitor=monitor, remediator=manager)

    def faulty_sensor():
        return SensorReading(
            name="long_term_memory",
            status="error",
            value=1.0,
            threshold=0.1,
            category="memory",
            message="memory locked",
            metadata={"module": "long_term_memory"},
        )

    health_monitor.register_sensor("long_term_memory", faulty_sensor)
    alerts = health_monitor.evaluate()

    assert restart_invoked["value"] is True
    assert alerts
    assert alerts[0].remediation is not None
    assert alerts[0].remediation.success is True
    assert alerts[0].remediation.method == "restart"

