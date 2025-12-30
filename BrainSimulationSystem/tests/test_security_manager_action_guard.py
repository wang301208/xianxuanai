"""SecurityManager integration with knowledge-graph ActionGuard (opt-in)."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def test_security_manager_blocks_action_when_action_guard_denies(monkeypatch):
    module_path = ROOT_DIR / "BrainSimulationSystem" / "environment" / "security_manager.py"
    spec = importlib.util.spec_from_file_location("bss_security_manager", module_path)
    assert spec and spec.loader
    sm_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sm_mod
    spec.loader.exec_module(sm_mod)
    SecurityManager = sm_mod.SecurityManager

    class _Result:
        def __init__(self, allowed: bool, reason: str = "", violations=None):
            self.allowed = allowed
            self.reason = reason
            self.violations = list(violations or [])

    class _Guard:
        def evaluate(self, command_name, command_args, *, context=None):
            assert command_name == "write_file"
            assert isinstance(command_args, dict)
            return _Result(False, reason="deny_test", violations=["no_write"])

    monkeypatch.setattr(sm_mod, "ActionGuard", _Guard)

    mgr = SecurityManager({"enabled": True, "action_guard_enabled": True})
    decision = mgr.decide({"type": "write_file", "path": "x.txt", "text": "hi"})

    assert decision.blocked is True
    assert decision.reason == "action_guard_denied"
    assert decision.details.get("guard_reason") == "deny_test"
