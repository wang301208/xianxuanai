"""System command and setting controls for ToolEnvironmentBridge."""

from __future__ import annotations

from pathlib import Path
import os
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402


def test_exec_system_cmd_blocked_by_default(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    obs, reward, terminated, info = bridge.step(
        {"type": "exec_system_cmd", "command": [sys.executable, "-c", "print(1)"]}
    )

    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "system_cmd_disabled"


def test_exec_system_cmd_requires_allowlist(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_system_cmd=True)
    obs, reward, terminated, info = bridge.step(
        {"type": "exec_system_cmd", "command": [sys.executable, "-c", "print(1)"]}
    )

    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "system_cmd_not_allowed"


def test_exec_system_cmd_runs_allowed_command_list(tmp_path):
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_system_cmd=True,
        allowed_system_cmd_prefixes=[[sys.executable]],
    )
    obs, reward, terminated, info = bridge.step(
        {"type": "exec_system_cmd", "command": [sys.executable, "-c", "print('ok')"]}
    )

    assert terminated is False
    assert info.get("returncode") == 0
    assert "ok" in (obs.get("text") or "")


def test_exec_system_cmd_parses_string_cmd(tmp_path):
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_system_cmd=True,
        allowed_system_cmd_prefixes=[[sys.executable]],
    )
    cmd = f'"{sys.executable}" -c "print(123)"'
    obs, reward, terminated, info = bridge.step({"type": "exec_system_cmd", "cmd": cmd})

    assert terminated is False
    assert info.get("returncode") == 0
    assert (obs.get("text") or "").strip() == "123"


def test_high_risk_system_cmd_blocked_without_explicit_enable(tmp_path):
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_system_cmd=True,
        allowed_system_cmd_prefixes=[["shutdown"]],
    )

    obs, reward, terminated, info = bridge.step(
        {"type": "exec_system_cmd", "command": ["shutdown", "/t", "0"], "dry_run": True}
    )

    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "high_risk_command_blocked"


def test_high_risk_system_cmd_requires_confirmation_token(tmp_path):
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_system_cmd=True,
        allowed_system_cmd_prefixes=[["shutdown"]],
        allow_high_risk_system_cmd=True,
        system_confirm_token="TOKEN",
    )

    obs, reward, terminated, info = bridge.step(
        {"type": "exec_system_cmd", "command": ["shutdown", "/t", "0"], "dry_run": True}
    )
    assert info.get("blocked") is True
    assert info.get("reason") == "confirmation_required"

    obs, reward, terminated, info = bridge.step(
        {
            "type": "exec_system_cmd",
            "command": ["shutdown", "/t", "0"],
            "dry_run": True,
            "confirm_token": "TOKEN",
        }
    )
    assert terminated is False
    assert info.get("dry_run") is True
    assert info.get("high_risk") is True
    assert (obs.get("text") or "") == "dry_run"


def test_change_system_setting_shutdown_maps_to_command_and_enforces_confirmation(tmp_path):
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_system_cmd=True,
        allowed_system_cmd_prefixes=[["shutdown"]],
        allow_high_risk_system_cmd=True,
        system_confirm_token="TOKEN",
    )

    obs, reward, terminated, info = bridge.step(
        {
            "type": "change_system_setting",
            "name": "power.shutdown",
            "value": {"delay_s": 0},
            "dry_run": True,
            "confirm_token": "TOKEN",
        }
    )

    assert terminated is False
    assert info.get("dry_run") is True
    assert info.get("setting_name") == "power.shutdown"
    assert info.get("high_risk") is True
    assert isinstance(info.get("command"), list)
    assert os.path.basename(str(info.get("command")[0])).lower().startswith("shutdown")

