"""Script execution controls for ToolEnvironmentBridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402
import BrainSimulationSystem.environment.tool_bridge as tool_bridge  # noqa: E402


@dataclass
class _FakeCompleted:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def _install_fake_subprocess_run(monkeypatch, calls: List[Dict[str, Any]], *, stdout: str = "ok\n") -> None:
    def fake_run(cmd: List[str], **kwargs: Any) -> _FakeCompleted:
        calls.append({"cmd": list(cmd), "kwargs": dict(kwargs)})
        return _FakeCompleted(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(tool_bridge.subprocess, "run", fake_run, raising=True)


def test_run_script_blocked_by_default(tmp_path):
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    obs, reward, terminated, info = bridge.step({"type": "run_script", "path": str(script)})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "script_execution_disabled"


def test_run_script_requires_allowlist(tmp_path):
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_script_execution=True)
    obs, reward, terminated, info = bridge.step({"type": "run_script", "path": str(script)})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "script_allowlist_not_configured"


def test_run_script_blocks_not_allowlisted(tmp_path):
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    other = tmp_path / "other.py"
    other.write_text("print('x')\n", encoding="utf-8")

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_script_execution=True,
        allowed_script_paths=[str(script)],
    )
    obs, reward, terminated, info = bridge.step({"type": "run_script", "path": str(other)})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "script_not_allowed"


def test_run_script_dry_run_returns_command(tmp_path, monkeypatch):
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    calls: List[Dict[str, Any]] = []
    _install_fake_subprocess_run(monkeypatch, calls)

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_script_execution=True,
        allowed_script_paths=[str(script)],
    )
    obs, reward, terminated, info = bridge.step({"type": "run_script", "path": str(script), "dry_run": True})
    assert terminated is False
    assert info.get("dry_run") is True
    assert isinstance(info.get("command"), list)
    assert calls == []


def test_run_script_runs_allowlisted_python_script(tmp_path, monkeypatch):
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    calls: List[Dict[str, Any]] = []
    _install_fake_subprocess_run(monkeypatch, calls, stdout="hello\n")

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_script_execution=True,
        allowed_script_paths=[str(script)],
        max_script_output_chars=100,
    )
    obs, reward, terminated, info = bridge.step({"type": "run_script", "path": str(script), "args": ["a", "b"]})
    assert terminated is False
    assert info.get("returncode") == 0
    assert (obs.get("text") or "").strip() == "hello"

    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert cmd[0] == sys.executable
    assert Path(cmd[1]).resolve() == script.resolve()
    assert cmd[2:] == ["a", "b"]
    assert calls[0]["kwargs"].get("cwd") == str(tmp_path.resolve())


def test_run_script_rejects_unknown_extension(tmp_path):
    script = tmp_path / "job.txt"
    script.write_text("noop\n", encoding="utf-8")

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_script_execution=True,
        allowed_script_paths=[str(script)],
    )
    obs, reward, terminated, info = bridge.step({"type": "run_script", "path": str(script)})
    assert terminated is False
    assert info.get("error") == "unsupported_script_type"


def test_run_script_blocks_obviously_dangerous_python(tmp_path, monkeypatch):
    script = tmp_path / "job.py"
    script.write_text("import os\nos.system('echo hi')\n", encoding="utf-8")

    calls: List[Dict[str, Any]] = []
    _install_fake_subprocess_run(monkeypatch, calls, stdout="should not run\n")

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_script_execution=True,
        allowed_script_paths=[str(script)],
    )
    obs, reward, terminated, info = bridge.step({"type": "run_script", "path": str(script)})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "script_failed_security_scan"
    assert calls == []
    scan = info.get("scan")
    assert isinstance(scan, dict)
    assert scan.get("ok") is False


def test_run_script_executes_filesystem_sandbox_overlay(tmp_path, monkeypatch):
    script = tmp_path / "job.py"
    sandbox_root = tmp_path / ".sandbox"

    calls: List[Dict[str, Any]] = []
    _install_fake_subprocess_run(monkeypatch, calls, stdout="hello\n")

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_file_write=True,
        allow_script_execution=True,
        allowed_script_paths=[str(script)],
        filesystem_sandbox={"enabled": True, "root": str(sandbox_root), "keep_history": True},
    )
    # Write into the sandbox overlay (no commit).
    obs, reward, terminated, info = bridge.step({"type": "write_file", "path": str(script), "text": "print('hi')\n"})
    assert info.get("sandboxed") is True
    assert script.exists() is False

    obs, reward, terminated, info = bridge.step({"type": "run_script", "path": str(script)})
    assert terminated is False
    assert info.get("returncode") == 0
    assert info.get("sandboxed") is True
    assert info.get("script_source") == "sandbox"
    assert (obs.get("text") or "").strip() == "hello"

    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert cmd[0] == sys.executable
    assert sandbox_root.resolve() in Path(cmd[1]).resolve().parents
