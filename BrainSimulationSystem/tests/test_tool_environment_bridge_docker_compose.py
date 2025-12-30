"""Docker Compose (CLI) controls for ToolEnvironmentBridge."""

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


def _install_fake_subprocess_run(monkeypatch, calls: List[Dict[str, Any]]) -> None:
    def fake_run(cmd: List[str], **kwargs: Any) -> _FakeCompleted:
        calls.append({"cmd": list(cmd), "kwargs": dict(kwargs)})
        return _FakeCompleted(stdout="ok\n", stderr="", returncode=0)

    monkeypatch.setattr(tool_bridge.subprocess, "run", fake_run, raising=True)


def test_docker_compose_blocked_by_default(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    obs, reward, terminated, info = bridge.step(
        {"type": "docker_compose", "action": "up", "project_dir": str(tmp_path)}
    )
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "docker_compose_disabled"


def test_docker_compose_up_runs(tmp_path, monkeypatch):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")

    calls: List[Dict[str, Any]] = []
    _install_fake_subprocess_run(monkeypatch, calls)

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_docker_compose=True)
    obs, reward, terminated, info = bridge.step(
        {
            "type": "docker_compose",
            "action": "up",
            "project_dir": str(tmp_path),
            "files": ["docker-compose.yml"],
            "detach": True,
        }
    )
    assert terminated is False
    assert info.get("returncode") == 0
    assert (obs.get("text") or "").strip() == "ok"

    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert cmd[:2] == ["docker", "compose"]
    assert "up" in cmd
    assert "-d" in cmd
    assert str(compose_file.resolve()) in cmd
    assert calls[0]["kwargs"].get("cwd") == str(tmp_path.resolve())


def test_docker_compose_down_requires_delete_enable(tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_docker_compose=True)
    obs, reward, terminated, info = bridge.step(
        {
            "type": "docker_compose",
            "action": "down",
            "project_dir": str(tmp_path),
            "files": [str(compose_file)],
        }
    )
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "docker_compose_delete_disabled"


def test_docker_compose_scale_builds_up_command(tmp_path, monkeypatch):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {web: {image: python:3.12}}\n", encoding="utf-8")

    calls: List[Dict[str, Any]] = []
    _install_fake_subprocess_run(monkeypatch, calls)

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_docker_compose=True)
    obs, reward, terminated, info = bridge.step(
        {
            "type": "docker_compose",
            "action": "scale",
            "project_dir": str(tmp_path),
            "files": ["docker-compose.yml"],
            "services": {"web": 3},
        }
    )
    assert terminated is False
    assert info.get("returncode") == 0
    assert len(calls) == 1

    cmd = calls[0]["cmd"]
    assert cmd[:2] == ["docker", "compose"]
    assert "up" in cmd
    assert "-d" in cmd
    assert "--scale" in cmd
    assert "web=3" in cmd
    assert "web" in cmd


def test_docker_compose_project_dir_must_be_allowed(tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_docker_compose=True)
    obs, reward, terminated, info = bridge.step(
        {
            "type": "docker_compose",
            "action": "up",
            "project_dir": str(tmp_path.parent),
            "files": [str(compose_file)],
            "detach": True,
        }
    )
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "path_not_allowed"
    assert info.get("context") == "project_dir"


def test_docker_compose_dry_run_returns_command(tmp_path, monkeypatch):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")

    calls: List[Dict[str, Any]] = []
    _install_fake_subprocess_run(monkeypatch, calls)

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_docker_compose=True)
    obs, reward, terminated, info = bridge.step(
        {
            "type": "docker_compose",
            "action": "up",
            "project_dir": str(tmp_path),
            "files": ["docker-compose.yml"],
            "detach": True,
            "dry_run": True,
        }
    )
    assert terminated is False
    assert info.get("dry_run") is True
    assert isinstance(info.get("command"), list)
    assert calls == []

