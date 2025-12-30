"""Docker (docker-py) controls for ToolEnvironmentBridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402
import BrainSimulationSystem.environment.tool_bridge as tool_bridge  # noqa: E402


@dataclass
class _FakeImage:
    name: str
    id: str = "sha256:deadbeef"

    @property
    def tags(self) -> List[str]:
        return [self.name]


@dataclass
class _FakeExecResult:
    exit_code: int
    output: bytes


class _FakeContainer:
    def __init__(self, cid: str, name: str) -> None:
        self.id = cid
        self.name = name
        self.status = "running"

    def stop(self, timeout: int = 10) -> None:
        self.status = "exited"

    def remove(self, *, force: bool = False, v: bool = False) -> None:
        self.status = "removed"

    def exec_run(self, cmd: Any, **kwargs: Any) -> _FakeExecResult:
        return _FakeExecResult(exit_code=0, output=b"ok")

    def logs(self, **kwargs: Any) -> bytes:
        return b"log"


class _FakeImages:
    def __init__(self) -> None:
        self.pulled: List[Dict[str, Any]] = []

    def pull(self, image: str, **kwargs: Any) -> _FakeImage:
        self.pulled.append({"image": image, **kwargs})
        return _FakeImage(image)


class _FakeContainers:
    def __init__(self) -> None:
        self._by_ref: Dict[str, _FakeContainer] = {}
        self._counter = 0

    def run(self, image: str, **kwargs: Any) -> _FakeContainer:
        self._counter += 1
        name = str(kwargs.get("name") or f"bss_{self._counter}")
        cid = f"container_{self._counter}"
        container = _FakeContainer(cid, name)
        self._by_ref[cid] = container
        self._by_ref[name] = container
        return container

    def get(self, ref: str) -> _FakeContainer:
        if ref not in self._by_ref:
            raise KeyError(f"not_found:{ref}")
        return self._by_ref[ref]


class _FakeDockerClient:
    def __init__(self) -> None:
        self.images = _FakeImages()
        self.containers = _FakeContainers()
        self.ping_calls = 0

    def ping(self) -> bool:
        self.ping_calls += 1
        return True


class _FakeDockerModule:
    def __init__(self, client: _FakeDockerClient) -> None:
        self._client = client
        self.from_env_calls: List[Dict[str, Any]] = []

    def from_env(self, **kwargs: Any) -> _FakeDockerClient:
        self.from_env_calls.append(dict(kwargs))
        return self._client


def _install_fake_docker(monkeypatch, client: Optional[_FakeDockerClient] = None) -> _FakeDockerClient:
    fake_client = client or _FakeDockerClient()
    monkeypatch.setattr(tool_bridge, "docker", _FakeDockerModule(fake_client), raising=True)
    return fake_client


def test_docker_blocked_by_default(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    obs, reward, terminated, info = bridge.step({"type": "docker", "action": "ping"})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "docker_control_disabled"


def test_docker_pull_requires_allowlist(tmp_path, monkeypatch):
    _install_fake_docker(monkeypatch)
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_docker_control=True)
    obs, reward, terminated, info = bridge.step({"type": "docker", "action": "images.pull", "image": "python:3.12"})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "docker_image_allowlist_not_configured"


def test_docker_pull_runs_with_allowlist(tmp_path, monkeypatch):
    fake_client = _install_fake_docker(monkeypatch)
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_docker_control=True,
        allowed_docker_image_prefixes=["python"],
    )
    obs, reward, terminated, info = bridge.step({"type": "docker", "action": "images.pull", "image": "python:3.12"})
    assert terminated is False
    assert info.get("action") == "images.pull"
    assert info.get("image") == "python:3.12"
    assert fake_client.images.pulled and fake_client.images.pulled[0]["image"] == "python:3.12"
    assert "pulled" in (obs.get("text") or "")


def test_docker_run_tracks_container_and_allows_stop_exec_logs(tmp_path, monkeypatch):
    _install_fake_docker(monkeypatch)
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_docker_control=True,
        allowed_docker_image_prefixes=["python"],
        allow_docker_delete=True,
    )

    obs, reward, terminated, info = bridge.step(
        {
            "type": "docker",
            "action": "containers.run",
            "image": "python:3.12",
            "name": "bss_test",
            "command": ["python", "-c", "print('hi')"],
            "detach": True,
        }
    )
    assert terminated is False
    assert info.get("action") == "containers.run"
    assert info.get("name") == "bss_test"

    obs, reward, terminated, info = bridge.step({"type": "docker", "action": "containers.stop", "container": "bss_test"})
    assert terminated is False
    assert info.get("stopped") is True

    obs, reward, terminated, info = bridge.step(
        {"type": "docker", "action": "containers.exec", "container": "bss_test", "cmd": ["echo", "ok"]}
    )
    assert terminated is False
    assert info.get("exit_code") == 0
    assert (obs.get("text") or "").strip() == "ok"

    obs, reward, terminated, info = bridge.step({"type": "docker", "action": "containers.logs", "container": "bss_test"})
    assert terminated is False
    assert (obs.get("text") or "").strip() == "log"

    obs, reward, terminated, info = bridge.step({"type": "docker", "action": "containers.remove", "container": "bss_test"})
    assert terminated is False
    assert info.get("removed") is True


def test_docker_stop_blocks_untracked_container(tmp_path, monkeypatch):
    _install_fake_docker(monkeypatch)
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_docker_control=True,
        allowed_docker_image_prefixes=["python"],
    )
    obs, reward, terminated, info = bridge.step({"type": "docker", "action": "containers.stop", "container": "other"})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "docker_container_not_tracked"

