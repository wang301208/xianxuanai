"""Extended ToolEnvironmentBridge actions and safety checks."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402


def test_write_file_blocked_by_default(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])

    obs, reward, terminated, info = bridge.step(
        {"type": "write_file", "path": str(tmp_path / "note.txt"), "text": "hello"}
    )

    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "write_disabled"
    assert (tmp_path / "note.txt").exists() is False


def test_file_create_modify_delete_roundtrip(tmp_path):
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_file_write=True,
        allow_file_delete=True,
    )

    work_dir = tmp_path / "workspace"
    obs, reward, terminated, info = bridge.step({"type": "create_dir", "path": str(work_dir)})
    assert terminated is False
    assert work_dir.is_dir()

    path = work_dir / "note.txt"
    bridge.step({"type": "create_file", "path": str(path), "text": "hello world\n"})
    assert path.read_text(encoding="utf-8") == "hello world\n"

    bridge.step({"type": "modify_file", "path": str(path), "operation": "replace", "old": "world", "new": "tool"})
    assert path.read_text(encoding="utf-8") == "hello tool\n"

    bridge.step({"type": "modify_file", "path": str(path), "operation": "regex_replace", "pattern": r"tool", "repl": "bridge"})
    assert path.read_text(encoding="utf-8") == "hello bridge\n"

    obs, reward, terminated, info = bridge.step({"type": "delete_file", "path": str(path)})
    assert terminated is False
    assert info.get("deleted") is True
    assert path.exists() is False


def test_launch_program_requires_allowlist_and_explicit_enable(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    obs, reward, terminated, info = bridge.step({"type": "launch_program", "command": [sys.executable, "-c", "print(1)"]})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "process_control_disabled"

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_process_control=True)
    obs, reward, terminated, info = bridge.step({"type": "launch_program", "command": [sys.executable, "-c", "print(1)"]})
    assert info.get("blocked") is True
    assert info.get("reason") == "program_not_allowed"


def test_launch_and_kill_tracked_process(tmp_path):
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_process_control=True,
        allowed_program_prefixes=[[sys.executable]],
    )

    obs, reward, terminated, info = bridge.step(
        {"type": "launch_program", "command": [sys.executable, "-c", "import time; time.sleep(10)"]}
    )
    assert terminated is False
    pid = info.get("pid")
    assert isinstance(pid, int) and pid > 0

    obs, reward, terminated, info = bridge.step({"type": "kill_process", "pid": pid, "timeout_s": 1.0, "force": True})
    assert terminated is False
    assert info.get("killed") is True
    assert info.get("tracked") is True

