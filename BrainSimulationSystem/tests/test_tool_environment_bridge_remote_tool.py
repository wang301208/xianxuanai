"""Remote tool proxy support for ToolEnvironmentBridge."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.remote_tool import RemoteToolServer  # noqa: E402
from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402


def test_remote_tool_blocked_by_default(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    obs, reward, terminated, info = bridge.step(
        {
            "type": "remote_tool",
            "endpoint": "127.0.0.1:9999",
            "token": "TOKEN",
            "action": {"type": "read_file", "path": str(tmp_path / "x.txt")},
        }
    )
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "remote_control_disabled"


def test_remote_tool_requires_allowlist_and_token(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_remote_control=True)
    obs, reward, terminated, info = bridge.step(
        {
            "type": "remote_tool",
            "endpoint": "127.0.0.1:9999",
            "token": "TOKEN",
            "action": {"type": "read_file", "path": str(tmp_path / "x.txt")},
        }
    )
    assert info.get("blocked") is True
    assert info.get("reason") == "remote_allowlist_not_configured"

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_remote_control=True,
        allowed_remote_endpoints=["127.0.0.1:9999"],
    )
    obs, reward, terminated, info = bridge.step(
        {
            "type": "remote_tool",
            "endpoint": "127.0.0.1:9999",
            "action": {"type": "read_file", "path": str(tmp_path / "x.txt")},
        }
    )
    assert info.get("blocked") is True
    assert info.get("reason") == "remote_auth_token_not_configured"


def test_remote_tool_roundtrip_step_and_reset(tmp_path):
    # Remote server (acts as the "other machine").
    remote_bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allowed_shell_prefixes=[[sys.executable]],
    )
    with RemoteToolServer(remote_bridge, auth_token="TOKEN") as server:
        host, port = server.address
        endpoint = f"{host}:{port}"

        # Local bridge proxying into the remote server.
        bridge = ToolEnvironmentBridge(
            allowed_roots=[tmp_path],
            allow_remote_control=True,
            allowed_remote_endpoints=[endpoint],
            remote_auth_token="TOKEN",
        )

        obs, reward, terminated, info = bridge.step({"type": "remote_tool", "endpoint": endpoint, "method": "reset"})
        assert terminated is False
        assert info.get("endpoint") == endpoint
        assert info.get("method") == "reset"
        assert obs.get("tool_state", {}).get("steps") == 1

        path = tmp_path / "note.txt"
        path.write_text("hello", encoding="utf-8")

        obs, reward, terminated, info = bridge.step(
            {
                "type": "remote_tool",
                "endpoint": endpoint,
                "action": {"type": "read_file", "path": str(path), "max_chars": 100},
            }
        )
        assert terminated is False
        assert info.get("endpoint") == endpoint
        assert info.get("method") == "step"
        assert "hello" in (obs.get("text") or "")
        remote_observation = obs.get("remote_observation", {})
        assert isinstance(remote_observation, dict)
        assert "hello" in (remote_observation.get("text") or "")


def test_remote_tool_blocks_disallowed_action_type(tmp_path):
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_remote_control=True,
        allowed_remote_endpoints=["127.0.0.1:9999"],
        remote_auth_token="TOKEN",
        # Leave allowed_remote_action_types unset -> defaults to read/list/shell.
    )
    obs, reward, terminated, info = bridge.step(
        {
            "type": "remote_tool",
            "endpoint": "127.0.0.1:9999",
            "action": {"type": "write_file", "path": str(tmp_path / "x.txt"), "text": "hi"},
        }
    )
    assert info.get("blocked") is True
    assert info.get("reason") == "remote_action_type_not_allowed"


def test_remote_tool_auth_failure(tmp_path):
    remote_bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    with RemoteToolServer(remote_bridge, auth_token="TOKEN") as server:
        host, port = server.address
        endpoint = f"{host}:{port}"

        bridge = ToolEnvironmentBridge(
            allowed_roots=[tmp_path],
            allow_remote_control=True,
            allowed_remote_endpoints=[endpoint],
            remote_auth_token="WRONG",
        )
        obs, reward, terminated, info = bridge.step(
            {
                "type": "remote_tool",
                "endpoint": endpoint,
                "action": {"type": "list_dir", "path": str(tmp_path)},
            }
        )
        assert info.get("blocked") is True
        assert info.get("reason") == "remote_auth_failed"

