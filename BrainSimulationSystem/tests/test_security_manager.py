"""SecurityManager permissions, approvals, audit, and sandbox integration."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.security_manager import SecurityManager  # noqa: E402
from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402


def test_security_manager_blocks_insufficient_permission():
    security = SecurityManager({"permission_level": "read_only"})
    decision = security.decide({"type": "write_file", "path": "x.txt", "text": "hi"})
    assert decision.blocked is True
    assert decision.reason == "permission_denied"


def test_security_manager_requires_and_grants_approval(tmp_path):
    security = SecurityManager(
        {
            "permission_level": "admin",
            "require_approval_for": ["delete_file"],
            "approval_token": "ADMIN",
        }
    )
    action = {"type": "delete_file", "path": str(tmp_path / "x.txt")}
    decision1 = security.decide(action)
    assert decision1.blocked is True
    assert decision1.reason == "approval_required"
    assert decision1.approval_id

    # Wrong token cannot approve.
    assert security.approve(decision1.approval_id, token="WRONG") is False
    assert security.approve(decision1.approval_id, token="ADMIN") is True

    decision2 = security.decide({**action, "approval_id": decision1.approval_id})
    assert decision2.blocked is False
    assert decision2.reason == "approval_granted"

    # Approval id cannot be reused for a different action payload.
    decision3 = security.decide({"type": "delete_file", "path": str(tmp_path / "y.txt"), "approval_id": decision1.approval_id})
    assert decision3.blocked is True
    assert decision3.reason == "approval_action_mismatch"


def test_security_manager_deduplicates_identical_approval_requests(tmp_path):
    security = SecurityManager({"permission_level": "admin", "require_approval_for": ["delete_file"]})
    action = {"type": "delete_file", "path": str(tmp_path / "x.txt")}
    d1 = security.decide(action)
    d2 = security.decide(action)
    assert d1.approval_id == d2.approval_id


def test_security_manager_audit_log_writes_jsonl(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    security = SecurityManager({"audit_enabled": True, "audit_log_path": str(audit_path)})
    security.decide({"type": "read_file", "path": "x.txt"})
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    assert "\"event\":\"decision\"" in lines[0]


def test_tool_bridge_security_manager_approval_roundtrip(tmp_path):
    target = tmp_path / "deleteme.txt"
    target.write_text("x", encoding="utf-8")

    security = SecurityManager({"permission_level": "admin", "require_approval_for": ["delete_file"]})
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_file_delete=True, security_manager=security)

    obs, reward, terminated, info = bridge.step({"type": "delete_file", "path": str(target)})
    assert info.get("blocked") is True
    assert info.get("reason") == "approval_required"
    approval_id = info.get("approval_id")
    assert approval_id
    assert target.exists() is True

    assert security.approve(approval_id) is True
    obs, reward, terminated, info = bridge.step({"type": "delete_file", "path": str(target), "approval_id": approval_id})
    assert info.get("deleted") is True
    assert target.exists() is False


def test_tool_bridge_filesystem_sandbox_overlay_and_commit(tmp_path):
    sandbox_root = tmp_path / ".sandbox"
    target = tmp_path / "note.txt"

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_file_write=True,
        allow_file_delete=True,
        filesystem_sandbox={"enabled": True, "root": str(sandbox_root), "keep_history": True},
        allow_sandbox_commit=True,
        sandbox_confirm_token="OK",
    )

    obs, reward, terminated, info = bridge.step({"type": "write_file", "path": str(target), "text": "hello"})
    assert info.get("sandboxed") is True
    assert target.exists() is False

    obs, reward, terminated, info = bridge.step({"type": "read_file", "path": str(target), "max_chars": 100})
    assert "hello" in (obs.get("text") or "")
    assert info.get("sandboxed") is True

    # Commit is gated by confirm token.
    obs, reward, terminated, info = bridge.step({"type": "sandbox_commit"})
    assert info.get("blocked") is True
    assert info.get("reason") == "sandbox_confirm_token_required"

    obs, reward, terminated, info = bridge.step({"type": "sandbox_commit", "confirm_token": "OK"})
    assert info.get("copied_files") >= 1
    assert target.exists() is True
    assert target.read_text(encoding="utf-8") == "hello"

