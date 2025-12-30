"""Security controls for environment actions (permissions, approvals, auditing).

This module provides a centralized, dependency-light security layer that can be
attached to :class:`~BrainSimulationSystem.environment.tool_bridge.ToolEnvironmentBridge`
and other executors to enforce:
  - Action allow/deny lists
  - Permission levels (graded capabilities)
  - Human approval workflow for high-risk actions
  - Audit logging (JSONL) for external side effects
  - Emergency stop kill-switch via environment variable

The security manager is **opt-in**: existing components keep their conservative
defaults (disabled-by-default features + allowlists). When enabled, this layer
adds an additional gate before actions are executed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional knowledge-graph action guard
    from modules.knowledge.action_guard import ActionGuard
except Exception:  # pragma: no cover
    ActionGuard = None  # type: ignore[assignment]


class PermissionLevel(IntEnum):
    """Graded permission levels for local/remote side effects."""

    READ_ONLY = 10
    USER = 20
    POWER_USER = 30
    ADMIN = 40

    @classmethod
    def from_value(cls, value: Any) -> "PermissionLevel":
        if isinstance(value, PermissionLevel):
            return value
        if isinstance(value, int):
            for member in cls:
                if int(member) == int(value):
                    return member
            # Clamp unknown numeric values.
            return cls.ADMIN if value >= int(cls.ADMIN) else cls.READ_ONLY
        text = str(value or "").strip().lower()
        mapping = {
            "read": cls.READ_ONLY,
            "read_only": cls.READ_ONLY,
            "readonly": cls.READ_ONLY,
            "user": cls.USER,
            "standard": cls.USER,
            "power": cls.POWER_USER,
            "power_user": cls.POWER_USER,
            "admin": cls.ADMIN,
            "root": cls.ADMIN,
        }
        return mapping.get(text, cls.USER)


_READ_ONLY_ACTIONS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_dir",
        "sandbox_status",
        "code_index_build",
        "code_index_search",
        "knowledge_import_directory",
        "knowledge_query",
        "parse_code",
        "summarize_doc",
        "terminate",
    }
)
_FS_WRITE_ACTIONS: frozenset[str] = frozenset({"write_file", "create_file", "modify_file", "create_dir"})
_FS_DELETE_ACTIONS: frozenset[str] = frozenset({"delete_file"})
_REPO_INGEST_ACTIONS: frozenset[str] = frozenset({"github_repo_ingest"})
_WEB_ACTIONS: frozenset[str] = frozenset(
    {"web_search", "web_scrape", "web_get", "github_code_search", "documentation_tool"}
)
_HUMAN_ACTIONS: frozenset[str] = frozenset({"ask_human"})
_PROCESS_ACTIONS: frozenset[str] = frozenset({"launch_program", "kill_process"})
_EXEC_ACTIONS: frozenset[str] = frozenset({"shell", "exec_system_cmd", "change_system_setting"})
_DOCKER_ACTIONS: frozenset[str] = frozenset({"docker"})
_DOCKER_COMPOSE_ACTIONS: frozenset[str] = frozenset({"docker_compose"})
_SCRIPT_ACTIONS: frozenset[str] = frozenset({"run_script"})
_REMOTE_ACTIONS: frozenset[str] = frozenset({"remote_tool"})
_UI_ACTIONS: frozenset[str] = frozenset({"ui", "motor"})


DEFAULT_REQUIRED_PERMISSION: Dict[str, PermissionLevel] = {
    **{a: PermissionLevel.READ_ONLY for a in _READ_ONLY_ACTIONS},
    **{a: PermissionLevel.USER for a in _FS_WRITE_ACTIONS},
    **{a: PermissionLevel.USER for a in _REPO_INGEST_ACTIONS},
    **{a: PermissionLevel.USER for a in _WEB_ACTIONS},
    **{a: PermissionLevel.USER for a in _HUMAN_ACTIONS},
    **{a: PermissionLevel.POWER_USER for a in _PROCESS_ACTIONS},
    **{a: PermissionLevel.POWER_USER for a in _EXEC_ACTIONS},
    **{a: PermissionLevel.POWER_USER for a in _DOCKER_ACTIONS},
    **{a: PermissionLevel.POWER_USER for a in _DOCKER_COMPOSE_ACTIONS},
    **{a: PermissionLevel.POWER_USER for a in _SCRIPT_ACTIONS},
    **{a: PermissionLevel.ADMIN for a in _FS_DELETE_ACTIONS},
    **{a: PermissionLevel.ADMIN for a in _REMOTE_ACTIONS},
    **{a: PermissionLevel.POWER_USER for a in _UI_ACTIONS},
}


DEFAULT_APPROVAL_REQUIRED: Tuple[str, ...] = (
    "delete_file",
    "exec_system_cmd",
    "change_system_setting",
    "kill_process",
    "docker",
    "docker_compose",
    "run_script",
    "remote_tool",
    "sandbox_commit",
    "sandbox_reset",
    "ui",
    "motor",
)


def _safe_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)


def _compile_patterns(patterns: Iterable[str]) -> Tuple[re.Pattern[str], ...]:
    compiled: List[re.Pattern[str]] = []
    for entry in patterns:
        try:
            compiled.append(re.compile(str(entry), re.IGNORECASE))
        except re.error:
            continue
    return tuple(compiled)


_FINGERPRINT_IGNORE_KEYS: frozenset[str] = frozenset(
    {
        "approval_id",
        "confirm_token",
        "token",
        "auth_token",
        "remote_auth_token",
        "system_confirm_token",
    }
)


def _normalize_action_for_fingerprint(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_action_for_fingerprint(v) for k, v in value.items() if k not in _FINGERPRINT_IGNORE_KEYS}
    if isinstance(value, list):
        return [_normalize_action_for_fingerprint(v) for v in value]
    return value


def action_fingerprint(action: Dict[str, Any]) -> str:
    normalized = _normalize_action_for_fingerprint(action or {})
    payload = _safe_json_dumps(normalized)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _redact_value(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, str):
        if max_chars > 0 and len(value) > max_chars:
            return value[:max_chars] + "..."
        return value
    if isinstance(value, dict):
        return {k: _redact_value(v, max_chars=max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v, max_chars=max_chars) for v in value]
    return value


def redact_action(action: Dict[str, Any], *, max_chars: int = 512) -> Dict[str, Any]:
    redacted = _redact_value(dict(action or {}), max_chars=max_chars)
    # Avoid logging large file payloads by default.
    for key in ("text", "content"):
        if key in redacted and isinstance(redacted[key], str) and max_chars >= 0:
            redacted[key] = _redact_value(redacted[key], max_chars=max_chars)
    return redacted


@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    created_at: float
    fingerprint: str
    action: Dict[str, Any]
    reason: str
    actor: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | approved | denied
    decided_at: Optional[float] = None
    decided_by: Optional[str] = None
    decision_note: Optional[str] = None


class ApprovalQueue:
    """In-memory approval queue with simple de-duplication by fingerprint."""

    def __init__(self, *, max_pending: int = 1000) -> None:
        self._lock = threading.Lock()
        self._requests: Dict[str, ApprovalRequest] = {}
        self._pending_by_fingerprint: Dict[str, str] = {}
        self._max_pending = max(1, int(max_pending))

    def request(self, action: Dict[str, Any], *, reason: str, actor: Optional[str], context: Dict[str, Any]) -> ApprovalRequest:
        fp = action_fingerprint(action)
        now = time.time()
        with self._lock:
            existing_id = self._pending_by_fingerprint.get(fp)
            if existing_id and existing_id in self._requests:
                return self._requests[existing_id]

            pending_count = sum(1 for req in self._requests.values() if req.status == "pending")
            if pending_count >= self._max_pending:
                # Evict oldest pending request (best-effort).
                oldest_id = None
                oldest_time = float("inf")
                for req_id, req in self._requests.items():
                    if req.status != "pending":
                        continue
                    if req.created_at < oldest_time:
                        oldest_time = req.created_at
                        oldest_id = req_id
                if oldest_id is not None:
                    old = self._requests.pop(oldest_id, None)
                    if old is not None:
                        self._pending_by_fingerprint.pop(old.fingerprint, None)

            req_id = uuid.uuid4().hex
            req = ApprovalRequest(
                id=req_id,
                created_at=now,
                fingerprint=fp,
                action=action,
                reason=str(reason or "approval_required"),
                actor=actor,
                context=dict(context or {}),
            )
            self._requests[req_id] = req
            self._pending_by_fingerprint[fp] = req_id
            return req

    def get(self, approval_id: str) -> Optional[ApprovalRequest]:
        with self._lock:
            return self._requests.get(str(approval_id))

    def list(self, *, status: Optional[str] = None, limit: int = 50) -> List[ApprovalRequest]:
        status = str(status).strip().lower() if status else None
        limit = max(0, int(limit))
        with self._lock:
            items = list(self._requests.values())
        if status:
            items = [req for req in items if req.status == status]
        items.sort(key=lambda req: req.created_at, reverse=True)
        return items[:limit] if limit else items

    def decide(
        self,
        approval_id: str,
        *,
        approve: bool,
        decided_by: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Optional[ApprovalRequest]:
        approval_id = str(approval_id)
        with self._lock:
            req = self._requests.get(approval_id)
            if req is None:
                return None
            if req.status != "pending":
                return req
            status = "approved" if approve else "denied"
            updated = ApprovalRequest(
                id=req.id,
                created_at=req.created_at,
                fingerprint=req.fingerprint,
                action=req.action,
                reason=req.reason,
                actor=req.actor,
                context=req.context,
                status=status,
                decided_at=time.time(),
                decided_by=decided_by,
                decision_note=note,
            )
            self._requests[approval_id] = updated
            if approve is False:
                self._pending_by_fingerprint.pop(req.fingerprint, None)
            elif approve is True:
                self._pending_by_fingerprint.pop(req.fingerprint, None)
            return updated


class AuditLogger:
    """Append-only JSONL audit logger."""

    def __init__(self, path: str, *, enabled: bool = True, max_field_chars: int = 2048) -> None:
        self.path = str(path)
        self.enabled = bool(enabled)
        self.max_field_chars = max(0, int(max_field_chars))
        self._lock = threading.Lock()

    def log(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = dict(event or {})
        payload.setdefault("t", time.time())
        sanitized = _redact_value(payload, max_chars=self.max_field_chars)
        line = _safe_json_dumps(sanitized)
        with self._lock:
            with open(self.path, "a", encoding="utf-8", errors="replace", newline="\n") as handle:
                handle.write(line + "\n")


@dataclass
class SecurityManagerConfig:
    """Configuration for :class:`SecurityManager`."""

    enabled: bool = True
    permission_level: PermissionLevel = PermissionLevel.USER

    allow_action_types: Tuple[str, ...] = ()
    deny_action_types: Tuple[str, ...] = ()

    deny_path_patterns: Tuple[str, ...] = ()
    deny_command_patterns: Tuple[str, ...] = ()

    require_approval_for: Tuple[str, ...] = DEFAULT_APPROVAL_REQUIRED
    confirm_token: Optional[str] = None  # bypass approval when provided per-action
    approval_token: Optional[str] = None  # required to approve/deny requests

    emergency_stop_env: str = "BSS_EMERGENCY_STOP"
    emergency_stop_allows_readonly: bool = True

    required_permission_by_action: Dict[str, PermissionLevel] = field(default_factory=lambda: dict(DEFAULT_REQUIRED_PERMISSION))

    audit_enabled: bool = False
    audit_log_path: Optional[str] = None
    max_audit_field_chars: int = 2048
    max_pending_approvals: int = 1000

    # Optional knowledge-graph constraint validation for actions.
    action_guard_enabled: bool = False


@dataclass(frozen=True)
class SecurityDecision:
    allowed: bool
    blocked: bool
    reason: str
    requires_approval: bool = False
    approval_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def as_info(self) -> Dict[str, Any]:
        info = {"blocked": bool(self.blocked), "reason": self.reason, **(self.details or {})}
        if self.requires_approval:
            info["requires_approval"] = True
        if self.approval_id:
            info["approval_id"] = self.approval_id
        return info


class SecurityManager:
    """Central evaluator for tool/UI/system actions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        env_confirm = os.environ.get("BSS_SECURITY_CONFIRM_TOKEN")
        env_approval = os.environ.get("BSS_SECURITY_APPROVAL_TOKEN")
        env_action_guard = os.environ.get("BSS_ACTION_GUARD_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}

        permission = PermissionLevel.from_value(cfg.get("permission_level", PermissionLevel.USER))
        required_permission_by_action = dict(DEFAULT_REQUIRED_PERMISSION)
        overrides = cfg.get("required_permission_by_action")
        if isinstance(overrides, dict):
            for key, val in overrides.items():
                required_permission_by_action[str(key)] = PermissionLevel.from_value(val)

        self.config = SecurityManagerConfig(
            enabled=bool(cfg.get("enabled", True)),
            permission_level=permission,
            allow_action_types=tuple(str(x) for x in (cfg.get("allow_action_types") or ())),
            deny_action_types=tuple(str(x) for x in (cfg.get("deny_action_types") or ())),
            deny_path_patterns=tuple(str(x) for x in (cfg.get("deny_path_patterns") or ())),
            deny_command_patterns=tuple(str(x) for x in (cfg.get("deny_command_patterns") or ())),
            require_approval_for=tuple(str(x) for x in (cfg.get("require_approval_for") or DEFAULT_APPROVAL_REQUIRED)),
            confirm_token=str(cfg.get("confirm_token") or env_confirm or "") or None,
            approval_token=str(cfg.get("approval_token") or env_approval or "") or None,
            emergency_stop_env=str(cfg.get("emergency_stop_env", "BSS_EMERGENCY_STOP") or "BSS_EMERGENCY_STOP"),
            emergency_stop_allows_readonly=bool(cfg.get("emergency_stop_allows_readonly", True)),
            required_permission_by_action=required_permission_by_action,
            audit_enabled=bool(cfg.get("audit_enabled", False)),
            audit_log_path=str(cfg.get("audit_log_path") or "") or None,
            max_audit_field_chars=int(cfg.get("max_audit_field_chars", 2048)),
            max_pending_approvals=int(cfg.get("max_pending_approvals", 1000)),
            action_guard_enabled=bool(cfg.get("action_guard_enabled", False)) or env_action_guard,
        )

        self._deny_path_res = _compile_patterns(self.config.deny_path_patterns)
        self._deny_cmd_res = _compile_patterns(self.config.deny_command_patterns)

        self.approvals = ApprovalQueue(max_pending=self.config.max_pending_approvals)

        self._action_guard = None
        if self.config.action_guard_enabled and ActionGuard is not None:
            try:
                self._action_guard = ActionGuard()
            except Exception:
                self._action_guard = None

        self.audit: Optional[AuditLogger] = None
        if self.config.audit_enabled and self.config.audit_log_path:
            self.audit = AuditLogger(
                self.config.audit_log_path,
                enabled=True,
                max_field_chars=self.config.max_audit_field_chars,
            )

    def snapshot(self) -> Dict[str, Any]:
        pending = len(self.approvals.list(status="pending", limit=0))
        return {
            "enabled": bool(self.config.enabled),
            "permission_level": self.config.permission_level.name.lower(),
            "pending_approvals": pending,
            "emergency_stop_env": self.config.emergency_stop_env,
            "emergency_stop_active": bool(os.environ.get(self.config.emergency_stop_env)),
            "action_guard_enabled": bool(self._action_guard is not None),
            "audit_enabled": bool(self.audit is not None and self.audit.enabled),
            "audit_log_path": self.config.audit_log_path,
        }

    def decide(
        self,
        action: Dict[str, Any],
        *,
        actor: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SecurityDecision:
        """Evaluate an action and return an allow/deny/approval decision."""

        context = dict(context or {})
        action = dict(action or {})
        action_type = str(action.get("type") or "").strip()
        if not action_type:
            return self._decision(False, True, "missing_action_type")

        if not self.config.enabled:
            return self._decision(True, False, "security_disabled")

        # Emergency stop kill switch: block everything except read-only actions (optional).
        if self.config.emergency_stop_env and os.environ.get(self.config.emergency_stop_env):
            if self.config.emergency_stop_allows_readonly and action_type in _READ_ONLY_ACTIONS:
                return self._decision(True, False, "emergency_stop_readonly_allowed")
            return self._decision(False, True, "emergency_stop_active", details={"kill_switch": self.config.emergency_stop_env})

        # Allow/deny lists for action types.
        if action_type in set(self.config.deny_action_types):
            return self._decision(False, True, "action_type_denied", details={"action_type": action_type})
        if self.config.allow_action_types and action_type not in set(self.config.allow_action_types):
            return self._decision(False, True, "action_type_not_allowed", details={"action_type": action_type})

        required = self.config.required_permission_by_action.get(action_type, PermissionLevel.ADMIN)
        if self.config.permission_level < required:
            return self._decision(
                False,
                True,
                "permission_denied",
                details={"action_type": action_type, "required": required.name.lower(), "current": self.config.permission_level.name.lower()},
            )

        # Deny patterns for paths/commands (best-effort; low-level bridges still enforce allowlists).
        path = action.get("path")
        if path is not None and self._deny_path_res:
            path_str = str(path)
            if any(rx.search(path_str) for rx in self._deny_path_res):
                return self._decision(False, True, "path_denied", details={"path": path_str})

        cmd_text = _action_command_text(action)
        if cmd_text and self._deny_cmd_res:
            if any(rx.search(cmd_text) for rx in self._deny_cmd_res):
                return self._decision(False, True, "command_denied", details={"command": cmd_text})

        # Knowledge-graph action constraints (optional).
        if self._action_guard is not None:
            try:
                args = {k: v for k, v in action.items() if k != "type"}
                guard_result = self._action_guard.evaluate(action_type, args, context=context)
                if getattr(guard_result, "allowed", True) is False:
                    return self._decision(
                        False,
                        True,
                        "action_guard_denied",
                        details={
                            "action_type": action_type,
                            "guard_reason": getattr(guard_result, "reason", ""),
                            "guard_violations": getattr(guard_result, "violations", []),
                        },
                    )
            except Exception:  # pragma: no cover - guard failures must not break execution
                pass

        # If an approval_id is provided, enforce that it's approved and matches the action fingerprint.
        approval_id = action.get("approval_id")
        if approval_id is not None:
            req = self.approvals.get(str(approval_id))
            if req is None:
                return self._decision(False, True, "unknown_approval_id", details={"approval_id": str(approval_id)})
            if req.status == "pending":
                return self._decision(False, True, "approval_pending", details={"approval_id": req.id})
            if req.status == "denied":
                return self._decision(False, True, "approval_denied", details={"approval_id": req.id})
            if action_fingerprint(action) != req.fingerprint:
                return self._decision(False, True, "approval_action_mismatch", details={"approval_id": req.id})
            return self._decision(True, False, "approval_granted", details={"approval_id": req.id})

        # Approval requirement for selected high-risk action types.
        requires_approval = action_type in set(self.config.require_approval_for)
        if requires_approval:
            confirm = self.config.confirm_token
            provided = action.get("confirm_token")
            if confirm and provided == confirm:
                return self._decision(True, False, "confirmed_by_token", details={"action_type": action_type})

            req = self.approvals.request(
                action,
                reason=f"approval_required:{action_type}",
                actor=actor,
                context=context,
            )
            return self._decision(
                False,
                True,
                "approval_required",
                requires_approval=True,
                approval_id=req.id,
                details={
                    "action_type": action_type,
                    "approval_id": req.id,
                    "approval_reason": req.reason,
                },
            )

        return self._decision(True, False, "allowed")

    def approve(
        self,
        approval_id: str,
        *,
        token: Optional[str] = None,
        decided_by: Optional[str] = None,
        note: Optional[str] = None,
    ) -> bool:
        if self.config.approval_token and token != self.config.approval_token:
            return False
        updated = self.approvals.decide(approval_id, approve=True, decided_by=decided_by, note=note)
        if updated is None:
            return False
        self._audit("approval_decision", {"approval_id": updated.id, "status": updated.status, "decided_by": decided_by, "note": note})
        return updated.status == "approved"

    def deny(
        self,
        approval_id: str,
        *,
        token: Optional[str] = None,
        decided_by: Optional[str] = None,
        note: Optional[str] = None,
    ) -> bool:
        if self.config.approval_token and token != self.config.approval_token:
            return False
        updated = self.approvals.decide(approval_id, approve=False, decided_by=decided_by, note=note)
        if updated is None:
            return False
        self._audit("approval_decision", {"approval_id": updated.id, "status": updated.status, "decided_by": decided_by, "note": note})
        return updated.status == "denied"

    # ------------------------------------------------------------------ #
    def _decision(
        self,
        allowed: bool,
        blocked: bool,
        reason: str,
        *,
        requires_approval: bool = False,
        approval_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> SecurityDecision:
        decision = SecurityDecision(
            allowed=bool(allowed),
            blocked=bool(blocked),
            reason=str(reason),
            requires_approval=bool(requires_approval),
            approval_id=str(approval_id) if approval_id else None,
            details=dict(details or {}),
        )
        self._audit(
            "decision",
            {
                "reason": decision.reason,
                "allowed": decision.allowed,
                "blocked": decision.blocked,
                "requires_approval": decision.requires_approval,
                "approval_id": decision.approval_id,
                "details": decision.details,
            },
        )
        return decision

    def _audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.audit is None:
            return
        try:
            self.audit.log({"event": str(event_type), **(payload or {})})
        except Exception:
            # Never fail the action path due to audit issues.
            return


def _action_command_text(action: Dict[str, Any]) -> str:
    for key in ("cmd", "command"):
        if key not in action:
            continue
        val = action.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            return " ".join(str(x) for x in val)
    return ""


__all__ = [
    "ApprovalQueue",
    "ApprovalRequest",
    "AuditLogger",
    "PermissionLevel",
    "SecurityDecision",
    "SecurityManager",
    "SecurityManagerConfig",
    "action_fingerprint",
    "redact_action",
]
