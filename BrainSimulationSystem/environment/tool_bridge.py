"""Safety-oriented tool/OS environment bridge.

This module provides a SimulationEnvironment implementation that turns tool
invocations (file inspection, limited shell commands) into environment steps,
so agents can learn/plan over tool usage with the same EnvironmentController
abstraction used for simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import sys
import time
import uuid
from pathlib import Path
import subprocess
from urllib.parse import urlparse
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - keep bridge usable without psutil
    psutil = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency (web fetch)
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency (docker control)
    import importlib

    import docker as docker  # type: ignore[no-redef]

    try:
        from docker.errors import DockerException  # type: ignore
    except Exception:  # pragma: no cover - defensive fallback
        DockerException = Exception  # type: ignore[misc,assignment]

    def _docker_module_is_local_stub(module: Any) -> bool:
        path = getattr(module, "__file__", None)
        if not path:
            return False
        try:
            mod_path = Path(str(path)).resolve()
        except Exception:
            return False
        repo_root = Path(__file__).resolve().parents[2]
        stub_root = (repo_root / "docker").resolve()
        return mod_path == stub_root / "__init__.py"

    if _docker_module_is_local_stub(docker):
        # Attempt to load the real Docker SDK from site-packages, even if a local
        # `docker/` stub package exists in the repo (which would otherwise shadow it).
        original_sys_path = list(sys.path)
        try:
            repo_root = Path(__file__).resolve().parents[2]
            filtered: List[str] = []
            for entry in sys.path:
                if entry in ("", "."):
                    continue
                try:
                    if Path(entry).resolve() == repo_root:
                        continue
                except Exception:
                    pass
                filtered.append(entry)
            sys.path = filtered

            for name in list(sys.modules):
                if name == "docker" or name.startswith("docker."):
                    sys.modules.pop(name, None)

            candidate = importlib.import_module("docker")
            if _docker_module_is_local_stub(candidate):
                raise ImportError("docker_sdk_shadowed_by_local_stub")
            docker = candidate  # type: ignore[assignment]
            try:
                DockerException = importlib.import_module("docker.errors").DockerException  # type: ignore[misc,assignment]
            except Exception:
                DockerException = Exception  # type: ignore[misc,assignment]
        except Exception:
            docker = None  # type: ignore[assignment]
            DockerException = Exception  # type: ignore[misc,assignment]
        finally:
            sys.path = original_sys_path
except Exception:  # pragma: no cover - keep bridge usable without docker-py
    docker = None  # type: ignore[assignment]
    DockerException = Exception  # type: ignore[misc,assignment]

try:  # pragma: no cover - optional dependency (network remote tool support)
    from .remote_tool import (
        RemoteToolAuthError,
        RemoteToolClient,
        RemoteToolConnectionError,
        RemoteToolError,
        RemoteToolProtocolError,
    )
except Exception:  # pragma: no cover - keep bridge usable without remote layer
    RemoteToolClient = None  # type: ignore[assignment]
    RemoteToolError = Exception  # type: ignore[misc,assignment]
    RemoteToolAuthError = Exception  # type: ignore[misc,assignment]
    RemoteToolConnectionError = Exception  # type: ignore[misc,assignment]
    RemoteToolProtocolError = Exception  # type: ignore[misc,assignment]

try:  # pragma: no cover - optional security layer
    from .security_manager import SecurityManager, redact_action
except Exception:  # pragma: no cover - keep bridge usable without security manager
    SecurityManager = None  # type: ignore[assignment]
    redact_action = None  # type: ignore[assignment]

try:  # pragma: no cover - optional filesystem sandbox
    from .filesystem_sandbox import FilesystemSandbox
except Exception:  # pragma: no cover - keep bridge usable without sandbox module
    FilesystemSandbox = None  # type: ignore[assignment]

from .code_safety import scan_script_file
from .license_detection import detect_repo_license, normalize_spdx_list


@dataclass(frozen=True)
class ToolStepResult:
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    info: Dict[str, Any]


class ToolEnvironmentBridge:
    """A minimal tool environment with a conservative safety model.

    Supported action payloads (dict):
      - {"type": "read_file", "path": "...", "max_chars": 4000}
      - {"type": "list_dir", "path": "...", "max_entries": 50}
      - {"type": "write_file", "path": "...", "text": "...", "append": false}
      - {"type": "create_file", "path": "...", "text": "...", "overwrite": false}
      - {"type": "modify_file", "path": "...", "operation": "replace", "old": "...", "new": "..."}
      - {"type": "modify_file", "path": "...", "operation": "regex_replace", "pattern": "...", "repl": "..."}
      - {"type": "delete_file", "path": "...", "missing_ok": false}
      - {"type": "create_dir", "path": "...", "parents": true, "exist_ok": true}
      - {"type": "web_search", "query": "...", "max_results": 5}
      - {"type": "web_scrape", "url": "...", "max_chars": 8000, "include_code": true}
      - {"type": "web_get", "url": "...", "max_chars": 8000}
      - {"type": "github_code_search", "query": "...", "max_results": 5}
      - {"type": "github_repo_ingest", "repo": "owner/repo", "ref": "main", "build_index": true}
      - {"type": "documentation_tool", "query": "...", "max_sources": 2}
      - {"type": "knowledge_import_directory", "path": "...", "source_name": "algorithms_pack", "include_suffixes": [".md", ".txt", ".py"]}
      - {"type": "knowledge_query", "query": "...", "top_k": 5, "include_metadata": true, "include_relations": false}
      - {"type": "parse_code", "path": "...", "language": "python", "include_docstrings": true, "max_chars": 120000}
      - {"type": "summarize_doc", "path": "...", "max_summary_chars": 900, "max_headings": 12}
      - {"type": "ask_human", "question": "...", "request_id": "...", "answer": "...", "dataset_path": "data/qa.jsonl"}
      - {"type": "code_index_build", "root": "...", "max_files": 800, "embedding_dimensions": 128}
      - {"type": "code_index_search", "root": "...", "query": "...", "top_k": 5}
      - {"type": "shell", "command": ["python", "-c", "print(1)"], "timeout_s": 10}
      - {"type": "launch_program", "command": ["python", "-m", "http.server"], "cwd": "..."}
      - {"type": "kill_process", "pid": 1234, "timeout_s": 3, "force": true}
      - {"type": "exec_system_cmd", "cmd": "netsh ...", "timeout_s": 10}
      - {"type": "exec_system_cmd", "command": ["netsh", "..."], "timeout_s": 10}
      - {"type": "change_system_setting", "name": "power.shutdown", "value": {"delay_s": 0}}
      - {"type": "docker", "action": "images.pull", "image": "python:3.12"}
      - {"type": "docker", "action": "containers.run", "image": "python:3.12", "command": ["python", "-c", "print(1)"], "detach": true}
      - {"type": "docker", "action": "containers.get", "container": "my_container"}
      - {"type": "docker", "action": "containers.stop", "container": "my_container"}
      - {"type": "docker_compose", "action": "up", "project_dir": "./deploy", "files": ["docker-compose.yml"], "detach": true}
      - {"type": "docker_compose", "action": "down", "project_dir": "./deploy", "files": ["docker-compose.yml"]}
      - {"type": "docker_compose", "action": "scale", "project_dir": "./deploy", "files": ["docker-compose.yml"], "services": {"web": 3}}
      - {"type": "run_script", "path": "./scripts/docker/restart_container.sh", "args": ["my_container"], "timeout_s": 60}
      - {"type": "remote_tool", "endpoint": "host:port", "action": {"type": "shell", ...}}
      - {"type": "sandbox_status"}
      - {"type": "sandbox_commit", "confirm_token": "..."}
      - {"type": "sandbox_reset", "confirm_token": "..."}
      - {"type": "terminate"}

    By default shell execution is disabled unless ``allowed_shell_prefixes`` is
    provided, and filesystem operations are restricted to ``allowed_roots``.
    Write/delete and process actions are disabled unless explicitly enabled.
    """

    def __init__(
        self,
        *,
        allowed_roots: Iterable[str | os.PathLike[str]] | None = None,
        allowed_shell_prefixes: Iterable[List[str]] | None = None,
        allowed_program_prefixes: Iterable[List[str]] | None = None,
        allowed_system_cmd_prefixes: Iterable[List[str]] | None = None,
        default_timeout_s: float = 10.0,
        allow_file_write: bool = False,
        allow_file_delete: bool = False,
        max_write_chars: int = 200_000,
        allow_process_control: bool = False,
        allow_kill_untracked: bool = False,
        cleanup_processes_on_close: bool = True,
        allow_system_cmd: bool = False,
        allow_high_risk_system_cmd: bool = False,
        system_confirm_token: Optional[str] = None,
        max_system_cmd_output_chars: int = 16_000,
        allow_docker_control: bool = False,
        allowed_docker_image_prefixes: Iterable[str] | None = None,
        allow_docker_delete: bool = False,
        allow_docker_untracked: bool = False,
        cleanup_containers_on_close: bool = True,
        max_docker_output_chars: int = 16_000,
        allow_docker_compose: bool = False,
        allow_docker_compose_delete: bool = False,
        allowed_docker_compose_binaries: Iterable[str] | None = None,
        allow_script_execution: bool = False,
        allowed_script_paths: Iterable[str | os.PathLike[str]] | None = None,
        max_script_output_chars: int = 16_000,
        allow_remote_control: bool = False,
        allowed_remote_endpoints: Iterable[str] | None = None,
        allowed_remote_action_types: Iterable[str] | None = None,
        remote_auth_token: Optional[str] = None,
        remote_default_timeout_s: float = 10.0,
        allow_web_access: bool = False,
        allowed_web_domains: Iterable[str] | None = None,
        max_web_output_chars: int = 24_000,
        max_code_index_output_chars: int = 24_000,
        prefer_real_web_search: bool = True,
        security_manager: Optional["SecurityManager"] = None,
        filesystem_sandbox: Optional[Dict[str, Any]] = None,
        allow_sandbox_commit: bool = False,
        sandbox_confirm_token: Optional[str] = None,
        source_policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        if allowed_roots is None:
            allowed_roots = [Path.cwd()]
        self._allowed_roots = [Path(root).resolve() for root in allowed_roots]
        self._allowed_shell_prefixes = [list(prefix) for prefix in (allowed_shell_prefixes or [])]
        self._allowed_program_prefixes = [list(prefix) for prefix in (allowed_program_prefixes or [])]
        self._allowed_system_cmd_prefixes = [list(prefix) for prefix in (allowed_system_cmd_prefixes or [])]
        self._default_timeout_s = float(default_timeout_s)
        self._allow_file_write = bool(allow_file_write)
        self._allow_file_delete = bool(allow_file_delete)
        self._max_write_chars = max(0, int(max_write_chars))
        self._allow_process_control = bool(allow_process_control)
        self._allow_kill_untracked = bool(allow_kill_untracked)
        self._cleanup_processes_on_close = bool(cleanup_processes_on_close)
        self._allow_system_cmd = bool(allow_system_cmd)
        self._allow_high_risk_system_cmd = bool(allow_high_risk_system_cmd)
        self._system_confirm_token = system_confirm_token or os.getenv("BSS_SYSTEM_CONFIRM_TOKEN")
        self._max_system_cmd_output_chars = max(0, int(max_system_cmd_output_chars))
        self._allow_docker_control = bool(allow_docker_control)
        self._allow_docker_delete = bool(allow_docker_delete)
        self._allow_docker_untracked = bool(allow_docker_untracked)
        self._cleanup_containers_on_close = bool(cleanup_containers_on_close)
        self._max_docker_output_chars = max(0, int(max_docker_output_chars))
        self._allow_docker_compose = bool(allow_docker_compose)
        self._allow_docker_compose_delete = bool(allow_docker_compose_delete)
        self._allowed_docker_compose_binaries = tuple(
            b
            for b in (
                self._normalize_docker_compose_binary_name(value)
                for value in (allowed_docker_compose_binaries or ("docker", "docker-compose"))
            )
            if b
        )
        self._allow_script_execution = bool(allow_script_execution)
        self._max_script_output_chars = max(0, int(max_script_output_chars))
        self._allowed_script_paths: set[Path] = set()
        for raw_path in allowed_script_paths or []:
            if not raw_path:
                continue
            try:
                resolved = Path(raw_path).resolve()
            except Exception:
                continue
            self._allowed_script_paths.add(resolved)
        self._allowed_docker_image_prefixes = tuple(
            str(prefix).strip()
            for prefix in (allowed_docker_image_prefixes or [])
            if str(prefix).strip()
        )
        self._docker_tracked_container_ids: set[str] = set()
        self._docker_tracked_container_names: set[str] = set()
        self._allow_remote_control = bool(allow_remote_control)
        self._remote_auth_token = remote_auth_token or os.getenv("BSS_REMOTE_AUTH_TOKEN")
        self._remote_default_timeout_s = float(remote_default_timeout_s)
        env_web = str(os.getenv("BSS_WEB_ENABLED") or "").strip().lower()
        self._allow_web_access = bool(allow_web_access) or env_web in {"1", "true", "yes", "on"}
        self._prefer_real_web_search = bool(prefer_real_web_search)
        self._max_web_output_chars = max(0, int(max_web_output_chars))
        self._max_code_index_output_chars = max(0, int(max_code_index_output_chars))
        self._allowed_web_domains = tuple(
            str(domain).strip().lower()
            for domain in (allowed_web_domains or [])
            if str(domain).strip()
        )
        source_cfg = source_policy if isinstance(source_policy, dict) else {}
        env_trusted = os.getenv("BSS_TRUSTED_WEB_DOMAINS")
        env_blocked = os.getenv("BSS_BLOCKED_WEB_DOMAINS")

        def _parse_domains(value: Any) -> tuple[str, ...]:
            if value is None:
                return ()
            if isinstance(value, str):
                raw = value.replace(";", ",")
                parts = [p.strip().lower().strip(".") for p in raw.split(",")]
                return tuple(p for p in parts if p)
            if isinstance(value, (list, tuple, set)):
                parts = [str(p).strip().lower().strip(".") for p in value if str(p).strip()]
                return tuple(dict.fromkeys(parts))
            raw = str(value).strip().lower().strip(".")
            return (raw,) if raw else ()

        self._trusted_web_domains = _parse_domains(
            source_cfg.get("trusted_web_domains") or source_cfg.get("trusted_domains") or env_trusted
        )
        self._blocked_web_domains = _parse_domains(
            source_cfg.get("blocked_web_domains") or source_cfg.get("blocked_domains") or env_blocked
        )
        self._web_domain_reputation: Dict[str, float] = {}
        rep = source_cfg.get("web_domain_reputation") or source_cfg.get("domain_reputation")
        if isinstance(rep, dict):
            for key, val in rep.items():
                dom = str(key or "").strip().lower().strip(".")
                if not dom:
                    continue
                try:
                    score = float(val)
                except Exception:
                    label = str(val or "").strip().lower()
                    score = 1.0 if label in {"high", "trusted"} else 0.2 if label in {"low", "untrusted"} else 0.5
                self._web_domain_reputation[dom] = max(0.0, min(1.0, score))

        env_license_enforce = str(os.getenv("BSS_LICENSE_ENFORCE") or "").strip().lower()
        env_license_require = str(os.getenv("BSS_LICENSE_REQUIRE") or "").strip().lower()
        env_license_allow = os.getenv("BSS_LICENSE_ALLOWLIST")
        env_license_deny = os.getenv("BSS_LICENSE_DENYLIST")

        self._license_policy = {
            "enforce": bool(source_cfg.get("license_enforce", False))
            or env_license_enforce in {"1", "true", "yes", "on"},
            "require": bool(source_cfg.get("license_require", False))
            or env_license_require in {"1", "true", "yes", "on"},
            "allow_unknown": bool(source_cfg.get("license_allow_unknown", True)),
            "allowlist": normalize_spdx_list(source_cfg.get("license_allowlist") or env_license_allow),
            "denylist": normalize_spdx_list(source_cfg.get("license_denylist") or env_license_deny),
        }
        self._code_indexes: Dict[str, Any] = {}
        self._code_index_metadata: Dict[str, Dict[str, Any]] = {}
        self._allowed_remote_endpoints = {
            self._normalize_remote_endpoint_value(value)
            for value in (allowed_remote_endpoints or [])
            if self._normalize_remote_endpoint_value(value)
        }
        if allowed_remote_action_types is None:
            self._allowed_remote_action_types: Optional[set[str]] = {"read_file", "list_dir", "shell"}
        else:
            normalized = {str(t).strip().lower() for t in allowed_remote_action_types if str(t).strip()}
            self._allowed_remote_action_types = normalized or None
        self._remote_clients: Dict[Tuple[str, int], Any] = {}
        self._security_manager = security_manager
        self._fs_sandbox = None
        self._fs_sandbox_status: Dict[str, Any] = {"enabled": False}
        self._allow_sandbox_commit = bool(allow_sandbox_commit)
        self._sandbox_confirm_token = sandbox_confirm_token or os.getenv("BSS_SANDBOX_CONFIRM_TOKEN")

        sandbox_cfg = filesystem_sandbox if isinstance(filesystem_sandbox, dict) else {}
        self._sandbox_keep_history = bool(sandbox_cfg.get("keep_history", True))
        sandbox_enabled = bool(sandbox_cfg.get("enabled", False) or sandbox_cfg.get("root"))
        if sandbox_enabled and FilesystemSandbox is not None:
            try:
                raw_root = sandbox_cfg.get("root")
                if not raw_root:
                    base_root = self._allowed_roots[0] if self._allowed_roots else Path.cwd().resolve()
                    raw_root = base_root / ".bss_sandbox"
                sandbox_root = Path(str(raw_root))
                if not self._is_allowed_path(sandbox_root):
                    raise ValueError("sandbox_root_not_allowed")
                self._fs_sandbox = FilesystemSandbox(sandbox_root, allowed_roots=self._allowed_roots)
                self._fs_sandbox_status = {"enabled": True, "root": str(self._fs_sandbox.root)}
            except Exception as exc:
                self._fs_sandbox = None
                self._fs_sandbox_status = {"enabled": False, "error": str(exc)}
        self._processes: Dict[int, subprocess.Popen] = {}
        self._initialized = False
        self._steps = 0
        self._web_disabled_until: float = 0.0
        self._web_disabled_reason: str | None = None
        self._runtime_blocked_web_domains: Dict[str, Dict[str, Any]] = {}

    def initialize(self) -> None:
        self._initialized = True

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()
        self._steps = 0
        return {"text": "", "tool_state": {"steps": 0}}

    def capability_snapshot(self) -> Dict[str, Any]:
        """Return a prompt-friendly snapshot of enabled tool capabilities.

        This is intended for agent introspection / logging. It avoids returning
        any secret tokens (e.g., confirm/auth tokens).
        """

        shell_enabled = bool(self._allowed_shell_prefixes)
        program_enabled = bool(self._allow_process_control and self._allowed_program_prefixes)
        system_cmd_enabled = bool(self._allow_system_cmd and self._allowed_system_cmd_prefixes)
        sandbox_enabled = bool(self._fs_sandbox_status.get("enabled")) if isinstance(self._fs_sandbox_status, dict) else False

        web_enabled = bool(self._web_is_enabled())
        enabled_actions: Dict[str, bool] = {
            "read_file": True,
            "list_dir": True,
            "shell": shell_enabled,
            "write_file": bool(self._allow_file_write),
            "create_file": bool(self._allow_file_write),
            "modify_file": bool(self._allow_file_write),
            "create_dir": bool(self._allow_file_write),
            "delete_file": bool(self._allow_file_delete),
            "web_search": web_enabled,
            "web_scrape": web_enabled,
            "web_get": web_enabled,
            "github_code_search": web_enabled,
            "github_repo_ingest": bool(web_enabled and self._allow_file_write),
            "documentation_tool": web_enabled,
            "knowledge_import_directory": True,
            "knowledge_query": True,
            "parse_code": True,
            "summarize_doc": True,
            "ask_human": True,
            "code_index_build": True,
            "code_index_search": True,
            "launch_program": program_enabled,
            "kill_process": bool(self._allow_process_control),
            "exec_system_cmd": system_cmd_enabled,
            "change_system_setting": system_cmd_enabled,
            "docker": bool(self._allow_docker_control),
            "docker_compose": bool(self._allow_docker_compose),
            "run_script": bool(self._allow_script_execution),
            "remote_tool": bool(self._allow_remote_control),
            "sandbox_status": sandbox_enabled,
            "sandbox_commit": bool(sandbox_enabled and self._allow_sandbox_commit),
            "sandbox_reset": sandbox_enabled,
        }

        security_snapshot: Dict[str, Any] = {}
        if self._security_manager is not None:
            snap = getattr(self._security_manager, "snapshot", None)
            if callable(snap):
                try:
                    security_snapshot = dict(snap())
                except Exception:
                    security_snapshot = {}

        allowed_roots = [str(root) for root in self._allowed_roots]
        script_paths = sorted(str(path) for path in self._allowed_script_paths) if self._allowed_script_paths else []
        remote_endpoints = sorted(self._allowed_remote_endpoints) if self._allowed_remote_endpoints else []
        remote_action_types = None
        if isinstance(self._allowed_remote_action_types, set):
            remote_action_types = sorted(self._allowed_remote_action_types)

        return {
            "enabled_actions": enabled_actions,
            "constraints": {
                "allowed_roots": allowed_roots,
                "default_timeout_s": float(self._default_timeout_s),
                "max_write_chars": int(self._max_write_chars),
                "max_system_cmd_output_chars": int(self._max_system_cmd_output_chars),
                "max_docker_output_chars": int(self._max_docker_output_chars),
                "max_script_output_chars": int(self._max_script_output_chars),
                "max_web_output_chars": int(self._max_web_output_chars),
                "max_code_index_output_chars": int(self._max_code_index_output_chars),
                "allowed_shell_prefixes": list(self._allowed_shell_prefixes),
                "allowed_program_prefixes": list(self._allowed_program_prefixes),
                "allowed_system_cmd_prefixes": list(self._allowed_system_cmd_prefixes),
                "allowed_docker_image_prefixes": list(self._allowed_docker_image_prefixes),
                "allowed_docker_compose_binaries": list(self._allowed_docker_compose_binaries),
                "allowed_script_paths": script_paths,
                "allowed_web_domains": list(self._allowed_web_domains),
                "trusted_web_domains": list(self._trusted_web_domains),
                "blocked_web_domains": list(self._blocked_web_domains),
                "runtime_blocked_web_domains": sorted(list((self._runtime_blocked_web_domains or {}).keys())),
                "web_quarantine": {
                    "disabled_until": float(self._web_disabled_until) if self._web_disabled_until else None,
                    "reason": self._web_disabled_reason,
                },
                "license_policy": {
                    "enforce": bool(self._license_policy.get("enforce")),
                    "require": bool(self._license_policy.get("require")),
                    "allow_unknown": bool(self._license_policy.get("allow_unknown", True)),
                    "allowlist": sorted(list(self._license_policy.get("allowlist") or set())),
                    "denylist": sorted(list(self._license_policy.get("denylist") or set())),
                },
                "prefer_real_web_search": bool(self._prefer_real_web_search),
                "allowed_remote_endpoints": remote_endpoints,
                "allowed_remote_action_types": remote_action_types,
                "remote_default_timeout_s": float(self._remote_default_timeout_s),
            },
            "sandbox": dict(self._fs_sandbox_status or {}),
            "security": security_snapshot,
        }

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._initialized:
            self.initialize()

        self._steps += 1
        if not isinstance(action, dict):
            result = ToolStepResult(
                observation={"text": ""},
                reward=-1.0,
                terminated=False,
                info={"error": "action_must_be_dict", "action_type": type(action).__name__},
            )
            return result.observation, result.reward, result.terminated, result.info

        action_type = action.get("type")
        if action_type == "terminate":
            result = ToolStepResult(
                observation={"text": "terminated", "tool_state": {"steps": self._steps}},
                reward=0.0,
                terminated=True,
                info={"terminated": True},
            )
            return result.observation, result.reward, result.terminated, result.info

        # Global emergency stop: block all non-read-only actions.
        if os.environ.get("BSS_EMERGENCY_STOP"):
            if action_type not in {
                "read_file",
                "list_dir",
                "sandbox_status",
                "code_index_build",
                "code_index_search",
                "knowledge_import_directory",
                "knowledge_query",
                "parse_code",
                "summarize_doc",
                "ask_human",
                "terminate",
            }:
                result = ToolStepResult(
                    observation={"text": "", "tool_state": self._tool_state()},
                    reward=-1.0,
                    terminated=False,
                    info={
                        "blocked": True,
                        "reason": "emergency_stop_active",
                        "kill_switch": "BSS_EMERGENCY_STOP",
                        "action_type": action_type,
                    },
                )
                return result.observation, result.reward, result.terminated, result.info

        if self._security_manager is not None:
            try:
                decision = self._security_manager.decide(action, context={"tool_state": self._tool_state()})
            except Exception as exc:  # pragma: no cover - defensive
                result = ToolStepResult(
                    observation={"text": "", "tool_state": self._tool_state()},
                    reward=-1.0,
                    terminated=False,
                    info={"blocked": True, "reason": "security_manager_failed", "exception": repr(exc)},
                )
                self._audit_action("blocked", action, result.observation, result.reward, result.terminated, result.info)
                return result.observation, result.reward, result.terminated, result.info
            if decision.blocked:
                result = ToolStepResult(
                    observation={"text": "", "tool_state": self._tool_state()},
                    reward=-1.0,
                    terminated=False,
                    info=decision.as_info(),
                )
                self._audit_action("blocked", action, result.observation, result.reward, result.terminated, result.info)
                return result.observation, result.reward, result.terminated, result.info

        handlers = {
            "read_file": self._handle_read_file,
            "list_dir": self._handle_list_dir,
            "write_file": self._handle_write_file,
            "create_file": self._handle_create_file,
            "modify_file": self._handle_modify_file,
            "delete_file": self._handle_delete_file,
            "create_dir": self._handle_create_dir,
            "web_search": self._handle_web_search,
            "web_scrape": self._handle_web_scrape,
            "web_get": self._handle_web_get,
            "github_code_search": self._handle_github_code_search,
            "github_repo_ingest": self._handle_github_repo_ingest,
            "documentation_tool": self._handle_documentation_tool,
            "knowledge_import_directory": self._handle_knowledge_import_directory,
            "knowledge_query": self._handle_knowledge_query,
            "parse_code": self._handle_parse_code,
            "summarize_doc": self._handle_summarize_doc,
            "ask_human": self._handle_ask_human,
            "code_index_build": self._handle_code_index_build,
            "code_index_search": self._handle_code_index_search,
            "shell": self._handle_shell,
            "launch_program": self._handle_launch_program,
            "kill_process": self._handle_kill_process,
            "exec_system_cmd": self._handle_exec_system_cmd,
            "change_system_setting": self._handle_change_system_setting,
            "docker": self._handle_docker,
            "docker_compose": self._handle_docker_compose,
            "run_script": self._handle_run_script,
            "remote_tool": self._handle_remote_tool,
            "sandbox_status": self._handle_sandbox_status,
            "sandbox_commit": self._handle_sandbox_commit,
            "sandbox_reset": self._handle_sandbox_reset,
        }

        handler = handlers.get(action_type)
        if handler is None:
            result = ToolStepResult(
                observation={"text": ""},
                reward=-1.0,
                terminated=False,
                info={"error": "unknown_action_type", "action_type": action_type},
            )
            observation, reward, terminated, info = result.observation, result.reward, result.terminated, result.info
        else:
            observation, reward, terminated, info = handler(action)

        self._audit_action("result", action, observation, reward, terminated, info)
        return observation, reward, terminated, info

    def close(self) -> None:
        if self._cleanup_processes_on_close and self._processes:
            for pid, proc in list(self._processes.items()):
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=1.0)
                        except Exception:
                            proc.kill()
                except Exception:
                    pass
                finally:
                    self._processes.pop(pid, None)

        if self._cleanup_containers_on_close and self._docker_tracked_container_ids:
            self._cleanup_docker_containers()
        self._initialized = False

    # ------------------------------------------------------------------ knowledge graph tools
    def _handle_knowledge_import_directory(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raw_path = action.get("path") or action.get("root") or action.get("directory")
        if not raw_path:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_path"}

        root = Path(str(raw_path))
        if not self._is_allowed_path(root):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "path_not_allowed", "path": str(raw_path)}
        if root.exists() and root.is_file():
            root = root.parent
        if not root.exists() or not root.is_dir():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "root_not_dir", "path": str(root)}

        source_name = str(action.get("source_name") or action.get("source") or "external_corpus").strip() or "external_corpus"

        include_suffixes = action.get("include_suffixes") or action.get("suffixes") or (".md", ".txt", ".rst", ".py")
        if isinstance(include_suffixes, str):
            include_suffixes = [include_suffixes]
        if not isinstance(include_suffixes, (list, tuple)):
            include_suffixes = (".md", ".txt", ".rst", ".py")
        suffixes = tuple(
            s if s.startswith(".") else f".{s}"
            for s in (str(item).strip().lower() for item in include_suffixes)
            if s
        ) or (".md", ".txt", ".rst", ".py")

        exclude_dirs = action.get("exclude_dirs")
        if isinstance(exclude_dirs, str):
            exclude_dirs = [exclude_dirs]
        if not isinstance(exclude_dirs, (list, tuple)):
            exclude_dirs = None

        max_files = max(1, min(int(action.get("max_files", 800)), 50_000))
        max_chars_per_file = max(200, min(int(action.get("max_chars_per_file", 12_000)), 400_000))
        description_chars = max(200, min(int(action.get("description_chars", 2400)), 24_000))
        create_group_node = bool(action.get("create_group_node", True))
        relation_label = str(action.get("relation_label") or action.get("relation") or "contains").strip() or "contains"

        try:
            from modules.knowledge.external_corpus_importer import ExternalCorpusImportConfig, import_external_corpus
        except Exception as exc:  # pragma: no cover - defensive
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "knowledge_import_unavailable", "exception": repr(exc)}

        try:
            cfg = ExternalCorpusImportConfig(
                source_name=source_name,
                include_suffixes=suffixes,
                exclude_dirs=exclude_dirs or (
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    ".venv",
                    "node_modules",
                    ".bss_sandbox",
                ),
                max_files=max_files,
                max_chars_per_file=max_chars_per_file,
                description_chars=description_chars,
                create_group_node=create_group_node,
                relation_label=relation_label,
            )
            result = import_external_corpus(root, config=cfg)
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "knowledge_import_failed", "exception": repr(exc)}

        summary = (
            f"knowledge_import_directory: source={source_name} root={str(root.resolve())} "
            f"files={result.get('processed_files')} skipped={result.get('skipped_files')} "
            f"nodes_added={result.get('import_result', {}).get('nodes_added')}"
        )
        text, truncated = self._clip_code_index_text(summary)
        info = {"root": str(root.resolve()), "source_name": source_name, "truncated": truncated, **dict(result or {})}
        obs = {"text": text, "result": result, "tool_state": self._tool_state()}
        return obs, 0.12, False, info

    def _handle_knowledge_query(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        query = str(action.get("query") or action.get("text") or "").strip()
        if not query:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_query"}

        top_k = max(1, min(int(action.get("top_k", 5)), 50))
        vector_type = str(action.get("vector_type") or "text").strip() or "text"
        include_metadata = bool(action.get("include_metadata", True))
        include_relations = bool(action.get("include_relations", False))

        try:
            from backend.knowledge.registry import require_default_aligner, get_graph_store_instance
        except Exception as exc:  # pragma: no cover - defensive
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "knowledge_query_unavailable", "exception": repr(exc)}

        try:
            aligner = require_default_aligner()
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "knowledge_aligner_unavailable", "exception": str(exc)}

        graph = get_graph_store_instance()

        expected_dim: int | None = None
        for node in getattr(aligner, "entities", {}).values():
            emb = getattr(node, "modalities", {}).get(vector_type)
            if isinstance(emb, list) and emb:
                expected_dim = len(emb)
                break

        embedding: List[float]
        if expected_dim is not None and expected_dim <= 32:
            # Likely hash-based embeddings; keep query in the same space.
            embedding = self._hash_embedding(query, dimensions=expected_dim)
        else:
            sentence_model = getattr(self, "_knowledge_query_sentence_model", None)
            if sentence_model is None:
                try:  # pragma: no cover - optional dependency
                    from sentence_transformers import SentenceTransformer  # type: ignore
                except Exception:
                    SentenceTransformer = None  # type: ignore
                if SentenceTransformer is not None:
                    try:
                        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                    except Exception:
                        sentence_model = None
                setattr(self, "_knowledge_query_sentence_model", sentence_model)

            if sentence_model is not None:
                try:
                    vector = sentence_model.encode(query, convert_to_numpy=True)
                    embedding = vector.astype(float).tolist()
                except Exception:
                    embedding = self._hash_embedding(query, dimensions=expected_dim or 12)
            else:
                embedding = self._hash_embedding(query, dimensions=expected_dim or 12)
        scored: List[Tuple[Any, float]] = []
        for node in getattr(aligner, "entities", {}).values():
            emb = getattr(node, "modalities", {}).get(vector_type)
            if not emb:
                continue
            try:
                score = float(getattr(aligner, "_cosine_similarity")(embedding, emb))
            except Exception:
                score = 0.0
            scored.append((node, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        ranked = scored[:top_k]

        results: List[Dict[str, Any]] = []
        lines: List[str] = []
        references: List[Dict[str, Any]] = []
        for node, score in ranked:
            meta = dict(getattr(node, "metadata", {}) or {})
            entry: Dict[str, Any] = {
                "id": getattr(node, "id", ""),
                "label": getattr(node, "label", ""),
                "similarity": round(float(score), 4),
            }
            if include_metadata:
                entry["metadata"] = meta
            if include_relations:
                try:
                    snapshot = graph.query(node_id=str(entry["id"]))
                    entry["relations"] = [
                        {
                            "source": getattr(edge, "source", ""),
                            "target": getattr(edge, "target", ""),
                            "type": getattr(getattr(edge, "type", None), "value", None) or str(getattr(edge, "type", "")),
                            "properties": dict(getattr(edge, "properties", {}) or {}),
                        }
                        for edge in (snapshot.get("edges") or [])
                    ]
                except Exception:
                    entry["relations"] = []

            results.append(entry)
            label = str(entry.get("label") or entry.get("id") or "").strip()
            lines.append(f"- {label} (sim={float(score):.3f})")

            rel_path = meta.get("relative_path") or meta.get("path")
            if isinstance(rel_path, str) and rel_path.strip():
                references.append({"url": rel_path, "title": label, "source": "knowledge_query"})

        text, truncated = self._clip_code_index_text("knowledge_query hits:\n" + "\n".join(lines))
        info = {"query": query, "returned": len(results), "truncated": truncated}
        obs = {"text": text, "results": results, "references": references[:top_k], "tool_state": self._tool_state()}
        return obs, 0.12, False, info

    # ------------------------------------------------------------------ human interaction (dev/training)
    def _handle_ask_human(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        request_id = str(action.get("request_id") or action.get("id") or "").strip() or None
        question = str(action.get("question") or action.get("query") or action.get("text") or "").strip()
        answer = action.get("answer") if "answer" in action else action.get("response")
        answer_text = str(answer).strip() if answer is not None else ""

        max_question_chars = max(120, min(int(action.get("max_question_chars", 1800)), 20_000))
        max_answer_chars = max(120, min(int(action.get("max_answer_chars", 4000)), 50_000))

        dataset_path_raw = action.get("dataset_path") or action.get("qa_dataset") or None
        simulate_only = bool(action.get("simulate_only", False))
        max_records = max(10, min(int(action.get("max_records", 2500)), 200_000))
        max_dataset_bytes = max(50_000, min(int(action.get("max_dataset_bytes", 8_000_000)), 300_000_000))
        min_score = float(action.get("min_score", 0.18))

        # Internal store (per ToolEnvironmentBridge instance).
        store = getattr(self, "_human_requests", None)
        if store is None or not isinstance(store, dict):
            store = {}
            setattr(self, "_human_requests", store)

        def _publish(event_type: str, payload: Dict[str, Any], *, summary: str, importance: float) -> None:
            try:
                from backend.monitoring.global_workspace import WorkspaceMessage, global_workspace
            except Exception:
                return
            try:
                global_workspace.publish_message(
                    WorkspaceMessage(
                        type=event_type,
                        source="tool_bridge",
                        payload=dict(payload),
                        summary=summary,
                        tags=("human",),
                        importance=float(importance),
                    ),
                    propagate=True,
                )
            except Exception:
                return

        def _tokens(text: str) -> set[str]:
            token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")
            return {tok.lower() for tok in token_re.findall(str(text or ""))}

        def _simulate_from_dataset(question_text: str, dataset_path: Path) -> Optional[Dict[str, Any]]:
            if not dataset_path.exists() or not dataset_path.is_file():
                return None
            query_tokens = _tokens(question_text)
            if not query_tokens:
                return None
            best: Optional[Dict[str, Any]] = None
            best_score = 0.0
            bytes_read = 0
            records = 0
            try:
                with dataset_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        if records >= max_records or bytes_read >= max_dataset_bytes:
                            break
                        bytes_read += len(line.encode("utf-8", errors="ignore"))
                        records += 1
                        raw = line.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        q = str(obj.get("question") or obj.get("q") or obj.get("title") or "").strip()
                        a = str(obj.get("answer") or obj.get("a") or obj.get("response") or obj.get("accepted_answer") or "").strip()
                        if not q or not a:
                            continue
                        q_tokens = _tokens(q)
                        if not q_tokens:
                            continue
                        overlap = len(query_tokens & q_tokens)
                        if overlap <= 0:
                            continue
                        denom = (len(query_tokens) * len(q_tokens)) ** 0.5 or 1.0
                        score = float(overlap) / float(denom)
                        if score > best_score:
                            best_score = score
                            best = {"question": q, "answer": a, "score": round(score, 4)}
            except Exception:
                return None
            if best is None or best_score < min_score:
                return None
            best["records_scanned"] = int(records)
            best["bytes_read"] = int(bytes_read)
            return best

        # 1) Record a response for an existing request.
        if answer_text:
            if request_id is None:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_request_id_for_answer"}
            entry = store.get(request_id) if isinstance(store, dict) else None
            if not isinstance(entry, dict):
                entry = {"id": request_id, "question": question, "created_at": time.time()}
                store[request_id] = entry
            entry["answer"] = answer_text[:max_answer_chars]
            entry["answered_at"] = time.time()
            entry["status"] = "answered"

            _publish(
                "human.response",
                {"request_id": request_id, "question": entry.get("question", ""), "answer": entry.get("answer", "")},
                summary=f"human response: {request_id}",
                importance=0.7,
            )

            text, truncated = self._clip_code_index_text(f"ask_human answered ({request_id}):\n{entry['answer']}")
            info = {"request_id": request_id, "status": "answered", "truncated": truncated}
            obs = {"text": text, "request": dict(entry), "tool_state": self._tool_state()}
            return obs, 0.12, False, info

        # 2) Poll for an existing request.
        if request_id is not None and not question:
            entry = store.get(request_id) if isinstance(store, dict) else None
            if not isinstance(entry, dict):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "unknown_request_id", "request_id": request_id}
            if entry.get("status") == "answered" and entry.get("answer"):
                text, truncated = self._clip_code_index_text(f"ask_human answer ({request_id}):\n{entry.get('answer')}")
                info = {"request_id": request_id, "status": "answered", "truncated": truncated}
                obs = {"text": text, "request": dict(entry), "tool_state": self._tool_state()}
                return obs, 0.12, False, info

            text, truncated = self._clip_code_index_text(f"ask_human pending ({request_id})")
            info = {"request_id": request_id, "status": "pending", "truncated": truncated, "requires_human": True}
            obs = {"text": text, "request": dict(entry), "tool_state": self._tool_state()}
            return obs, 0.08, False, info

        # 3) New question required to either simulate or create a request.
        if not question:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_question"}

        clipped_question = question[:max_question_chars]

        # 3a) Optional offline simulation (e.g., StackOverflow-style JSONL dataset).
        if dataset_path_raw:
            dataset_path = Path(str(dataset_path_raw))
            if not self._is_allowed_path(dataset_path):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "path_not_allowed", "path": str(dataset_path_raw)}
            hit = _simulate_from_dataset(clipped_question, dataset_path)
            if hit is not None:
                _publish(
                    "human.simulated",
                    {"question": clipped_question, "answer": hit.get("answer", ""), "score": hit.get("score")},
                    summary="human simulated answer",
                    importance=0.55,
                )
                text, truncated = self._clip_code_index_text(
                    f"ask_human simulated answer (score={hit.get('score')}):\n{str(hit.get('answer') or '')[:max_answer_chars]}"
                )
                info = {"status": "answered", "simulated": True, "score": hit.get("score"), "dataset_path": str(dataset_path), "truncated": truncated}
                obs = {"text": text, "answer": hit.get("answer", ""), "match": hit, "tool_state": self._tool_state()}
                return obs, 0.12, False, info
            if simulate_only:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "no_simulated_match", "dataset_path": str(dataset_path)}

        # 3b) Create a new human request (asynchronous handshake).
        request_id = request_id or uuid.uuid4().hex
        entry = {
            "id": request_id,
            "question": clipped_question,
            "created_at": time.time(),
            "status": "pending",
        }
        store[request_id] = entry

        _publish(
            "human.request",
            {"request_id": request_id, "question": clipped_question},
            summary=f"human request: {request_id}",
            importance=0.6,
        )

        text, truncated = self._clip_code_index_text(f"ask_human request ({request_id}):\n{clipped_question}")
        info = {"request_id": request_id, "status": "pending", "requires_human": True, "truncated": truncated}
        obs = {"text": text, "request": dict(entry), "tool_state": self._tool_state()}
        return obs, 0.08, False, info

    # ------------------------------------------------------------------ parsing tools
    def _handle_parse_code(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raw_text = action.get("text")
        raw_path = action.get("path")

        max_chars = max(200, min(int(action.get("max_chars", 120_000)), 1_500_000))
        include_docstrings = bool(action.get("include_docstrings", True))
        max_items = max(10, min(int(action.get("max_items", 200)), 2000))

        language = str(action.get("language") or "").strip().lower()
        path = None
        if raw_path:
            path = Path(str(raw_path))
            if not self._is_allowed_path(path):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "path_not_allowed", "path": str(raw_path)}
            if not path.exists() or not path.is_file():
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "file_not_found", "path": str(path)}
            if not language:
                language = path.suffix.lower().lstrip(".") or "python"

        if not language:
            language = "python"

        if str(language) in {"py", "python"}:
            language = "python"

        if language != "python":
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "unsupported_language", "language": language}

        code_text = ""
        read_info: Dict[str, Any] = {}
        if isinstance(raw_text, str) and raw_text.strip():
            code_text = raw_text[:max_chars]
            read_info = {"source": "inline", "chars": len(code_text)}
        elif path is not None:
            if self._fs_sandbox is not None:
                try:
                    content, extra = self._fs_sandbox.read_text(path, encoding="utf-8", errors="replace")
                except Exception as exc:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "read_failed", "path": str(path), "exception": repr(exc), "sandboxed": True}
                if isinstance(extra, dict) and extra.get("error"):
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, extra
                code_text = content[:max_chars]
                read_info = {"path": str(path), "chars": len(code_text), **dict(extra or {})}
            else:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "read_failed", "path": str(path), "exception": repr(exc)}
                code_text = content[:max_chars]
                read_info = {"path": str(path), "chars": len(code_text), "truncated": len(content) > len(code_text)}
        else:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_input", "reason": "provide_path_or_text"}

        try:
            from modules.knowledge.parsing_tools import parse_python_code
        except Exception as exc:  # pragma: no cover
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "parse_code_unavailable", "exception": repr(exc)}

        parsed = parse_python_code(code_text, filename=str(path) if path is not None else None, include_docstrings=include_docstrings, max_items=max_items)
        if isinstance(parsed, dict) and parsed.get("error"):
            info = {"error": str(parsed.get("error")), **read_info}
            for key in ("message", "lineno", "offset"):
                if key in parsed:
                    info[key] = parsed[key]
            return {"text": "", "tool_state": self._tool_state(), "parsed": parsed}, -1.0, False, info

        module = parsed.get("module") if isinstance(parsed, dict) else None
        func_names: List[str] = []
        class_names: List[str] = []
        if isinstance(module, dict):
            for fn in module.get("functions") or []:
                if isinstance(fn, dict) and fn.get("name"):
                    func_names.append(str(fn["name"]))
            for cl in module.get("classes") or []:
                if isinstance(cl, dict) and cl.get("name"):
                    class_names.append(str(cl["name"]))

        lines: List[str] = []
        if class_names:
            lines.append("classes: " + ", ".join(class_names[:12]))
        if func_names:
            lines.append("functions: " + ", ".join(func_names[:16]))
        if not lines:
            lines.append("no top-level classes/functions found")

        header = f"parse_code ({language})"
        if path is not None:
            header += f": {path.name}"
        text, truncated = self._clip_code_index_text(header + "\n" + "\n".join(f"- {line}" for line in lines))
        info: Dict[str, Any] = {"language": language, "truncated": truncated, "classes": len(class_names), "functions": len(func_names)}
        info.update(read_info)
        references: List[Dict[str, Any]] = []
        if path is not None:
            references.append({"url": str(path), "title": path.name, "source": "parse_code"})
        obs = {"text": text, "parsed": parsed, "references": references, "tool_state": self._tool_state()}
        return obs, 0.12, False, info

    def _handle_summarize_doc(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raw_text = action.get("text")
        raw_path = action.get("path")

        max_chars = max(200, min(int(action.get("max_chars", 120_000)), 1_500_000))
        max_summary_chars = max(120, min(int(action.get("max_summary_chars", 900)), 8000))
        max_headings = max(0, min(int(action.get("max_headings", 12)), 80))
        max_keywords = max(0, min(int(action.get("max_keywords", 12)), 80))
        max_formulas = max(0, min(int(action.get("max_formulas", 8)), 80))
        max_bytes = max(50_000, min(int(action.get("max_bytes", 15_000_000)), 200_000_000))

        path = None
        suffix = ""
        if raw_path:
            path = Path(str(raw_path))
            if not self._is_allowed_path(path):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "path_not_allowed", "path": str(raw_path)}
            if not path.exists() or not path.is_file():
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "file_not_found", "path": str(path)}
            suffix = path.suffix.lower()

        try:
            from modules.knowledge.parsing_tools import summarize_document
        except Exception as exc:  # pragma: no cover
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "summarize_doc_unavailable", "exception": repr(exc)}

        doc_text = None
        pdf_bytes = None
        read_info: Dict[str, Any] = {}

        if isinstance(raw_text, str) and raw_text.strip() and path is None:
            doc_text = raw_text[:max_chars]
            read_info = {"source": "inline", "chars": len(doc_text)}
        elif path is not None and suffix == ".pdf":
            if self._fs_sandbox is not None:
                try:
                    data, extra = self._fs_sandbox.read_bytes(path)
                except Exception as exc:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "read_failed", "path": str(path), "exception": repr(exc), "sandboxed": True}
                if isinstance(extra, dict) and extra.get("error"):
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, extra
                pdf_bytes = data[:max_bytes]
                read_info = {"path": str(path), "bytes": len(pdf_bytes), **dict(extra or {})}
            else:
                try:
                    data = path.read_bytes()
                except Exception as exc:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "read_failed", "path": str(path), "exception": repr(exc)}
                pdf_bytes = data[:max_bytes]
                read_info = {"path": str(path), "bytes": len(pdf_bytes), "truncated": len(data) > len(pdf_bytes)}
        elif path is not None:
            if self._fs_sandbox is not None:
                try:
                    content, extra = self._fs_sandbox.read_text(path, encoding="utf-8", errors="replace")
                except Exception as exc:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "read_failed", "path": str(path), "exception": repr(exc), "sandboxed": True}
                if isinstance(extra, dict) and extra.get("error"):
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, extra
                doc_text = content[:max_chars]
                read_info = {"path": str(path), "chars": len(doc_text), **dict(extra or {})}
            else:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "read_failed", "path": str(path), "exception": repr(exc)}
                doc_text = content[:max_chars]
                read_info = {"path": str(path), "chars": len(doc_text), "truncated": len(content) > len(doc_text)}
        else:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_input", "reason": "provide_path_or_text"}

        summary = summarize_document(
            text=doc_text,
            path=str(path) if path is not None else None,
            pdf_bytes=pdf_bytes,
            max_chars=max_chars,
            max_summary_chars=max_summary_chars,
            max_headings=max_headings,
            max_keywords=max_keywords,
            max_formulas=max_formulas,
        )
        if isinstance(summary, dict) and summary.get("error"):
            info = {"error": str(summary.get("error")), **read_info}
            meta = summary.get("meta")
            if isinstance(meta, dict):
                info["meta"] = meta
            details = summary.get("details")
            if isinstance(details, dict):
                info["details"] = details
            return {"text": "", "tool_state": self._tool_state(), "summary": summary}, -1.0, False, info

        lines: List[str] = []
        brief = str(summary.get("summary") or "").strip()
        if brief:
            lines.append(brief)
        headings = summary.get("headings")
        if isinstance(headings, list) and headings:
            lines.append("headings: " + "; ".join(str(h) for h in headings[: max(1, max_headings)]))
        keywords = summary.get("keywords")
        if isinstance(keywords, list) and keywords:
            lines.append("keywords: " + ", ".join(str(k) for k in keywords[: max(1, max_keywords)]))
        formulas = summary.get("formulas")
        if isinstance(formulas, list) and formulas:
            lines.append("formulas: " + "; ".join(str(f) for f in formulas[: max(1, max_formulas)]))

        header = "summarize_doc"
        if path is not None:
            header += f": {path.name}"
        text, truncated = self._clip_code_index_text(header + "\n" + "\n".join(f"- {line}" for line in lines if line))
        info: Dict[str, Any] = {"truncated": truncated, "suffix": suffix or None}
        info.update(read_info)
        references: List[Dict[str, Any]] = []
        if path is not None:
            references.append({"url": str(path), "title": path.name, "source": "summarize_doc"})
        obs = {"text": text, "summary": summary, "references": references, "tool_state": self._tool_state()}
        return obs, 0.12, False, info

    @staticmethod
    def _hash_embedding(text: str, *, dimensions: int = 12) -> List[float]:
        digest = hashlib.sha256(str(text or "").encode("utf-8")).digest()
        chunk = max(1, len(digest) // max(1, int(dimensions)))
        embedding: List[float] = []
        for index in range(0, len(digest), chunk):
            piece = digest[index : index + chunk]
            if not piece:
                continue
            integer = int.from_bytes(piece, byteorder="big", signed=False)
            embedding.append(integer / float(256 ** len(piece)))
            if len(embedding) == dimensions:
                break
        while len(embedding) < dimensions:
            embedding.append(0.0)
        return embedding[:dimensions]

    # --------------------------------------------------------------------- #
    def _is_allowed_path(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except Exception:
            return False
        for root in self._allowed_roots:
            try:
                resolved.relative_to(root)
                return True
            except Exception:
                continue
        return False

    def _handle_read_file(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raw_path = action.get("path")
        max_chars = int(action.get("max_chars", 4000))
        if not raw_path:
            return {"text": ""}, -1.0, False, {"error": "missing_path"}

        path = Path(str(raw_path))
        if not self._is_allowed_path(path):
            return {"text": ""}, -1.0, False, {"blocked": True, "reason": "path_not_allowed", "path": str(raw_path)}

        if self._fs_sandbox is not None:
            try:
                content, extra = self._fs_sandbox.read_text(path, encoding="utf-8", errors="replace")
            except Exception as exc:
                return {"text": ""}, -1.0, False, {"error": "read_failed", "path": str(path), "exception": repr(exc), "sandboxed": True}
            if isinstance(extra, dict) and extra.get("error"):
                return {"text": ""}, -1.0, False, extra
        else:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                extra = {}
            except Exception as exc:
                return {"text": ""}, -1.0, False, {"error": "read_failed", "path": str(path), "exception": repr(exc)}

        clipped = content[: max(0, max_chars)]
        info: Dict[str, Any] = {"path": str(path), "chars": len(clipped), "truncated": len(content) > len(clipped)}
        if isinstance(extra, dict):
            info.update(extra)
        return {"text": clipped, "tool_state": {"steps": self._steps}}, 0.2, False, info

    def _handle_list_dir(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raw_path = action.get("path", ".")
        max_entries = int(action.get("max_entries", 50))
        path = Path(str(raw_path))
        if not self._is_allowed_path(path):
            return {"text": ""}, -1.0, False, {"blocked": True, "reason": "path_not_allowed", "path": str(raw_path)}

        if self._fs_sandbox is not None:
            try:
                names, extra = self._fs_sandbox.list_dir(path, max_entries=max_entries)
            except Exception as exc:
                return {"text": ""}, -1.0, False, {"error": "list_failed", "path": str(path), "exception": repr(exc), "sandboxed": True}
            text = "\n".join(names)
            info: Dict[str, Any] = {
                "path": str(path.resolve()),
                "entries": names,
                "truncated": False,
                **(extra or {}),
            }
            return {"text": text, "tool_state": {"steps": self._steps}}, 0.1, False, info

        try:
            entries = sorted(path.iterdir(), key=lambda p: p.name)
        except Exception as exc:
            return {"text": ""}, -1.0, False, {"error": "list_failed", "path": str(path), "exception": repr(exc)}

        names = [p.name + ("/" if p.is_dir() else "") for p in entries[: max(0, max_entries)]]
        text = "\n".join(names)
        info: Dict[str, Any] = {"path": str(path), "entries": names, "truncated": len(entries) > len(names)}
        return {"text": text, "tool_state": {"steps": self._steps}}, 0.1, False, info

    # ------------------------------------------------------------------ web tools
    def _web_is_enabled(self) -> bool:
        if not bool(self._allow_web_access):
            return False
        if self._web_disabled_until:
            now = time.time()
            if now < self._web_disabled_until:
                return False
            self._web_disabled_until = 0.0
            self._web_disabled_reason = None
        return True

    def disable_web_access(self, *, reason: str, cooldown_s: float = 300.0) -> Dict[str, Any]:
        """Temporarily disable web tools (quarantine) to mitigate unsafe retrieval."""

        try:
            cooldown = max(0.0, float(cooldown_s))
        except Exception:
            cooldown = 300.0
        until = time.time() + cooldown
        self._web_disabled_until = max(self._web_disabled_until, until)
        self._web_disabled_reason = str(reason or "disabled_by_monitor")

        payload = {
            "event": "web_quarantine",
            "action": "disable_web_access",
            "reason": self._web_disabled_reason,
            "cooldown_s": round(float(cooldown), 3),
            "disabled_until": float(self._web_disabled_until),
        }
        security = getattr(self, "_security_manager", None)
        audit = getattr(security, "audit", None) if security is not None else None
        if audit is not None and getattr(audit, "enabled", True):
            try:
                audit.log(payload)
            except Exception:
                pass
        return payload

    def block_web_domain(self, domain: str, *, reason: str | None = None) -> Dict[str, Any]:
        """Block a web domain at runtime (in addition to static blocked lists)."""

        dom = str(domain or "").strip().lower().strip(".")
        if not dom:
            return {"blocked": False, "error": "missing_domain"}
        entry = {
            "domain": dom,
            "reason": str(reason or "blocked_by_monitor"),
            "timestamp": time.time(),
        }
        self._runtime_blocked_web_domains[dom] = entry

        security = getattr(self, "_security_manager", None)
        audit = getattr(security, "audit", None) if security is not None else None
        if audit is not None and getattr(audit, "enabled", True):
            try:
                audit.log({"event": "web_quarantine", "action": "block_web_domain", **entry})
            except Exception:
                pass
        return {"blocked": True, **entry}

    def _duckduckgo_module_is_local_stub(self, module: Any) -> bool:
        path = getattr(module, "__file__", None)
        if not path:
            return False
        try:
            mod_path = Path(str(path)).resolve()
        except Exception:
            return False
        repo_root = Path(__file__).resolve().parents[2]
        stub_root = (repo_root / "duckduckgo_search").resolve()
        return mod_path == stub_root / "__init__.py"

    def _try_load_real_duckduckgo_ddgs(self):
        if not self._prefer_real_web_search:
            return None
        try:
            import importlib

            original_sys_path = list(sys.path)
            try:
                repo_root = Path(__file__).resolve().parents[2]
                filtered: List[str] = []
                for entry in sys.path:
                    if entry in ("", "."):
                        continue
                    try:
                        if Path(entry).resolve() == repo_root:
                            continue
                    except Exception:
                        pass
                    filtered.append(entry)
                sys.path = filtered

                for name in list(sys.modules):
                    if name == "duckduckgo_search" or name.startswith("duckduckgo_search."):
                        sys.modules.pop(name, None)

                candidate = importlib.import_module("duckduckgo_search")
                if self._duckduckgo_module_is_local_stub(candidate):
                    return None
                return getattr(candidate, "DDGS", None)
            finally:
                sys.path = original_sys_path
        except Exception:
            return None

    def _get_ddgs_class(self):
        try:
            import duckduckgo_search as ddg_mod  # type: ignore
        except Exception:
            return None, "unavailable"

        ddgs = getattr(ddg_mod, "DDGS", None)
        if ddgs is None:
            return None, "unavailable"

        backend = "duckduckgo_search"
        if self._duckduckgo_module_is_local_stub(ddg_mod):
            backend = "stub"
            real = self._try_load_real_duckduckgo_ddgs()
            if real is not None:
                ddgs = real
                backend = "duckduckgo_search"
        return ddgs, backend

    def _is_allowed_web_domain(self, hostname: str) -> bool:
        host = str(hostname or "").strip().lower().strip(".")
        if not host:
            return False
        if not self._allowed_web_domains:
            return True
        for domain in self._allowed_web_domains:
            dom = str(domain or "").strip().lower().strip(".")
            if not dom:
                continue
            if host == dom or host.endswith("." + dom):
                return True
        return False

    def _is_blocked_web_domain(self, hostname: str) -> bool:
        host = str(hostname or "").strip().lower().strip(".")
        if not host:
            return False
        for domain in self._blocked_web_domains:
            dom = str(domain or "").strip().lower().strip(".")
            if not dom:
                continue
            if host == dom or host.endswith("." + dom):
                return True
        for dom in (self._runtime_blocked_web_domains or {}).keys():
            d = str(dom or "").strip().lower().strip(".")
            if not d:
                continue
            if host == d or host.endswith("." + d):
                return True
        return False

    def _web_trust_score(self, hostname: str) -> float:
        host = str(hostname or "").strip().lower().strip(".")
        if not host:
            return 0.5
        if self._is_blocked_web_domain(host):
            return 0.0
        for domain in self._trusted_web_domains:
            dom = str(domain or "").strip().lower().strip(".")
            if not dom:
                continue
            if host == dom or host.endswith("." + dom):
                return 1.0
        # Reputation map: best suffix match wins.
        best_score: float | None = None
        best_len = 0
        for dom, score in (self._web_domain_reputation or {}).items():
            d = str(dom or "").strip().lower().strip(".")
            if not d:
                continue
            if host == d or host.endswith("." + d):
                if len(d) >= best_len:
                    best_len = len(d)
                    best_score = float(score)
        if best_score is not None:
            return max(0.0, min(1.0, float(best_score)))
        return 0.5

    @staticmethod
    def _token_set_for_consensus(text: str, *, max_tokens: int = 80) -> set[str]:
        raw = str(text or "").lower()
        token_re = re.compile(r"\w+")
        stop = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "into",
            "your",
            "you",
            "are",
            "was",
            "were",
            "have",
            "has",
            "had",
            "will",
            "shall",
            "can",
            "could",
            "may",
            "might",
            "should",
            "would",
            "not",
            "its",
            "it's",
            "they",
            "their",
            "them",
            "we",
            "our",
            "us",
            "in",
            "on",
            "at",
            "to",
            "of",
            "a",
            "an",
            "is",
            "as",
            "be",
            "or",
            "by",
        }
        freq: Dict[str, int] = {}
        for tok in token_re.findall(raw):
            if len(tok) < 3:
                continue
            if tok.isdigit():
                continue
            if tok in stop:
                continue
            freq[tok] = freq.get(tok, 0) + 1
        ordered = sorted(freq.items(), key=lambda item: (-int(item[1]), item[0]))[: max(0, int(max_tokens))]
        return {t for t, _ in ordered if t}

    def _summarize_source_consensus(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        hosts: List[str] = []
        trust_scores: List[float] = []
        trust_counts = {"high": 0, "medium": 0, "low": 0}
        token_sets: List[set[str]] = []

        for src in sources or []:
            search = src.get("search") if isinstance(src, dict) else None
            page = src.get("page") if isinstance(src, dict) else None
            url = ""
            if isinstance(search, dict):
                url = str(search.get("url") or "")
            if not url and isinstance(page, dict):
                url = str(page.get("final_url") or page.get("url") or "")
            host = ""
            try:
                host = str(urlparse(url).hostname or "")
            except Exception:
                host = ""
            host = host.strip().lower().strip(".")
            if host:
                hosts.append(host)

            trust_score = None
            trust = None
            if isinstance(search, dict):
                trust = search.get("trust")
                trust_score = search.get("trust_score")
            if trust_score is None:
                trust_score = self._web_trust_score(host) if host else 0.5
            try:
                trust_f = float(trust_score)
            except Exception:
                trust_f = 0.5
            trust_scores.append(trust_f)

            trust_label = str(trust or "").strip().lower()
            if trust_label not in trust_counts:
                trust_label = "high" if trust_f >= 0.8 else "medium" if trust_f >= 0.4 else "low"
            trust_counts[trust_label] += 1

            page_text = ""
            if isinstance(page, dict):
                page_text = str(page.get("text") or "")
            token_sets.append(self._token_set_for_consensus(page_text, max_tokens=80))

        unique_hosts = sorted({h for h in hosts if h})
        avg_trust = sum(trust_scores) / max(1, len(trust_scores)) if trust_scores else 0.5
        min_trust = min(trust_scores) if trust_scores else 0.5
        max_trust = max(trust_scores) if trust_scores else 0.5

        similarity_avg: float | None = None
        sims: List[float] = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                a = token_sets[i]
                b = token_sets[j]
                union = a | b
                if not union:
                    sims.append(0.0)
                else:
                    sims.append(len(a & b) / len(union))
        if sims:
            similarity_avg = sum(sims) / len(sims)

        level = "unknown"
        if similarity_avg is not None:
            if similarity_avg >= 0.22:
                level = "high"
            elif similarity_avg >= 0.12:
                level = "medium"
            else:
                level = "low"

        warnings: List[str] = []
        if len(unique_hosts) < 2:
            warnings.append("single_host")
        if trust_counts["low"] > 0:
            warnings.append("low_trust_source")
        if level in {"unknown", "low"}:
            warnings.append("low_consensus")

        needs_verification = bool(warnings)
        return {
            "sources": int(len(sources or [])),
            "unique_hosts": int(len(unique_hosts)),
            "avg_trust": round(float(avg_trust), 3),
            "min_trust": round(float(min_trust), 3),
            "max_trust": round(float(max_trust), 3),
            "trust_counts": dict(trust_counts),
            "similarity_avg": round(float(similarity_avg), 3) if similarity_avg is not None else None,
            "level": level,
            "needs_verification": bool(needs_verification),
            "warnings": warnings,
        }

    def _is_public_hostname(self, hostname: str) -> bool:
        host = str(hostname or "").strip().lower()
        if not host:
            return False
        if host in {"localhost"} or host.endswith(".local"):
            return False

        try:
            ip = ipaddress.ip_address(host)
            return not (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            )
        except ValueError:
            pass

        try:
            infos = socket.getaddrinfo(host, None)
        except Exception:
            return False
        for info in infos or []:
            addr = info[4][0] if isinstance(info, (list, tuple)) and len(info) > 4 else None
            if not addr:
                continue
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                continue
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True

    def _validate_web_url(self, url: str) -> Tuple[bool, Dict[str, Any]]:
        parsed = urlparse(str(url or "").strip())
        if parsed.scheme not in {"http", "https"}:
            return False, {"blocked": True, "reason": "url_scheme_not_allowed", "scheme": parsed.scheme}
        host = parsed.hostname or ""
        if not host:
            return False, {"blocked": True, "reason": "url_missing_host"}
        if not self._is_allowed_web_domain(host):
            return False, {"blocked": True, "reason": "web_domain_not_allowed", "host": host}
        if self._is_blocked_web_domain(host):
            return False, {"blocked": True, "reason": "web_domain_blocked", "host": host}
        if not self._is_public_hostname(host):
            return False, {"blocked": True, "reason": "url_host_not_public", "host": host}
        return True, {"host": host, "scheme": parsed.scheme}

    def _clip_web_text(self, text: str) -> Tuple[str, bool]:
        limit = int(self._max_web_output_chars)
        if limit <= 0:
            return text, False
        if len(text) <= limit:
            return text, False
        return text[:limit], True

    def _clip_code_index_text(self, text: str) -> Tuple[str, bool]:
        limit = int(self._max_code_index_output_chars)
        if limit <= 0:
            return text, False
        if len(text) <= limit:
            return text, False
        return text[:limit], True

    def _handle_web_search(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._web_is_enabled():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "web_access_disabled"}

        query = str(action.get("query") or action.get("q") or "").strip()
        if not query:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_query"}

        max_results = max(1, min(int(action.get("max_results", 5)), 10))
        ddgs_cls, backend = self._get_ddgs_class()
        if ddgs_cls is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "duckduckgo_search_unavailable"}

        try:
            raw = ddgs_cls().text(query, max_results=max_results)
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "web_search_failed", "exception": repr(exc)}

        results: List[Dict[str, Any]] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "")
            url = str(item.get("href") or item.get("url") or "")
            snippet = str(item.get("body") or item.get("snippet") or "")
            if not (title or url or snippet):
                continue
            host = ""
            try:
                host = str(urlparse(url).hostname or "")
            except Exception:
                host = ""
            trust_score = float(self._web_trust_score(host)) if host else 0.5
            trust = "high" if trust_score >= 0.8 else "medium" if trust_score >= 0.4 else "low"
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "host": host,
                    "trust": trust,
                    "trust_score": round(float(trust_score), 3),
                    "blocked_domain": bool(host and self._is_blocked_web_domain(host)),
                }
            )

        enumerated = list(enumerate(results))
        enumerated.sort(key=lambda pair: (-float(pair[1].get("trust_score", 0.5)), pair[0]))
        results = [pair[1] for pair in enumerated][:max_results]

        hosts = [str(item.get("host") or "").strip().lower() for item in results if str(item.get("host") or "").strip()]
        unique_hosts = sorted({h for h in hosts if h})
        scores: List[float] = []
        trust_counts = {"high": 0, "medium": 0, "low": 0}
        for item in results:
            trust = str(item.get("trust") or "").strip().lower()
            if trust in trust_counts:
                trust_counts[trust] += 1
            try:
                scores.append(float(item.get("trust_score", 0.5)))
            except Exception:
                pass
        avg_trust = round(sum(scores) / max(1, len(scores)), 3) if scores else 0.5

        lines: List[str] = []
        for item in results:
            label = str(item.get("title") or item.get("url") or query)
            url = str(item.get("url") or "")
            if url:
                lines.append(f"- {label} ({url})")
            else:
                lines.append(f"- {label}")

        header = (
            "Results "
            f"(unique_hosts={len(unique_hosts)}, trust=high:{trust_counts['high']}/"
            f"med:{trust_counts['medium']}/low:{trust_counts['low']}, avg_trust={avg_trust:.3f})"
        )
        text, truncated = self._clip_web_text(header + ":\n" + "\n".join(lines))
        urls = [str(item.get("url") or "") for item in results if str(item.get("url") or "")]
        low_trust_urls = [
            str(item.get("url") or "")
            for item in results
            if str(item.get("url") or "") and str(item.get("trust") or "").lower() == "low"
        ]
        info: Dict[str, Any] = {
            "query": query,
            "returned": len(results),
            "truncated": truncated,
            "backend": backend,
            "unique_hosts": int(len(unique_hosts)),
            "trust_counts": dict(trust_counts),
            "avg_trust": float(avg_trust),
            "urls": urls[:max_results],
            "low_trust_urls": low_trust_urls[: max(0, min(5, max_results))],
        }
        return {"text": text, "results": results, "tool_state": self._tool_state()}, 0.12, False, info

    def _handle_web_scrape(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._web_is_enabled():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "web_access_disabled"}

        url = str(action.get("url") or "").strip()
        if not url:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_url"}

        allowed, info = self._validate_web_url(url)
        if not allowed:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, info

        max_chars = max(200, int(action.get("max_chars", 8000)))
        timeout_s = float(action.get("timeout_s", self._default_timeout_s))
        include_code = bool(action.get("include_code", True))
        max_code_blocks = max(0, int(action.get("max_code_blocks", 6)))

        try:
            payload = self._scrape_url(
                url,
                timeout_s=timeout_s,
                max_chars=max_chars,
                include_code=include_code,
                max_code_blocks=max_code_blocks,
            )
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "web_scrape_failed", "exception": repr(exc)}

        text_out = payload.get("text") or ""
        clipped, truncated = self._clip_web_text(str(text_out))
        obs = {
            "text": clipped,
            "page": payload,
            "tool_state": self._tool_state(),
        }
        meta = {
            **info,
            "status_code": payload.get("status_code"),
            "content_type": payload.get("content_type"),
            "title": payload.get("title"),
            "truncated": truncated,
        }
        reward = 0.18 if payload.get("status_code") and int(payload.get("status_code")) < 400 else 0.05
        return obs, reward, False, meta

    def _handle_web_get(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._web_is_enabled():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "web_access_disabled"}
        if requests is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "requests_unavailable"}

        url = str(action.get("url") or action.get("href") or "").strip()
        if not url:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_url"}

        allowed, info = self._validate_web_url(url)
        if not allowed:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, info

        timeout_s = float(action.get("timeout_s", self._default_timeout_s))
        max_bytes = max(0, int(action.get("max_bytes", 2_000_000)))
        max_chars = max(0, int(action.get("max_chars", 0)))
        if max_chars <= 0:
            max_chars = int(self._max_web_output_chars)

        headers = {"User-Agent": "BSS-WebGet/1.0"}
        try:
            resp = requests.get(  # type: ignore[union-attr]
                url,
                headers=headers,
                timeout=float(timeout_s),
            )
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "web_get_failed", "exception": repr(exc), **info}

        status_code = int(getattr(resp, "status_code", 0) or 0)
        content_type = str(getattr(resp, "headers", {}).get("Content-Type", ""))
        raw = getattr(resp, "content", b"") or b""
        truncated_bytes = False
        if max_bytes > 0 and len(raw) > max_bytes:
            raw = raw[:max_bytes]
            truncated_bytes = True

        encoding = getattr(resp, "encoding", None) or "utf-8"
        try:
            text_full = raw.decode(str(action.get("encoding") or encoding), errors="replace")
        except Exception:
            text_full = raw.decode("utf-8", errors="replace")

        truncated_by_max_chars = False
        if max_chars > 0 and len(text_full) > max_chars:
            truncated_by_max_chars = True
            text = text_full[:max_chars]
        else:
            text = text_full
        clipped, clipped_truncated = self._clip_web_text(text)
        truncated_text = bool(truncated_by_max_chars or clipped_truncated)

        obs = {
            "text": clipped,
            "url": url,
            "final_url": str(getattr(resp, "url", url) or url),
            "status_code": status_code,
            "content_type": content_type,
            "tool_state": self._tool_state(),
        }
        reward = 0.16 if status_code and status_code < 400 else 0.05
        meta: Dict[str, Any] = {
            **info,
            "status_code": status_code,
            "content_type": content_type,
            "returned_chars": len(clipped),
            "truncated": bool(truncated_text or truncated_bytes),
            "truncated_text": bool(truncated_text),
            "truncated_bytes": bool(truncated_bytes),
        }
        return obs, reward, False, meta

    def _scrape_url(
        self,
        url: str,
        *,
        timeout_s: float,
        max_chars: int,
        include_code: bool,
        max_code_blocks: int,
        max_bytes: int = 2_000_000,
    ) -> Dict[str, Any]:
        """Fetch and extract readable text from a URL with minimal dependencies.

        This intentionally avoids importing the full AutoGPT ability stack to
        keep BrainSimulationSystem usable in lightweight environments.
        """

        if requests is None:
            raise RuntimeError("requests_unavailable")

        headers = {"User-Agent": "BSS-WebScrape/1.0"}
        resp = requests.get(  # type: ignore[union-attr]
            url,
            headers=headers,
            timeout=float(timeout_s),
        )
        content_type = str(getattr(resp, "headers", {}).get("Content-Type", ""))
        raw = getattr(resp, "content", b"") or b""
        if max_bytes > 0 and len(raw) > max_bytes:
            raw = raw[:max_bytes]
        encoding = getattr(resp, "encoding", None) or "utf-8"
        try:
            html = raw.decode(encoding, errors="replace")
        except Exception:
            html = raw.decode("utf-8", errors="replace")

        title = ""
        code_blocks: List[str] = []
        extracted = html
        if "text/html" in content_type.lower() or "<html" in html.lower():
            extracted, title, code_blocks = self._extract_from_html(
                html,
                include_code=include_code,
                max_code_blocks=max_code_blocks,
            )
        else:
            extracted = re.sub(r"\s+", " ", extracted).strip()

        if max_chars > 0 and len(extracted) > max_chars:
            extracted = extracted[:max_chars]

        return {
            "url": url,
            "final_url": str(getattr(resp, "url", url) or url),
            "status_code": int(getattr(resp, "status_code", 0) or 0),
            "content_type": content_type,
            "title": title,
            "text": extracted,
            "code_blocks": code_blocks,
        }

    def _extract_from_html(
        self,
        html: str,
        *,
        include_code: bool,
        max_code_blocks: int,
    ) -> Tuple[str, str, List[str]]:
        title = ""
        main_html = html

        try:  # readability-lxml optional
            from readability import Document  # type: ignore

            main_html = Document(html).summary(html_partial=True) or html
        except Exception:
            main_html = html

        code_blocks: List[str] = []
        soup = None
        try:  # BeautifulSoup optional
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(main_html, "html.parser")
        except Exception:
            soup = None

        if soup is None:
            match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
            if match:
                title = re.sub(r"\s+", " ", match.group(1) or "").strip()
            text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", main_html)
            text = re.sub(r"(?is)<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text, title, []

        if soup.title and soup.title.string:
            title = re.sub(r"\s+", " ", soup.title.string).strip()
        for tag in soup(["script", "style", "noscript"]):
            try:
                tag.decompose()
            except Exception:
                pass

        if include_code and max_code_blocks > 0:
            for pre in soup.find_all("pre"):
                snippet = (pre.get_text("\n") or "").strip()
                if not snippet:
                    continue
                code_blocks.append(snippet[:2000])
                if len(code_blocks) >= max_code_blocks:
                    break

        text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        return text, title, code_blocks

    def _handle_github_code_search(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._web_is_enabled():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "web_access_disabled"}
        if requests is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "requests_unavailable"}

        query = str(action.get("query") or "").strip()
        if not query:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_query"}
        max_results = max(1, min(int(action.get("max_results", 5)), 20))

        token = action.get("token")
        headers = {
            "Accept": "application/vnd.github.text-match+json",
            "User-Agent": "BSS-GitHubCodeSearch/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            resp = requests.get(  # type: ignore[union-attr]
                "https://api.github.com/search/code",
                headers=headers,
                params={"q": query, "per_page": min(100, max_results)},
                timeout=10.0,
            )
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "github_search_failed", "exception": repr(exc)}

        try:
            payload = resp.json()
        except Exception:
            payload = {"message": resp.text}

        if int(resp.status_code) >= 400:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "github_api_error", "status_code": int(resp.status_code), "message": payload.get("message")}

        items = payload.get("items", []) if isinstance(payload, dict) else []
        results: List[Dict[str, Any]] = []
        lines: List[str] = []
        for item in (items or [])[:max_results]:
            if not isinstance(item, dict):
                continue
            repo = item.get("repository") or {}
            repo_full = repo.get("full_name") if isinstance(repo, dict) else ""
            path = str(item.get("path") or "")
            url = str(item.get("html_url") or "")
            fragments: List[str] = []
            text_matches = item.get("text_matches") or []
            if isinstance(text_matches, list):
                for match in text_matches[:3]:
                    if isinstance(match, dict) and match.get("fragment"):
                        fragments.append(str(match["fragment"]))
            results.append({"repository": repo_full, "path": path, "html_url": url, "fragments": fragments})
            lines.append(f"- {repo_full}/{path} ({url})")

        text, truncated = self._clip_web_text("GitHub matches:\n" + "\n".join(lines))
        urls = [str(item.get("html_url") or "") for item in results if str(item.get("html_url") or "")]
        info = {
            "query": query,
            "returned": len(results),
            "status_code": int(resp.status_code),
            "truncated": truncated,
            "urls": urls[:max_results],
        }
        return {"text": text, "results": results, "tool_state": self._tool_state()}, 0.12, False, info

    def _handle_github_repo_ingest(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._web_is_enabled():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "web_access_disabled"}
        if not self._allow_file_write:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "file_write_disabled"}

        raw_repo = action.get("repo") or action.get("repository") or action.get("url")
        repo = str(raw_repo or "").strip()
        if not repo:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_repo"}

        ref = str(action.get("ref") or action.get("branch") or action.get("tag") or "main").strip() or "main"
        force = bool(action.get("force", False))
        timeout_s = float(action.get("timeout_s", max(self._default_timeout_s, 20.0)))

        raw_dest_root = action.get("dest_root") or action.get("dest") or action.get("root")
        if raw_dest_root:
            dest_root = Path(str(raw_dest_root))
        else:
            base_root = self._allowed_roots[0] if self._allowed_roots else Path.cwd().resolve()
            dest_root = base_root / "data" / "external_repos"

        if not self._is_allowed_path(dest_root):
            return (
                {"text": "", "tool_state": self._tool_state()},
                -1.0,
                False,
                {"blocked": True, "reason": "path_not_allowed", "dest_root": str(dest_root)},
            )

        max_download_bytes = max(50_000, min(int(action.get("max_download_bytes", 30_000_000)), 250_000_000))
        max_unzipped_bytes = max(200_000, min(int(action.get("max_unzipped_bytes", 120_000_000)), 1_000_000_000))
        max_extract_files = max(1, min(int(action.get("max_extract_files", 20_000)), 200_000))

        extract_suffixes = action.get("extract_suffixes") if "extract_suffixes" in action else (".py", ".md", ".txt")
        if isinstance(extract_suffixes, str):
            extract_suffixes = [extract_suffixes]
        if extract_suffixes is not None and not isinstance(extract_suffixes, (list, tuple)):
            extract_suffixes = None

        try:
            from modules.knowledge.github_repo_ingest import (
                GitHubRepoIngestConfig,
                GitHubRepoIngestor,
                build_codeload_zip_url,
                parse_github_repo,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "repo_ingest_unavailable", "exception": repr(exc)}

        try:
            owner, name = parse_github_repo(repo)
            url = build_codeload_zip_url(owner, name, ref)
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "invalid_repo", "exception": str(exc)}

        allowed, why = self._validate_web_url(url)
        if not allowed:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, dict(why or {"blocked": True})

        try:
            ingestor = GitHubRepoIngestor()
            ingest_cfg = GitHubRepoIngestConfig(
                dest_root=dest_root,
                max_download_bytes=max_download_bytes,
                max_unzipped_bytes=max_unzipped_bytes,
                max_extract_files=max_extract_files,
                extract_suffixes=tuple(extract_suffixes) if extract_suffixes is not None else None,
                timeout_s=timeout_s,
            )
            ingest_result = ingestor.ingest(
                repo,
                ref=ref,
                config=ingest_cfg,
                force=force,
                metadata={"requested_via": "tool_bridge"},
            )
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "repo_ingest_failed", "exception": repr(exc), "url": url}

        repo_root = Path(ingest_result.repo_root)
        if not repo_root.exists() or not repo_root.is_dir():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "repo_root_missing", "repo_root": ingest_result.repo_root}

        key = str(repo_root.resolve())
        license_info: Dict[str, Any]
        try:
            license_info = detect_repo_license(repo_root)
        except Exception as exc:
            license_info = {"spdx": None, "confidence": 0.0, "file": None, "copyleft": False, "error": repr(exc)}

        policy = dict(self._license_policy or {})
        enforce_license = bool(action.get("enforce_license", policy.get("enforce", False)))
        require_license = bool(action.get("require_license", policy.get("require", False)))
        allow_unknown_license = bool(action.get("allow_unknown_license", policy.get("allow_unknown", True)))
        allowlist = normalize_spdx_list(
            action.get("license_allowlist") or action.get("allowed_licenses") or policy.get("allowlist")
        )
        denylist = normalize_spdx_list(
            action.get("license_denylist") or action.get("deny_licenses") or policy.get("denylist")
        )
        if enforce_license and not allowlist and not denylist:
            denylist = set(denylist) | {"GPL-3.0", "GPL-2.0", "AGPL-3.0", "LGPL-3.0", "LGPL-2.1"}

        license_spdx = str(license_info.get("spdx") or "").strip() or None
        license_allowed = True
        if enforce_license or allowlist or denylist or require_license:
            if license_spdx is None:
                if require_license:
                    license_allowed = False
                elif allowlist and not allow_unknown_license:
                    license_allowed = False
            else:
                if license_spdx in denylist:
                    license_allowed = False
                elif allowlist and license_spdx not in allowlist:
                    license_allowed = False

        if not license_allowed:
            cleanup: Dict[str, Any] = {"attempted": False}
            try:
                extracted = Path(ingest_result.extracted_dir)
                if extracted.exists():
                    cleanup["attempted"] = True
                    shutil.rmtree(extracted, ignore_errors=True)
                    cleanup["deleted"] = not extracted.exists()
                    cleanup["path"] = str(extracted)
            except Exception as exc:
                cleanup["error"] = repr(exc)

            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "blocked": True,
                "reason": "repo_license_not_allowed",
                "repo": ingest_result.repo,
                "ref": ingest_result.ref,
                "repo_root": key,
                "license": license_info,
                "license_policy": {
                    "enforce": bool(enforce_license),
                    "require": bool(require_license),
                    "allow_unknown": bool(allow_unknown_license),
                    "allowlist": sorted(list(allowlist)),
                    "denylist": sorted(list(denylist)),
                },
                "cleanup": cleanup,
            }

        self._code_index_metadata[key] = {
            "source": "github_repo_ingest",
            "repo": ingest_result.repo,
            "ref": ingest_result.ref,
            "url": ingest_result.url,
            "license": dict(license_info or {}),
        }

        index_stats: Dict[str, Any] | None = None
        build_index = bool(action.get("build_index", True))
        if build_index:
            try:
                from modules.knowledge.code_vector_index import CodeVectorIndex

                default_exclude_dirs = (
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    ".venv",
                    "node_modules",
                    ".bss_sandbox",
                )

                include_suffixes = action.get("index_include_suffixes") or action.get("index_suffixes") or (".py", ".md", ".txt", ".rst")
                if isinstance(include_suffixes, str):
                    include_suffixes = [include_suffixes]
                if not isinstance(include_suffixes, (list, tuple)):
                    include_suffixes = (".py", ".md", ".txt", ".rst")
                suffixes = tuple(
                    s if s.startswith(".") else f".{s}"
                    for s in (str(item).strip().lower() for item in include_suffixes)
                    if s
                ) or (".py",)

                exclude_dirs = action.get("index_exclude_dirs")
                if isinstance(exclude_dirs, str):
                    exclude_dirs = [exclude_dirs]
                if not isinstance(exclude_dirs, (list, tuple)):
                    exclude_dirs = None

                max_files = max(1, min(int(action.get("index_max_files", 800)), 5000))
                max_file_chars = max(1, min(int(action.get("index_max_file_chars", 250_000)), 1_500_000))
                chunk_lines = max(5, min(int(action.get("index_chunk_lines", 80)), 300))
                chunk_overlap = max(0, min(int(action.get("index_chunk_overlap", 10)), chunk_lines // 2))
                max_chunk_chars = max(200, min(int(action.get("index_max_chunk_chars", 8000)), 40_000))
                embedding_dimensions = max(8, min(int(action.get("index_embedding_dimensions", 128)), 2048))

                index = CodeVectorIndex(
                    root=key,
                    persist_path=None,
                    read_text_fn=None,
                    include_suffixes=suffixes,
                    exclude_dirs=exclude_dirs or default_exclude_dirs,
                    max_files=max_files,
                    max_file_chars=max_file_chars,
                    chunk_lines=chunk_lines,
                    chunk_overlap=chunk_overlap,
                    max_chunk_chars=max_chunk_chars,
                    embedding_dimensions=embedding_dimensions,
                    use_faiss=False,
                )
                index_stats = index.build(save=False)
                self._code_indexes[key] = index
            except Exception as exc:
                index_stats = {"error": "code_index_build_failed", "exception": repr(exc)}

        license_tag = str(license_spdx or "unknown")
        summary = (
            f"github_repo_ingest: repo={ingest_result.repo}@{ingest_result.ref} "
            f"root={key} license={license_tag} extracted_files={ingest_result.extract_stats.get('extracted_files')} "
            f"download_bytes={ingest_result.download_bytes}"
        )
        text, truncated = self._clip_code_index_text(summary)
        info: Dict[str, Any] = {
            "repo": ingest_result.repo,
            "ref": ingest_result.ref,
            "repo_root": key,
            "url": ingest_result.url,
            "truncated": truncated,
            "build_index": build_index,
            "license": dict(license_info or {}),
            "license_allowed": True,
        }
        if isinstance(index_stats, dict):
            info["index"] = dict(index_stats)

        obs: Dict[str, Any] = {
            "text": text,
            "repo": ingest_result.repo,
            "ref": ingest_result.ref,
            "repo_root": key,
            "archive_url": ingest_result.url,
            "license": dict(license_info or {}),
            "ingest": {
                "extracted_dir": ingest_result.extracted_dir,
                "manifest_path": ingest_result.manifest_path,
                "download_bytes": ingest_result.download_bytes,
                "extract": dict(ingest_result.extract_stats or {}),
            },
            "tool_state": self._tool_state(),
        }
        if index_stats is not None:
            obs["index_stats"] = index_stats
        return obs, 0.12, False, info

    def _handle_documentation_tool(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._web_is_enabled():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "web_access_disabled"}

        query = str(action.get("query") or "").strip()
        if not query:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_query"}

        max_sources = max(1, min(int(action.get("max_sources", 2)), 5))
        max_chars = max(200, int(action.get("max_chars_per_source", 8000)))
        timeout_s = float(action.get("timeout_s", self._default_timeout_s))

        search_obs, _, _, search_info = self.step(
            {"type": "web_search", "query": f"{query} documentation", "max_results": max(3, max_sources * 3)}
        )
        if isinstance(search_info, dict) and (search_info.get("blocked") or search_info.get("error")):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, dict(search_info)
        raw_results = search_obs.get("results") if isinstance(search_obs, dict) else None
        results = list(raw_results) if isinstance(raw_results, list) else []

        sources: List[Dict[str, Any]] = []
        lines: List[str] = []
        source_urls: List[str] = []
        source_hosts: List[str] = []
        blocked_urls: List[str] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            allowed, why = self._validate_web_url(url)
            if not allowed:
                if isinstance(why, dict) and (why.get("blocked") or why.get("reason")):
                    blocked_urls.append(url)
                continue
            scrape_obs, _, _, scrape_info = self.step(
                {
                    "type": "web_scrape",
                    "url": url,
                    "max_chars": max_chars,
                    "timeout_s": timeout_s,
                    "include_code": True,
                }
            )
            if isinstance(scrape_info, dict) and (scrape_info.get("blocked") or scrape_info.get("error")):
                blocked_urls.append(url)
                continue
            page = scrape_obs.get("page") if isinstance(scrape_obs, dict) else None
            sources.append({"search": item, "page": page, "scrape_info": scrape_info})
            title = str(item.get("title") or url)
            lines.append(f"- {title} ({url})")
            source_urls.append(url)
            host = str(scrape_info.get("host") or item.get("host") or "").strip().lower().strip(".") if isinstance(scrape_info, dict) else ""
            if host:
                source_hosts.append(host)
            if len(sources) >= max_sources:
                break

        consensus = self._summarize_source_consensus(sources)
        level = str(consensus.get("level") or "unknown")
        score = consensus.get("similarity_avg")
        avg_trust = consensus.get("avg_trust")
        header = "Documentation sources"
        header += f" (unique_hosts={int(consensus.get('unique_hosts', 0))}"
        header += f", consensus={level}"
        if score is not None:
            try:
                header += f" score={float(score):.3f}"
            except Exception:
                pass
        if avg_trust is not None:
            try:
                header += f", avg_trust={float(avg_trust):.3f}"
            except Exception:
                pass
        if consensus.get("needs_verification"):
            header += ", needs_verification"
        header += ")"

        text, truncated = self._clip_web_text(header + ":\n" + "\n".join(lines))
        info = {
            "query": query,
            "returned": len(sources),
            "truncated": truncated,
            "search": dict(search_info or {}),
            "consensus": consensus,
            "source_urls": source_urls[:max_sources],
            "source_hosts": sorted({h for h in source_hosts if h})[: max(0, max_sources * 2)],
            "blocked_urls": blocked_urls[: max(0, max_sources * 3)],
        }
        return {
            "text": text,
            "sources": sources,
            "consensus": consensus,
            "tool_state": self._tool_state(),
        }, 0.12, False, info

    # ------------------------------------------------------------------ local code index tools
    def _handle_code_index_build(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raw_root = action.get("root") or action.get("path") or "."
        root = Path(str(raw_root))
        if not self._is_allowed_path(root):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "path_not_allowed", "root": str(raw_root)}

        if root.exists() and root.is_file():
            root = root.parent
        if not root.exists() or not root.is_dir():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "root_not_dir", "root": str(root)}

        def _read_text(path: Path, max_chars: int) -> str:
            if not self._is_allowed_path(path):
                return ""
            if self._fs_sandbox is not None:
                try:
                    text, extra = self._fs_sandbox.read_text(path, encoding="utf-8", errors="replace")
                except Exception:
                    return ""
                if isinstance(extra, dict) and extra.get("error"):
                    return ""
            else:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    return ""
            limit = max(0, int(max_chars))
            if limit > 0 and len(text) > limit:
                return text[:limit]
            return text

        include_suffixes = action.get("include_suffixes") or action.get("suffixes") or (".py", ".md", ".txt", ".rst")
        if isinstance(include_suffixes, str):
            include_suffixes = [include_suffixes]
        if not isinstance(include_suffixes, (list, tuple)):
            include_suffixes = (".py", ".md", ".txt", ".rst")
        suffixes = tuple(
            s if s.startswith(".") else f".{s}"
            for s in (str(item).strip().lower() for item in include_suffixes)
            if s
        ) or (".py",)

        exclude_dirs = action.get("exclude_dirs")
        if isinstance(exclude_dirs, str):
            exclude_dirs = [exclude_dirs]
        if not isinstance(exclude_dirs, (list, tuple)):
            exclude_dirs = None

        max_files = max(1, min(int(action.get("max_files", 800)), 5000))
        max_file_chars = max(1, min(int(action.get("max_file_chars", 250_000)), 1_500_000))
        chunk_lines = max(5, min(int(action.get("chunk_lines", 80)), 300))
        chunk_overlap = max(0, min(int(action.get("chunk_overlap", 10)), chunk_lines // 2))
        max_chunk_chars = max(200, min(int(action.get("max_chunk_chars", 8000)), 40_000))
        embedding_dimensions = max(8, min(int(action.get("embedding_dimensions", 128)), 2048))

        try:
            from modules.knowledge.code_vector_index import CodeVectorIndex
        except Exception as exc:  # pragma: no cover - defensive
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "code_index_unavailable", "exception": repr(exc)}

        default_exclude_dirs = (
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "node_modules",
            ".bss_sandbox",
        )

        index = CodeVectorIndex(
            root=str(root),
            persist_path=None,
            read_text_fn=_read_text,
            include_suffixes=suffixes,
            exclude_dirs=exclude_dirs or default_exclude_dirs,
            max_files=max_files,
            max_file_chars=max_file_chars,
            chunk_lines=chunk_lines,
            chunk_overlap=chunk_overlap,
            max_chunk_chars=max_chunk_chars,
            embedding_dimensions=embedding_dimensions,
            use_faiss=False,
        )
        stats = index.build(save=False)

        key = str(root.resolve())
        self._code_indexes[key] = index

        summary = (
            f"code_index built: files={stats.get('files_scanned')} chunks={stats.get('chunks_indexed')} "
            f"errors={stats.get('file_errors')} duration_s={stats.get('duration_s')} root={key}"
        )
        text, truncated = self._clip_code_index_text(summary)
        return {"text": text, "stats": stats, "tool_state": self._tool_state()}, 0.15, False, {"root": key, "truncated": truncated, **stats}

    def _handle_code_index_search(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raw_root = action.get("root") or action.get("path") or "."
        root = Path(str(raw_root))
        if not self._is_allowed_path(root):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "path_not_allowed", "root": str(raw_root)}

        if root.exists() and root.is_file():
            root = root.parent
        key = str(root.resolve())
        index = self._code_indexes.get(key)
        if index is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "code_index_not_built", "root": key}

        query = str(action.get("query") or action.get("q") or "").strip()
        if not query:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_query"}

        top_k = max(1, min(int(action.get("top_k", 5)), 25))
        max_chars_per_hit = max(0, int(action.get("max_chars_per_hit", 2000)))
        max_reference_chars = max(0, int(action.get("max_reference_chars", 2000)))

        try:
            hits = index.search(query, top_k=top_k)
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "code_index_search_failed", "exception": repr(exc)}

        clipped_hits: List[Dict[str, Any]] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            payload = dict(hit)
            snippet = payload.get("text")
            if isinstance(snippet, str) and max_chars_per_hit > 0 and len(snippet) > max_chars_per_hit:
                payload["text"] = snippet[:max_chars_per_hit]
                payload["truncated"] = True
            clipped_hits.append(payload)

        try:
            references = index.hits_as_references(clipped_hits, max_chars=max_reference_chars)
        except Exception:
            references = []

        root_meta = self._code_index_metadata.get(key) if hasattr(self, "_code_index_metadata") else None
        license_meta = None
        if isinstance(root_meta, dict) and isinstance(root_meta.get("license"), dict):
            license_meta = dict(root_meta.get("license") or {})
            spdx = str(license_meta.get("spdx") or "").strip() or None
            copyleft = bool(license_meta.get("copyleft"))
            for hit in clipped_hits:
                if not isinstance(hit, dict):
                    continue
                if spdx:
                    hit.setdefault("license_spdx", spdx)
                hit.setdefault("license_copyleft", copyleft)
            for ref in references:
                if not isinstance(ref, dict):
                    continue
                if spdx:
                    ref.setdefault("license_spdx", spdx)
                ref.setdefault("license_copyleft", copyleft)

        lines: List[str] = []
        for hit in clipped_hits[:top_k]:
            path = str(hit.get("path") or "")
            symbol = str(hit.get("symbol") or "")
            start = hit.get("start_line")
            sim = hit.get("similarity")
            label = path
            if symbol:
                label += f"::{symbol}"
            if start:
                label += f":{int(start)}"
            if sim is not None:
                try:
                    label += f" (sim={float(sim):.3f})"
                except Exception:
                    pass
            lines.append(f"- {label}")

        header = "code_index hits"
        if isinstance(license_meta, dict) and license_meta.get("spdx"):
            header = f"{header} (license={license_meta.get('spdx')})"
            if license_meta.get("copyleft"):
                header = f"{header} [copyleft: do not copy verbatim; reimplement]"
        text, truncated = self._clip_code_index_text(header + ":\n" + "\n".join(lines))
        info: Dict[str, Any] = {"root": key, "returned": len(clipped_hits), "truncated": truncated}
        if isinstance(license_meta, dict) and license_meta:
            info["license"] = dict(license_meta)
        return {"text": text, "hits": clipped_hits, "references": references, "tool_state": self._tool_state()}, 0.12, False, info

    def _handle_sandbox_status(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._fs_sandbox is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "sandbox_not_enabled", **(self._fs_sandbox_status or {})}

        status_obj = self._fs_sandbox.status()
        status = {
            "enabled": True,
            "root": status_obj.root,
            "changed": list(status_obj.changed),
            "deleted": list(status_obj.deleted),
        }
        return {"text": "sandbox status", "tool_state": self._tool_state(), "sandbox": status}, 0.05, False, status

    def _handle_sandbox_commit(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._fs_sandbox is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "sandbox_not_enabled", **(self._fs_sandbox_status or {})}
        if not self._allow_sandbox_commit:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "sandbox_commit_disabled"}

        expected = self._sandbox_confirm_token
        provided = action.get("confirm_token")
        if expected and provided != expected:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "sandbox_confirm_token_required"}

        try:
            summary = self._fs_sandbox.commit()
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "sandbox_commit_failed", "exception": repr(exc)}

        if not self._sandbox_keep_history:
            try:
                self._fs_sandbox.reset()
            except Exception:
                pass

        return {"text": "sandbox committed", "tool_state": self._tool_state(), "sandbox": summary}, 0.1, False, summary

    def _handle_sandbox_reset(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._fs_sandbox is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "sandbox_not_enabled", **(self._fs_sandbox_status or {})}
        if not self._allow_sandbox_commit:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "sandbox_reset_disabled"}

        expected = self._sandbox_confirm_token
        provided = action.get("confirm_token")
        if expected and provided != expected:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "sandbox_confirm_token_required"}

        try:
            summary = self._fs_sandbox.reset()
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "sandbox_reset_failed", "exception": repr(exc)}

        return {"text": "sandbox reset", "tool_state": self._tool_state(), "sandbox": summary}, 0.05, False, summary

    def _tool_state(self) -> Dict[str, Any]:
        return {"steps": self._steps}

    def _audit_action(
        self,
        stage: str,
        action: Dict[str, Any],
        observation: Dict[str, Any],
        reward: float,
        terminated: bool,
        info: Dict[str, Any],
    ) -> None:
        security = getattr(self, "_security_manager", None)
        audit = getattr(security, "audit", None) if security is not None else None
        if audit is None or not getattr(audit, "enabled", True):
            return

        try:
            payload: Dict[str, Any] = {
                "event": "tool_action",
                "stage": str(stage),
                "steps": int(self._steps),
                "action_type": str(action.get("type") or ""),
                "action": redact_action(action, max_chars=512) if callable(redact_action) else dict(action),
                "reward": float(reward),
                "terminated": bool(terminated),
                "info": dict(info or {}),
            }
            text = observation.get("text")
            if isinstance(text, str):
                payload["observation_chars"] = len(text)
            audit.log(payload)
        except Exception:
            return

    def _normalize_path(self, raw_path: Any) -> Tuple[Optional[Path], Dict[str, Any]]:
        if not raw_path:
            return None, {"error": "missing_path"}
        path = Path(str(raw_path))
        if not self._is_allowed_path(path):
            return None, {"blocked": True, "reason": "path_not_allowed", "path": str(raw_path)}
        try:
            return path.resolve(), {}
        except Exception as exc:
            return None, {"error": "invalid_path", "path": str(raw_path), "exception": repr(exc)}

    def _handle_write_file(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._allow_file_write:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "write_disabled"}

        path, info = self._normalize_path(action.get("path"))
        if path is None:
            info.setdefault("blocked", info.get("blocked", False))
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, info

        text = action.get("text")
        if text is None:
            text = action.get("content")
        if text is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_text", "path": str(path)}

        encoding = str(action.get("encoding", "utf-8"))
        append = bool(action.get("append", False))
        create_parents = bool(action.get("create_parents", True))
        payload = str(text)
        if self._max_write_chars and len(payload) > self._max_write_chars:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "error": "write_too_large",
                "path": str(path),
                "chars": len(payload),
                "max_chars": self._max_write_chars,
            }

        if self._fs_sandbox is not None:
            try:
                extra = self._fs_sandbox.write_text(
                    path,
                    payload,
                    encoding=encoding,
                    errors="replace",
                    append=append,
                    create_parents=create_parents,
                )
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "write_failed", "path": str(path), "exception": repr(exc), "sandboxed": True}
        else:
            extra = {}
            try:
                if create_parents:
                    path.parent.mkdir(parents=True, exist_ok=True)
                mode = "a" if append else "w"
                with path.open(mode, encoding=encoding, errors="replace", newline="") as handle:
                    handle.write(payload)
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "write_failed", "path": str(path), "exception": repr(exc)}

        action_label = "appended" if append else "written"
        content_sha1 = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()
        info: Dict[str, Any] = {"path": str(path), "chars": len(payload), "append": append, "content_sha1": content_sha1}
        if isinstance(extra, dict):
            info.update(extra)
        return {"text": f"{action_label} {len(payload)} chars", "tool_state": self._tool_state()}, 0.2, False, info

    def _handle_create_file(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._allow_file_write:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "write_disabled"}

        path, info = self._normalize_path(action.get("path"))
        if path is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, info

        overwrite = bool(action.get("overwrite", False))
        existed_before = self._fs_sandbox.exists(path) if self._fs_sandbox is not None else path.exists()
        if existed_before and not overwrite:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "file_exists", "path": str(path)}

        text = action.get("text")
        if text is None:
            text = action.get("content", "")
        encoding = str(action.get("encoding", "utf-8"))
        create_parents = bool(action.get("create_parents", True))
        payload = str(text)
        if self._max_write_chars and len(payload) > self._max_write_chars:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "error": "write_too_large",
                "path": str(path),
                "chars": len(payload),
                "max_chars": self._max_write_chars,
            }

        if self._fs_sandbox is not None:
            try:
                extra = self._fs_sandbox.write_text(
                    path,
                    payload,
                    encoding=encoding,
                    errors="replace",
                    append=False,
                    create_parents=create_parents,
                )
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "create_file_failed", "path": str(path), "exception": repr(exc), "sandboxed": True}
        else:
            extra = {}
            try:
                if create_parents:
                    path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload, encoding=encoding, errors="replace")
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "create_file_failed", "path": str(path), "exception": repr(exc)}

        content_sha1 = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()
        info_out: Dict[str, Any] = {
            "path": str(path),
            "chars": len(payload),
            "overwritten": bool(existed_before and overwrite),
            "content_sha1": content_sha1,
        }
        if isinstance(extra, dict):
            info_out.update(extra)
        return {"text": f"created {path.name}", "tool_state": self._tool_state()}, 0.2, False, info_out

    def _handle_modify_file(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._allow_file_write:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "write_disabled"}

        path, info = self._normalize_path(action.get("path"))
        if path is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, info

        exists = self._fs_sandbox.exists(path) if self._fs_sandbox is not None else path.exists()
        if not exists:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "file_missing", "path": str(path)}
        if path.is_dir():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "path_is_dir", "path": str(path)}

        encoding = str(action.get("encoding", "utf-8"))
        if self._fs_sandbox is not None:
            try:
                content, extra_read = self._fs_sandbox.read_text(path, encoding=encoding, errors="replace")
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "read_failed", "path": str(path), "exception": repr(exc), "sandboxed": True}
            if isinstance(extra_read, dict) and extra_read.get("error"):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, extra_read
        else:
            extra_read = {}
            try:
                content = path.read_text(encoding=encoding, errors="replace")
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "read_failed", "path": str(path), "exception": repr(exc)}

        operation = str(action.get("operation", "replace")).strip().lower()
        updated = content
        changes = 0

        if operation in ("replace", "string_replace"):
            old = action.get("old")
            new = action.get("new", "")
            if old is None:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_old", "path": str(path)}
            count = int(action.get("count", 0))
            if count <= 0:
                updated = content.replace(str(old), str(new))
                changes = content.count(str(old))
            else:
                updated = content.replace(str(old), str(new), count)
                changes = min(content.count(str(old)), count)
        elif operation in ("regex_replace", "re_sub"):
            pattern = action.get("pattern")
            repl = action.get("repl", "")
            if pattern is None:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_pattern", "path": str(path)}
            flags_val = 0
            for flag in action.get("flags", []) if isinstance(action.get("flags", []), list) else []:
                flag_name = str(flag).upper()
                if flag_name == "MULTILINE":
                    flags_val |= re.MULTILINE
                elif flag_name == "DOTALL":
                    flags_val |= re.DOTALL
                elif flag_name == "IGNORECASE":
                    flags_val |= re.IGNORECASE
            count = int(action.get("count", 0))
            updated, changes = re.subn(str(pattern), str(repl), content, count=0 if count <= 0 else count, flags=flags_val)
        else:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "unknown_modify_operation", "operation": operation}

        if updated == content:
            return {"text": "no changes", "tool_state": self._tool_state()}, 0.0, False, {"path": str(path), "changes": 0, "operation": operation}

        if self._max_write_chars and len(updated) > self._max_write_chars:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "error": "modified_content_too_large",
                "path": str(path),
                "chars": len(updated),
                "max_chars": self._max_write_chars,
            }

        if self._fs_sandbox is not None:
            try:
                extra_write = self._fs_sandbox.write_text(
                    path,
                    updated,
                    encoding=encoding,
                    errors="replace",
                    append=False,
                    create_parents=True,
                )
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "modify_write_failed", "path": str(path), "exception": repr(exc), "sandboxed": True}
        else:
            extra_write = {}
            try:
                path.write_text(updated, encoding=encoding, errors="replace")
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "modify_write_failed", "path": str(path), "exception": repr(exc)}

        info_out: Dict[str, Any] = {"path": str(path), "changes": int(changes), "operation": operation}
        if isinstance(extra_read, dict):
            info_out.update(extra_read)
        if isinstance(extra_write, dict):
            info_out.update(extra_write)
        return {"text": f"modified ({changes} changes)", "tool_state": self._tool_state()}, 0.2, False, info_out

    def _handle_delete_file(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._allow_file_delete:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "delete_disabled"}

        path, info = self._normalize_path(action.get("path"))
        if path is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, info

        missing_ok = bool(action.get("missing_ok", False))
        exists = self._fs_sandbox.exists(path) if self._fs_sandbox is not None else path.exists()
        if not exists:
            if missing_ok:
                return {"text": "missing (ok)", "tool_state": self._tool_state()}, 0.0, False, {"path": str(path), "deleted": False, "missing": True}
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "file_missing", "path": str(path)}
        if path.is_dir():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "path_is_dir", "path": str(path)}

        if self._fs_sandbox is not None:
            try:
                extra = self._fs_sandbox.delete_file(path)
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "delete_failed", "path": str(path), "exception": repr(exc), "sandboxed": True}
            info_out: Dict[str, Any] = {"path": str(path), "deleted": True}
            if isinstance(extra, dict):
                info_out.update(extra)
            return {"text": f"deleted {path.name} (sandbox)", "tool_state": self._tool_state()}, 0.1, False, info_out

        try:
            path.unlink()
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "delete_failed", "path": str(path), "exception": repr(exc)}

        return {"text": f"deleted {path.name}", "tool_state": self._tool_state()}, 0.1, False, {"path": str(path), "deleted": True}

    def _handle_create_dir(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._allow_file_write:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "write_disabled"}

        path, info = self._normalize_path(action.get("path"))
        if path is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, info

        parents = bool(action.get("parents", True))
        exist_ok = bool(action.get("exist_ok", True))

        if self._fs_sandbox is not None:
            try:
                extra = self._fs_sandbox.mkdir(path, parents=parents, exist_ok=exist_ok)
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "create_dir_failed", "path": str(path), "exception": repr(exc), "sandboxed": True}
            info_out: Dict[str, Any] = {"path": str(path)}
            if isinstance(extra, dict):
                info_out.update(extra)
            return {"text": f"created dir {path.name} (sandbox)", "tool_state": self._tool_state()}, 0.1, False, info_out

        try:
            path.mkdir(parents=parents, exist_ok=exist_ok)
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "create_dir_failed", "path": str(path), "exception": repr(exc)}

        return {"text": f"created dir {path.name}", "tool_state": self._tool_state()}, 0.1, False, {"path": str(path)}

    def _handle_shell(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        command = action.get("command")
        timeout_s = float(action.get("timeout_s", self._default_timeout_s))
        if not isinstance(command, list) or not command:
            return {"text": ""}, -1.0, False, {"error": "command_must_be_list", "blocked": True}

        if not self._is_allowed_shell_command(command):
            return {"text": ""}, -1.0, False, {"blocked": True, "reason": "shell_command_not_allowed", "command": command}

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(0.1, timeout_s),
                check=False,
            )
        except Exception as exc:
            return {"text": ""}, -1.0, False, {"error": "shell_failed", "command": command, "exception": repr(exc)}

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        text = (stdout + ("\n" if stdout and stderr else "") + stderr).strip()
        reward = 0.2 if completed.returncode == 0 else -0.2
        info: Dict[str, Any] = {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout_chars": len(stdout),
            "stderr_chars": len(stderr),
        }
        return {"text": text, "tool_state": {"steps": self._steps}}, reward, False, info

    def _is_allowed_shell_command(self, command: List[str]) -> bool:
        if not self._allowed_shell_prefixes:
            return False
        for prefix in self._allowed_shell_prefixes:
            if len(command) >= len(prefix) and command[: len(prefix)] == prefix:
                return True
        return False

    def _is_allowed_program_command(self, command: List[str]) -> bool:
        if not self._allowed_program_prefixes:
            return False
        for prefix in self._allowed_program_prefixes:
            if len(command) >= len(prefix) and command[: len(prefix)] == prefix:
                return True
        return False

    def _handle_launch_program(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._allow_process_control:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "process_control_disabled"}

        command = action.get("command")
        if not isinstance(command, list) or not command:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "command_must_be_list", "blocked": True}

        if not self._is_allowed_program_command(command):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "program_not_allowed", "command": command}

        cwd = action.get("cwd")
        cwd_path = None
        if cwd:
            cwd_path, cwd_info = self._normalize_path(cwd)
            if cwd_path is None:
                cwd_info.setdefault("blocked", cwd_info.get("blocked", False))
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, cwd_info
            if not cwd_path.exists() or not cwd_path.is_dir():
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "cwd_not_dir", "cwd": str(cwd_path)}

        try:
            proc = subprocess.Popen(
                command,
                cwd=str(cwd_path) if cwd_path else None,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "launch_failed", "command": command, "exception": repr(exc)}

        self._processes[int(proc.pid)] = proc
        info: Dict[str, Any] = {"pid": int(proc.pid), "command": command, "cwd": str(cwd_path) if cwd_path else None}
        return {"text": f"launched pid={proc.pid}", "tool_state": self._tool_state()}, 0.3, False, info

    def _handle_kill_process(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._allow_process_control:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "process_control_disabled"}

        pid_raw = action.get("pid")
        try:
            pid = int(pid_raw)
        except Exception:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_or_invalid_pid", "pid": pid_raw}

        tracked = pid in self._processes
        if not tracked and not self._allow_kill_untracked:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "pid_not_tracked", "pid": pid}

        timeout_s = float(action.get("timeout_s", 3.0))
        force = bool(action.get("force", True))

        returncode: Optional[int] = None
        was_running: Optional[bool] = None

        try:
            proc_obj = self._processes.get(pid)
            if proc_obj is not None:
                was_running = proc_obj.poll() is None
                if was_running:
                    proc_obj.terminate()
                    try:
                        proc_obj.wait(timeout=max(0.1, timeout_s))
                    except Exception:
                        if force:
                            proc_obj.kill()
                            proc_obj.wait(timeout=max(0.1, timeout_s))
                returncode = proc_obj.returncode
            else:
                # Untracked PID: only allowed in explicit override mode.
                if psutil is not None and hasattr(psutil, "Process") and hasattr(psutil.Process, "terminate"):
                    process = psutil.Process(pid)
                    process.terminate()
                    try:
                        returncode = process.wait(timeout=max(0.1, timeout_s))
                    except Exception:
                        if force and hasattr(process, "kill"):
                            process.kill()
                            returncode = process.wait(timeout=max(0.1, timeout_s))
                else:
                    # Last resort: best-effort terminate via platform tools.
                    if os.name == "nt":  # pragma: no cover - platform specific
                        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, check=False)
                    else:  # pragma: no cover - platform specific
                        os.kill(pid, 15)
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "kill_failed", "pid": pid, "exception": repr(exc)}
        finally:
            if tracked:
                self._processes.pop(pid, None)

        info: Dict[str, Any] = {"pid": pid, "killed": True, "returncode": returncode, "was_running": was_running, "tracked": tracked}
        return {"text": f"killed pid={pid}", "tool_state": self._tool_state()}, 0.1, False, info

    # --------------------------------------------------------------------- #
    # System command / settings control ----------------------------------- #
    def _split_command_string(self, raw: str) -> List[str]:
        posix = os.name != "nt"
        parts = shlex.split(str(raw), posix=posix)
        if not posix:
            cleaned: List[str] = []
            for item in parts:
                token = str(item)
                if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
                    token = token[1:-1]
                cleaned.append(token)
            return cleaned
        return [str(p) for p in parts]

    def _parse_command(self, action: Dict[str, Any]) -> Tuple[Optional[List[str]], Dict[str, Any]]:
        raw = action.get("command")
        if raw is None:
            raw = action.get("cmd")
        if raw is None:
            return None, {"error": "missing_command"}

        if isinstance(raw, list):
            command = [str(item) for item in raw if item is not None]
        elif isinstance(raw, str):
            try:
                command = self._split_command_string(raw)
            except Exception as exc:
                return None, {"error": "command_parse_failed", "exception": repr(exc)}
        else:
            return None, {"error": "command_must_be_list_or_str", "command_type": type(raw).__name__}

        if not command:
            return None, {"error": "empty_command"}

        return command, {}

    @staticmethod
    def _normalize_binary_name(token: str) -> str:
        name = Path(str(token).strip().strip("'").strip('"')).name.lower()
        return name[:-4] if name.endswith(".exe") else name

    def _command_prefix_matches(self, command: List[str], prefix: List[str]) -> bool:
        if not prefix:
            return False
        if len(command) < len(prefix):
            return False
        if command[: len(prefix)] == prefix:
            return True
        # Relax match for the binary path component (token 0).
        if self._normalize_binary_name(command[0]) != self._normalize_binary_name(prefix[0]):
            return False
        if len(prefix) == 1:
            return True
        return command[1 : len(prefix)] == prefix[1:]

    def _is_allowed_system_cmd(self, command: List[str]) -> bool:
        if not self._allowed_system_cmd_prefixes:
            return False
        return any(self._command_prefix_matches(command, prefix) for prefix in self._allowed_system_cmd_prefixes)

    def _is_high_risk_system_cmd(self, command: List[str]) -> bool:
        binary = self._normalize_binary_name(command[0]) if command else ""
        if not binary:
            return False
        # Conservative: treat explicit power/network/registry/disk admin tools as high risk.
        high_risk = {
            # Power/session control
            "shutdown",
            "reboot",
            "poweroff",
            "halt",
            "init",
            "systemctl",
            "logoff",
            # Network configuration
            "netsh",
            "nmcli",
            "ifconfig",
            "ip",
            "iptables",
            "ufw",
            # System config / boot / registry
            "reg",
            "regedit",
            "bcdedit",
            # Disk/filesystem admin
            "diskpart",
            "format",
            "mkfs",
            "mount",
            "umount",
            # Process killing (prefer dedicated kill_process API)
            "taskkill",
            "kill",
            "killall",
        }
        if binary in high_risk:
            return True
        # Shells can execute arbitrary commands.
        shells = {"cmd", "powershell", "pwsh", "bash", "sh", "zsh"}
        return binary in shells

    def _handle_exec_system_cmd(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if os.getenv("BSS_SYSTEM_CONTROL_DISABLE") in ("1", "true", "TRUE"):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "system_control_disabled_by_env"}

        if not self._allow_system_cmd:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "system_cmd_disabled"}

        command, parse_info = self._parse_command(action)
        if command is None:
            parse_info.setdefault("blocked", True)
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, parse_info

        if not self._is_allowed_system_cmd(command):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "system_cmd_not_allowed", "command": command}

        high_risk = self._is_high_risk_system_cmd(command)
        if high_risk:
            if not self._allow_high_risk_system_cmd:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "high_risk_command_blocked", "command": command}
            if not self._system_confirm_token:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "confirmation_token_not_configured", "command": command}
            provided = action.get("confirm_token") or action.get("confirmation_token")
            if str(provided or "") != str(self._system_confirm_token):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "confirmation_required", "command": command, "high_risk": True}

        timeout_s = float(action.get("timeout_s", self._default_timeout_s))
        dry_run = bool(action.get("dry_run", False))

        cwd_raw = action.get("cwd")
        cwd_path = None
        if cwd_raw:
            cwd_path, cwd_info = self._normalize_path(cwd_raw)
            if cwd_path is None:
                cwd_info.setdefault("blocked", cwd_info.get("blocked", False))
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, cwd_info
            if not cwd_path.exists() or not cwd_path.is_dir():
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "cwd_not_dir", "cwd": str(cwd_path)}

        if dry_run:
            info: Dict[str, Any] = {"command": command, "dry_run": True, "high_risk": high_risk, "cwd": str(cwd_path) if cwd_path else None}
            return {"text": "dry_run", "tool_state": self._tool_state()}, 0.05, False, info

        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd_path) if cwd_path else None,
                capture_output=True,
                text=True,
                timeout=max(0.1, timeout_s),
                check=False,
            )
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "system_cmd_failed", "command": command, "exception": repr(exc)}

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        text = (stdout + ("\n" if stdout and stderr else "") + stderr).strip()
        clipped = text
        truncated = False
        if self._max_system_cmd_output_chars and len(clipped) > self._max_system_cmd_output_chars:
            clipped = clipped[: self._max_system_cmd_output_chars]
            truncated = True

        reward = 0.2 if completed.returncode == 0 else -0.2
        info = {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout_chars": len(stdout),
            "stderr_chars": len(stderr),
            "truncated": truncated,
            "high_risk": high_risk,
            "cwd": str(cwd_path) if cwd_path else None,
        }
        return {"text": clipped, "tool_state": self._tool_state()}, reward, False, info

    def _build_system_setting_command(self, name: str, value: Any) -> Tuple[Optional[List[str]], Dict[str, Any]]:
        key = str(name or "").strip().lower()
        if not key:
            return None, {"error": "missing_setting_name"}

        payload: Dict[str, Any] = {}
        if isinstance(value, dict):
            payload.update(value)
        elif isinstance(value, (int, float)):
            payload["delay_s"] = int(value)
        if key in ("shutdown", "power.shutdown", "system.shutdown"):
            delay_s = int(payload.get("delay_s", 0))
            force = bool(payload.get("force", True))
            if os.name == "nt":
                cmd = ["shutdown", "/s", "/t", str(max(0, delay_s))]
                if force:
                    cmd.insert(2, "/f")
                return cmd, {"high_risk": True}
            return ["shutdown", "-h", "now"], {"high_risk": True}

        if key in ("restart", "power.restart", "system.restart", "power.reboot"):
            delay_s = int(payload.get("delay_s", 0))
            force = bool(payload.get("force", True))
            if os.name == "nt":
                cmd = ["shutdown", "/r", "/t", str(max(0, delay_s))]
                if force:
                    cmd.insert(2, "/f")
                return cmd, {"high_risk": True}
            return ["shutdown", "-r", "now"], {"high_risk": True}

        if key in ("network.flush_dns", "network.flushdns", "dns.flush"):
            if os.name == "nt":
                return ["ipconfig", "/flushdns"], {"high_risk": False}
            return None, {"error": "setting_not_supported_on_platform", "platform": sys.platform}

        return None, {"error": "unknown_setting", "name": key}

    def _handle_change_system_setting(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        name = action.get("name") or action.get("setting")
        raw_value = action.get("value", {})
        value: Any = raw_value
        if isinstance(raw_value, dict):
            merged_value = dict(raw_value)
        else:
            merged_value = {}
        for key in ("delay_s", "force"):
            if key in action and key not in merged_value:
                merged_value[key] = action.get(key)
        if merged_value:
            if isinstance(raw_value, dict):
                value = merged_value
            else:
                value = {**merged_value, "value": raw_value}

        command, info = self._build_system_setting_command(str(name or ""), value)
        if command is None:
            info.setdefault("blocked", True)
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, info

        forwarded = dict(action)
        forwarded["command"] = command
        forwarded["type"] = "exec_system_cmd"
        obs, reward, terminated, cmd_info = self._handle_exec_system_cmd(forwarded)
        merged = dict(cmd_info or {})
        merged["setting_name"] = str(name)
        return obs, reward, terminated, merged

    # --------------------------------------------------------------------- #
    # Docker control ------------------------------------------------------- #
    def _is_allowed_docker_image(self, image: str) -> bool:
        image_str = str(image or "").strip()
        if not image_str:
            return False
        if not self._allowed_docker_image_prefixes:
            return False
        return any(image_str.startswith(prefix) for prefix in self._allowed_docker_image_prefixes)

    def _docker_client(self, *, timeout_s: float) -> Any:
        if docker is None:
            raise RuntimeError("docker_sdk_unavailable")
        return docker.from_env(timeout=max(0.1, float(timeout_s)))

    @staticmethod
    def _docker_container_ref(action: Dict[str, Any]) -> Optional[str]:
        for key in ("container", "name", "id", "container_id", "container_name"):
            value = action.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def _docker_container_is_tracked(self, ref: str) -> bool:
        if not ref:
            return False
        return ref in self._docker_tracked_container_ids or ref in self._docker_tracked_container_names

    def _docker_track_container(self, container: Any) -> None:
        try:
            cid = str(getattr(container, "id", "") or "").strip()
            name = str(getattr(container, "name", "") or "").strip()
        except Exception:  # pragma: no cover - defensive
            return
        if cid:
            self._docker_tracked_container_ids.add(cid)
        if name:
            self._docker_tracked_container_names.add(name)

    def _docker_untrack_container(self, container: Any) -> None:
        try:
            cid = str(getattr(container, "id", "") or "").strip()
            name = str(getattr(container, "name", "") or "").strip()
        except Exception:  # pragma: no cover - defensive
            cid = ""
            name = ""
        if cid:
            self._docker_tracked_container_ids.discard(cid)
        if name:
            self._docker_tracked_container_names.discard(name)

    def _clip_docker_text(self, text: str) -> Tuple[str, bool]:
        raw = str(text or "")
        clipped = raw[: max(0, int(self._max_docker_output_chars))]
        return clipped, len(raw) > len(clipped)

    def _normalize_docker_volumes(self, volumes: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if volumes is None:
            return {}, {}
        if not isinstance(volumes, dict):
            return None, {"error": "docker_volumes_must_be_dict"}

        normalized: Dict[str, Any] = {}
        for host_raw, spec in volumes.items():
            host_path = Path(str(host_raw))
            if not self._is_allowed_path(host_path):
                return None, {"blocked": True, "reason": "docker_volume_host_path_not_allowed", "path": str(host_raw)}
            try:
                host_resolved = str(host_path.resolve())
            except Exception as exc:
                return None, {"error": "docker_volume_invalid_host_path", "path": str(host_raw), "exception": repr(exc)}

            if isinstance(spec, str):
                bind = spec
                mode = "rw"
            elif isinstance(spec, dict):
                bind = spec.get("bind") or spec.get("target")
                if not bind:
                    return None, {"error": "docker_volume_missing_bind", "path": str(host_raw)}
                mode = spec.get("mode", "rw")
            else:
                return None, {"error": "docker_volume_invalid_spec", "path": str(host_raw)}

            mode_str = str(mode).lower()
            if mode_str not in {"ro", "rw"}:
                return None, {"error": "docker_volume_invalid_mode", "path": str(host_raw), "mode": str(mode)}
            normalized[host_resolved] = {"bind": str(bind), "mode": mode_str}

        return normalized, {}

    def _cleanup_docker_containers(self) -> None:
        if docker is None:
            return
        ids = list(self._docker_tracked_container_ids)
        if not ids:
            return
        try:
            client = docker.from_env(timeout=max(0.1, float(self._default_timeout_s)))
        except Exception:
            return
        for cid in ids:
            try:
                container = client.containers.get(cid)
            except Exception:
                continue
            try:
                container.stop(timeout=1)
            except Exception:
                continue
        self._docker_tracked_container_ids.clear()
        self._docker_tracked_container_names.clear()

    def _handle_docker(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if os.getenv("BSS_DOCKER_CONTROL_DISABLE") in ("1", "true", "TRUE"):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "blocked": True,
                "reason": "docker_control_disabled_by_env",
            }
        if not self._allow_docker_control:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_control_disabled"}
        if docker is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_sdk_unavailable"}

        op = str(action.get("action") or action.get("op") or "").strip().lower()
        if not op:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_docker_action"}

        timeout_s = float(action.get("timeout_s", self._default_timeout_s))
        dry_run = bool(action.get("dry_run", False))

        try:
            client = self._docker_client(timeout_s=timeout_s)
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_client_failed", "exception": repr(exc)}

        if op == "ping":
            if dry_run:
                return {"text": "dry_run", "tool_state": self._tool_state()}, 0.05, False, {"action": "ping", "dry_run": True}
            try:
                ok = bool(client.ping())
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_ping_failed", "exception": repr(exc)}
            return {"text": "pong" if ok else "no_pong", "tool_state": self._tool_state()}, (0.1 if ok else -0.2), False, {"action": "ping", "ok": ok}

        if op in {"images.pull", "image.pull", "pull_image", "pull"}:
            image = action.get("image") or action.get("name") or action.get("repository")
            image_str = str(image or "").strip()
            if not image_str:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_image"}
            if not self._allowed_docker_image_prefixes:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_image_allowlist_not_configured"}
            if not self._is_allowed_docker_image(image_str):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_image_not_allowed", "image": image_str}

            tag = action.get("tag")
            platform = action.get("platform")
            if dry_run:
                return {"text": "dry_run", "tool_state": self._tool_state()}, 0.05, False, {
                    "action": "images.pull",
                    "image": image_str,
                    "tag": str(tag) if tag else None,
                    "platform": str(platform) if platform else None,
                    "dry_run": True,
                }

            try:
                if tag:
                    pulled = client.images.pull(image_str, tag=str(tag), platform=str(platform) if platform else None)
                else:
                    pulled = client.images.pull(image_str, platform=str(platform) if platform else None)
            except DockerException as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_pull_failed", "image": image_str, "exception": str(exc)}
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_pull_failed", "image": image_str, "exception": repr(exc)}

            image_obj = pulled[0] if isinstance(pulled, list) and pulled else pulled
            info: Dict[str, Any] = {
                "action": "images.pull",
                "image": image_str,
                "id": getattr(image_obj, "id", None),
                "tags": getattr(image_obj, "tags", None),
            }
            return {"text": f"pulled {image_str}", "tool_state": self._tool_state(), "docker": {"image": info}}, 0.2, False, info

        if op in {"containers.run", "container.run", "run_container", "run"}:
            image = action.get("image") or action.get("name") or action.get("repository")
            image_str = str(image or "").strip()
            if not image_str:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_image"}
            if not self._allowed_docker_image_prefixes:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_image_allowlist_not_configured"}
            if not self._is_allowed_docker_image(image_str):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_image_not_allowed", "image": image_str}

            forbidden_keys = {
                "privileged",
                "cap_add",
                "cap_drop",
                "devices",
                "device_requests",
                "mounts",
                "network_mode",
                "pid_mode",
                "ipc_mode",
                "extra_hosts",
                "security_opt",
                "sysctls",
            }
            for key in forbidden_keys:
                if key in action:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                        "blocked": True,
                        "reason": "docker_run_option_not_allowed",
                        "option": key,
                    }

            detach = bool(action.get("detach", True))
            if not detach:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_run_requires_detach"}

            run_kwargs: Dict[str, Any] = {"detach": True}
            if "remove" in action:
                run_kwargs["remove"] = bool(action.get("remove"))
            if action.get("name"):
                run_kwargs["name"] = str(action.get("name"))
            if action.get("command") is not None:
                cmd = action.get("command")
                if isinstance(cmd, list):
                    run_kwargs["command"] = [str(x) for x in cmd]
                else:
                    run_kwargs["command"] = str(cmd)
            if action.get("environment") is not None:
                env = action.get("environment")
                if isinstance(env, dict):
                    run_kwargs["environment"] = {str(k): str(v) for k, v in env.items()}
                else:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_environment_must_be_dict"}
            if action.get("ports") is not None:
                ports = action.get("ports")
                if isinstance(ports, dict):
                    run_kwargs["ports"] = dict(ports)
                else:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_ports_must_be_dict"}
            if action.get("working_dir") is not None:
                run_kwargs["working_dir"] = str(action.get("working_dir"))
            if action.get("volumes") is not None:
                vols, vols_info = self._normalize_docker_volumes(action.get("volumes"))
                if vols is None:
                    vols_info.setdefault("blocked", vols_info.get("blocked", False))
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, vols_info
                run_kwargs["volumes"] = vols

            if dry_run:
                redacted = {k: v for k, v in run_kwargs.items() if k != "environment"}
                info = {"action": "containers.run", "image": image_str, "dry_run": True, "kwargs": redacted}
                return {"text": "dry_run", "tool_state": self._tool_state(), "docker": {"run": info}}, 0.05, False, info

            try:
                container = client.containers.run(image_str, **run_kwargs)
            except DockerException as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_run_failed", "image": image_str, "exception": str(exc)}
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_run_failed", "image": image_str, "exception": repr(exc)}

            self._docker_track_container(container)
            info = {
                "action": "containers.run",
                "image": image_str,
                "id": getattr(container, "id", None),
                "name": getattr(container, "name", None),
                "status": getattr(container, "status", None),
            }
            text = f"started {info.get('name') or info.get('id') or 'container'}"
            return {"text": text, "tool_state": self._tool_state(), "docker": {"container": info}}, 0.3, False, info

        if op in {"containers.get", "container.get", "containers.status", "container.status", "get_container"}:
            ref = self._docker_container_ref(action)
            if not ref:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_container"}
            if not self._allow_docker_untracked and not self._docker_container_is_tracked(ref):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_container_not_tracked", "container": ref}
            if dry_run:
                return {"text": "dry_run", "tool_state": self._tool_state()}, 0.05, False, {"action": "containers.get", "container": ref, "dry_run": True}

            try:
                container = client.containers.get(ref)
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_container_get_failed", "container": ref, "exception": repr(exc)}

            summary = {
                "action": "containers.get",
                "id": getattr(container, "id", None),
                "name": getattr(container, "name", None),
                "status": getattr(container, "status", None),
            }
            text = f"{summary.get('name') or ref} status={summary.get('status')}"
            return {"text": text, "tool_state": self._tool_state(), "docker": {"container": summary}}, 0.05, False, summary

        if op in {"containers.stop", "container.stop", "stop_container", "stop"}:
            ref = self._docker_container_ref(action)
            if not ref:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_container"}
            if not self._allow_docker_untracked and not self._docker_container_is_tracked(ref):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_container_not_tracked", "container": ref}
            stop_timeout_raw = action.get("stop_timeout_s", 10)
            try:
                stop_timeout = int(stop_timeout_raw)
            except Exception:
                stop_timeout = 10
            if dry_run:
                return {"text": "dry_run", "tool_state": self._tool_state()}, 0.05, False, {
                    "action": "containers.stop",
                    "container": ref,
                    "stop_timeout_s": stop_timeout,
                    "dry_run": True,
                }
            try:
                container = client.containers.get(ref)
                container.stop(timeout=stop_timeout)
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_container_stop_failed", "container": ref, "exception": repr(exc)}
            return {"text": f"stopped {ref}", "tool_state": self._tool_state()}, 0.1, False, {"action": "containers.stop", "container": ref, "stopped": True}

        if op in {"containers.remove", "container.remove", "delete_container", "remove_container"}:
            if not self._allow_docker_delete:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_delete_disabled"}
            ref = self._docker_container_ref(action)
            if not ref:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_container"}
            if not self._allow_docker_untracked and not self._docker_container_is_tracked(ref):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_container_not_tracked", "container": ref}
            force = bool(action.get("force", False))
            remove_volumes = bool(action.get("v", False))
            if dry_run:
                return {"text": "dry_run", "tool_state": self._tool_state()}, 0.05, False, {
                    "action": "containers.remove",
                    "container": ref,
                    "force": force,
                    "v": remove_volumes,
                    "dry_run": True,
                }
            try:
                container = client.containers.get(ref)
                container.remove(force=force, v=remove_volumes)
                self._docker_untrack_container(container)
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_container_remove_failed", "container": ref, "exception": repr(exc)}
            return {"text": f"removed {ref}", "tool_state": self._tool_state()}, 0.1, False, {"action": "containers.remove", "container": ref, "removed": True}

        if op in {"containers.exec", "container.exec", "exec", "exec_run"}:
            ref = self._docker_container_ref(action)
            if not ref:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_container"}
            if not self._allow_docker_untracked and not self._docker_container_is_tracked(ref):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_container_not_tracked", "container": ref}
            cmd = action.get("cmd") if action.get("cmd") is not None else action.get("command")
            if cmd is None:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_exec_command"}

            if isinstance(cmd, list):
                cmd_val: Any = [str(x) for x in cmd]
            else:
                cmd_val = str(cmd)

            exec_kwargs: Dict[str, Any] = {"stdout": True, "stderr": True, "demux": False}
            if action.get("workdir") is not None:
                exec_kwargs["workdir"] = str(action.get("workdir"))
            if action.get("user") is not None:
                exec_kwargs["user"] = str(action.get("user"))

            if dry_run:
                return {"text": "dry_run", "tool_state": self._tool_state()}, 0.05, False, {
                    "action": "containers.exec",
                    "container": ref,
                    "cmd": cmd_val,
                    "dry_run": True,
                }

            try:
                container = client.containers.get(ref)
                result = container.exec_run(cmd_val, **exec_kwargs)
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_exec_failed", "container": ref, "exception": repr(exc)}

            exit_code = getattr(result, "exit_code", None)
            output = getattr(result, "output", None)
            if isinstance(result, tuple) and len(result) == 2:
                exit_code = result[0]
                output = result[1]
            if isinstance(output, bytes):
                text_out = output.decode("utf-8", errors="replace")
            else:
                text_out = str(output or "")

            clipped, truncated = self._clip_docker_text(text_out)
            code = int(exit_code or 0)
            reward = 0.2 if code == 0 else -0.2
            info = {"action": "containers.exec", "container": ref, "exit_code": code, "truncated": truncated}
            return {"text": clipped, "tool_state": self._tool_state()}, reward, False, info

        if op in {"containers.logs", "container.logs", "logs"}:
            ref = self._docker_container_ref(action)
            if not ref:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_container"}
            if not self._allow_docker_untracked and not self._docker_container_is_tracked(ref):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "docker_container_not_tracked", "container": ref}
            tail_raw = action.get("tail", 200)
            try:
                tail = int(tail_raw) if str(tail_raw).lower() != "all" else "all"
            except Exception:
                tail = 200
            if dry_run:
                return {"text": "dry_run", "tool_state": self._tool_state()}, 0.05, False, {"action": "containers.logs", "container": ref, "tail": tail, "dry_run": True}
            try:
                container = client.containers.get(ref)
                output = container.logs(tail=tail)
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "docker_logs_failed", "container": ref, "exception": repr(exc)}
            if isinstance(output, bytes):
                text_out = output.decode("utf-8", errors="replace")
            else:
                text_out = str(output or "")
            clipped, truncated = self._clip_docker_text(text_out)
            return {"text": clipped, "tool_state": self._tool_state()}, 0.05, False, {
                "action": "containers.logs",
                "container": ref,
                "tail": tail,
                "truncated": truncated,
            }

        return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "unknown_docker_action", "action": op}

    # --------------------------------------------------------------------- #
    # Docker Compose control ------------------------------------------------ #
    @staticmethod
    def _normalize_docker_compose_binary_name(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        base = os.path.basename(raw).strip().lower()
        if base.endswith(".exe"):
            base = base[:-4]
        return base if base in {"docker", "docker-compose"} else ""

    @staticmethod
    def _normalize_compose_service_name(value: Any) -> Optional[str]:
        raw = str(value or "").strip()
        if not raw:
            return None
        # Conservative: service names are simple identifiers in compose files.
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", raw):
            return None
        return raw

    def _normalize_compose_files(
        self, project_dir: Path, files: List[Any]
    ) -> Tuple[Optional[List[str]], Dict[str, Any]]:
        if not files:
            return [], {}

        normalized: List[str] = []
        for raw in files:
            path = Path(str(raw))
            if not path.is_absolute():
                path = project_dir / path
            if not self._is_allowed_path(path):
                return None, {"blocked": True, "reason": "path_not_allowed", "path": str(path)}
            try:
                resolved = path.resolve()
            except Exception as exc:
                return None, {"error": "invalid_path", "path": str(path), "exception": repr(exc)}
            if not resolved.exists() or not resolved.is_file():
                return None, {"error": "compose_file_not_found", "path": str(resolved)}
            normalized.append(str(resolved))

        return normalized, {}

    def _handle_docker_compose(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if os.getenv("BSS_DOCKER_CONTROL_DISABLE") in ("1", "true", "TRUE") or os.getenv(
            "BSS_DOCKER_COMPOSE_DISABLE"
        ) in ("1", "true", "TRUE"):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "blocked": True,
                "reason": "docker_control_disabled_by_env",
            }

        if not self._allow_docker_compose:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "blocked": True,
                "reason": "docker_compose_disabled",
            }

        op = str(action.get("action") or action.get("op") or "").strip().lower()
        if not op:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_compose_action"}

        project_dir_raw = action.get("project_dir") or action.get("project_directory") or action.get("cwd")
        if not project_dir_raw:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_project_dir"}

        project_dir, project_info = self._normalize_path(project_dir_raw)
        if project_dir is None:
            project_info.setdefault("blocked", project_info.get("blocked", False))
            project_info.setdefault("context", "project_dir")
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, project_info
        if not project_dir.exists() or not project_dir.is_dir():
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "error": "project_dir_not_dir",
                "project_dir": str(project_dir),
            }

        files_raw = action.get("files") or action.get("file") or action.get("compose_file")
        if files_raw is None:
            files: List[Any] = []
        elif isinstance(files_raw, str):
            files = [files_raw]
        elif isinstance(files_raw, list):
            files = list(files_raw)
        else:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "compose_files_must_be_list"}

        files_norm, files_info = self._normalize_compose_files(project_dir, files)
        if files_norm is None:
            files_info.setdefault("blocked", files_info.get("blocked", False))
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, files_info

        binary_raw = action.get("binary") or "docker"
        binary = str(binary_raw or "").strip() or "docker"
        binary_norm = self._normalize_docker_compose_binary_name(binary)
        if not self._allowed_docker_compose_binaries:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "blocked": True,
                "reason": "docker_compose_binary_allowlist_not_configured",
            }
        if not binary_norm or binary_norm not in self._allowed_docker_compose_binaries:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "blocked": True,
                "reason": "docker_compose_binary_not_allowed",
                "binary": str(binary),
            }

        if binary_norm == "docker-compose":
            cmd: List[str] = [binary]
        else:
            cmd = [binary, "compose"]

        for path in files_norm:
            cmd.extend(["-f", path])

        project_name = action.get("project_name") or action.get("project")
        if project_name is not None:
            project_str = str(project_name).strip()
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", project_str):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                    "error": "invalid_project_name",
                    "project_name": project_str,
                }
            cmd.extend(["-p", project_str])

        dry_run = bool(action.get("dry_run", False))
        timeout_s = float(action.get("timeout_s", self._default_timeout_s))

        services_raw = action.get("services")
        services: List[str] = []
        if isinstance(services_raw, list):
            for entry in services_raw:
                name = self._normalize_compose_service_name(entry)
                if name is None:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                        "error": "invalid_service_name",
                        "service": str(entry),
                    }
                services.append(name)
        elif services_raw is not None and not isinstance(services_raw, dict):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "services_must_be_list_or_dict"}

        if op == "up":
            detach = bool(action.get("detach", True))
            cmd.append("up")
            if detach:
                cmd.append("-d")
            if bool(action.get("remove_orphans", False)):
                cmd.append("--remove-orphans")
            if services:
                cmd.extend(services)
        elif op == "down":
            if not self._allow_docker_compose_delete:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                    "blocked": True,
                    "reason": "docker_compose_delete_disabled",
                }
            if action.get("rmi") is not None or action.get("remove_images") is not None:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                    "blocked": True,
                    "reason": "docker_compose_option_not_allowed",
                    "option": "rmi",
                }
            cmd.append("down")
            if bool(action.get("remove_orphans", False)):
                cmd.append("--remove-orphans")
            if bool(action.get("remove_volumes", False)):
                cmd.append("-v")
            timeout_down = action.get("timeout_s") or action.get("timeout")
            if timeout_down is not None:
                try:
                    timeout_down_i = int(timeout_down)
                except Exception:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                        "error": "invalid_down_timeout",
                        "timeout_s": timeout_down,
                    }
                cmd.extend(["-t", str(max(0, timeout_down_i))])
        elif op == "scale":
            scales_raw = action.get("services") if isinstance(action.get("services"), dict) else action.get("scale")
            if not isinstance(scales_raw, dict) or not scales_raw:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_scale_services"}

            cmd.extend(["up", "-d"])
            scale_services: List[str] = []
            for svc, replicas_raw in scales_raw.items():
                name = self._normalize_compose_service_name(svc)
                if name is None:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                        "error": "invalid_service_name",
                        "service": str(svc),
                    }
                try:
                    replicas = int(replicas_raw)
                except Exception:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                        "error": "invalid_scale_replicas",
                        "service": name,
                        "replicas": replicas_raw,
                    }
                if replicas < 0:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                        "error": "invalid_scale_replicas",
                        "service": name,
                        "replicas": replicas,
                    }
                cmd.extend(["--scale", f"{name}={replicas}"])
                scale_services.append(name)
            if bool(action.get("remove_orphans", False)):
                cmd.append("--remove-orphans")
            cmd.extend(scale_services)
        elif op == "ps":
            cmd.append("ps")
            if services:
                cmd.extend(services)
        elif op == "logs":
            cmd.append("logs")
            if bool(action.get("follow", False)):
                cmd.append("-f")
            tail_raw = action.get("tail")
            if tail_raw is not None:
                try:
                    tail = int(tail_raw) if str(tail_raw).lower() != "all" else "all"
                except Exception:
                    return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                        "error": "invalid_logs_tail",
                        "tail": tail_raw,
                    }
                cmd.extend(["--tail", str(tail)])
            if services:
                cmd.extend(services)
        else:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "error": "unknown_compose_action",
                "action": op,
            }

        if dry_run:
            info: Dict[str, Any] = {"action": op, "command": cmd, "cwd": str(project_dir), "dry_run": True}
            return {"text": "dry_run", "tool_state": self._tool_state()}, 0.05, False, info

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=max(0.1, timeout_s),
                check=False,
            )
        except FileNotFoundError as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "error": "docker_compose_cli_not_found",
                "binary": binary,
                "exception": repr(exc),
            }
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "error": "docker_compose_failed",
                "command": cmd,
                "exception": repr(exc),
            }

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        combined = (stdout + ("\n" if stdout and stderr else "") + stderr).strip()
        clipped, truncated = self._clip_docker_text(combined)
        reward = 0.2 if completed.returncode == 0 else -0.2
        info = {
            "action": op,
            "command": cmd,
            "cwd": str(project_dir),
            "returncode": int(completed.returncode),
            "stdout_chars": len(stdout),
            "stderr_chars": len(stderr),
            "truncated": truncated,
        }
        return {"text": clipped, "tool_state": self._tool_state()}, reward, False, info

    # --------------------------------------------------------------------- #
    # Script runner -------------------------------------------------------- #
    def _is_allowed_script_path(self, path: Path) -> bool:
        if not self._allowed_script_paths:
            return False
        try:
            resolved = path.resolve()
        except Exception:
            return False
        return resolved in self._allowed_script_paths

    def _clip_script_text(self, text: str) -> Tuple[str, bool]:
        raw = str(text or "")
        clipped = raw[: max(0, int(self._max_script_output_chars))]
        return clipped, len(raw) > len(clipped)

    def _handle_run_script(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if os.getenv("BSS_SCRIPT_EXEC_DISABLE") in ("1", "true", "TRUE"):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "script_execution_disabled_by_env"}
        if not self._allow_script_execution:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "script_execution_disabled"}

        path, info = self._normalize_path(action.get("path"))
        if path is None:
            info.setdefault("blocked", info.get("blocked", False))
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, info
        exec_path = path
        overlay_info: Dict[str, Any] = {}
        if self._fs_sandbox is not None:
            try:
                exists = self._fs_sandbox.exists(path)
            except Exception:
                exists = path.exists()
            if not exists:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                    "error": "script_not_found",
                    "path": str(path),
                    "sandboxed": True,
                }
            try:
                _, _, sandbox_path = self._fs_sandbox._map(path)  # type: ignore[attr-defined]
                if sandbox_path.exists() and sandbox_path.is_file():
                    exec_path = sandbox_path
                    overlay_info = {"sandboxed": True, "script_source": "sandbox", "sandbox_path": str(sandbox_path)}
                else:
                    overlay_info = {"sandboxed": True, "script_source": "original"}
            except Exception:
                overlay_info = {"sandboxed": True, "script_source": "unknown"}
            if not exec_path.exists() or not exec_path.is_file():
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                    "error": "script_not_found",
                    "path": str(path),
                    "exec_path": str(exec_path),
                    **overlay_info,
                }
        else:
            if not path.exists() or not path.is_file():
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "script_not_found", "path": str(path)}

        if not self._allowed_script_paths:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "script_allowlist_not_configured"}
        if not self._is_allowed_script_path(path):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "script_not_allowed", "path": str(path)}

        scan_disabled = str(os.getenv("BSS_SCRIPT_SCAN_DISABLE") or "").strip().lower() in {"1", "true", "yes", "on"}
        if not scan_disabled:
            try:
                scan = scan_script_file(exec_path).as_dict(max_findings=12)
            except Exception as exc:
                scan = {"ok": False, "error": "scan_failed", "exception": repr(exc), "path": str(exec_path)}
            if not bool(scan.get("ok", False)):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                    "blocked": True,
                    "reason": "script_failed_security_scan",
                    "path": str(path),
                    "exec_path": str(exec_path),
                    **overlay_info,
                    "scan": scan,
                }

        args_raw = action.get("args", [])
        if args_raw is None:
            args_raw = []
        if not isinstance(args_raw, list):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "script_args_must_be_list", "args": args_raw}
        args = [str(x) for x in args_raw]

        interpreter_raw = str(action.get("interpreter") or "auto").strip().lower()
        ext = path.suffix.lower()
        cmd: List[str]

        if ext == ".py":
            if interpreter_raw not in {"auto", "python"}:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "unsupported_interpreter", "interpreter": interpreter_raw}
            cmd = [sys.executable, str(exec_path), *args]
        elif ext == ".sh":
            if interpreter_raw not in {"auto", "bash", "sh"}:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "unsupported_interpreter", "interpreter": interpreter_raw}
            shell_bin = "sh" if interpreter_raw == "sh" else "bash"
            cmd = [shell_bin, str(exec_path), *args]
        elif ext == ".ps1":
            if interpreter_raw not in {"auto", "powershell", "pwsh"}:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "unsupported_interpreter", "interpreter": interpreter_raw}
            if interpreter_raw == "pwsh" or (interpreter_raw == "auto" and os.name != "nt"):
                ps_bin = "pwsh"
                cmd = [ps_bin, "-NoProfile", "-NonInteractive", "-File", str(exec_path), *args]
            else:
                ps_bin = "powershell"
                cmd = [ps_bin, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(exec_path), *args]
        else:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "unsupported_script_type", "path": str(path), "ext": ext, **overlay_info}

        cwd_raw = action.get("cwd")
        cwd_path = exec_path.parent
        if cwd_raw:
            cwd_path, cwd_info = self._normalize_path(cwd_raw)
            if cwd_path is None:
                cwd_info.setdefault("blocked", cwd_info.get("blocked", False))
                cwd_info.setdefault("context", "cwd")
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, cwd_info
            if not cwd_path.exists() or not cwd_path.is_dir():
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "cwd_not_dir", "cwd": str(cwd_path)}

        env_override = action.get("env") or action.get("environment")
        env = None
        env_keys: List[str] = []
        if env_override is not None:
            if not isinstance(env_override, dict):
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "script_env_must_be_dict"}
            env = dict(os.environ)
            for key, val in env_override.items():
                env[str(key)] = str(val)
                env_keys.append(str(key))

        timeout_s = float(action.get("timeout_s", self._default_timeout_s))
        dry_run = bool(action.get("dry_run", False))
        if dry_run:
            return {"text": "dry_run", "tool_state": self._tool_state()}, 0.05, False, {
                "action": "run_script",
                "path": str(path),
                "exec_path": str(exec_path),
                "command": cmd,
                "cwd": str(cwd_path),
                "env_keys": env_keys,
                "dry_run": True,
                **overlay_info,
            }

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(cwd_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=max(0.1, timeout_s),
                check=False,
            )
        except FileNotFoundError as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "script_interpreter_not_found", "exception": repr(exc)}
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "script_failed", "exception": repr(exc)}

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        combined = (stdout + ("\n" if stdout and stderr else "") + stderr).strip()
        clipped, truncated = self._clip_script_text(combined)
        reward = 0.2 if completed.returncode == 0 else -0.2
        return {"text": clipped, "tool_state": self._tool_state()}, reward, False, {
            "action": "run_script",
            "path": str(path),
            "exec_path": str(exec_path),
            "command": cmd,
            "cwd": str(cwd_path),
            "returncode": int(completed.returncode),
            "stdout_chars": len(stdout),
            "stderr_chars": len(stderr),
            "truncated": truncated,
            **overlay_info,
        }

    # --------------------------------------------------------------------- #
    # Remote tool proxy ---------------------------------------------------- #
    @staticmethod
    def _normalize_remote_endpoint(host: str, port: int) -> str:
        host_str = str(host or "").strip()
        if not host_str:
            host_str = "localhost"
        if host_str.startswith("[") and host_str.endswith("]"):
            host_str = host_str[1:-1]
        port_int = int(port)
        if ":" in host_str and not host_str.startswith("[") and host_str.count(":") > 1:
            return f"[{host_str}]:{port_int}"
        return f"{host_str}:{port_int}"

    @classmethod
    def _parse_remote_endpoint(cls, value: Any) -> Tuple[str, int]:
        if isinstance(value, dict):
            host = value.get("host") or value.get("hostname")
            port = value.get("port")
            if host is None or port is None:
                raise ValueError("endpoint_dict_requires_host_and_port")
            return str(host), int(port)

        raw = str(value or "").strip()
        if raw.startswith("tcp://"):
            raw = raw[len("tcp://") :]
        if raw.startswith("[") and "]" in raw:
            host_part, rest = raw[1:].split("]", 1)
            if not rest.startswith(":"):
                raise ValueError("invalid_endpoint")
            return host_part, int(rest[1:])
        if ":" not in raw:
            raise ValueError("invalid_endpoint")
        host_part, port_part = raw.rsplit(":", 1)
        return host_part, int(port_part)

    @classmethod
    def _normalize_remote_endpoint_value(cls, value: Any) -> Optional[str]:
        try:
            host, port = cls._parse_remote_endpoint(value)
        except Exception:
            return None
        try:
            return cls._normalize_remote_endpoint(host, port)
        except Exception:
            return None

    def _handle_remote_tool(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if os.getenv("BSS_REMOTE_CONTROL_DISABLE") in ("1", "true", "TRUE"):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "remote_control_disabled_by_env"}

        if not self._allow_remote_control:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "remote_control_disabled"}

        if RemoteToolClient is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "remote_tool_unavailable"}

        endpoint_value = action.get("endpoint") or action.get("address")
        if endpoint_value is None and action.get("host") is not None and action.get("port") is not None:
            endpoint_value = {"host": action.get("host"), "port": action.get("port")}

        if endpoint_value is None:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_endpoint"}

        try:
            host, port = self._parse_remote_endpoint(endpoint_value)
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "invalid_endpoint", "exception": repr(exc)}

        endpoint_norm = self._normalize_remote_endpoint(host, port)

        if not self._allowed_remote_endpoints:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "remote_allowlist_not_configured"}
        if endpoint_norm not in self._allowed_remote_endpoints:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "remote_endpoint_not_allowed", "endpoint": endpoint_norm}

        token = action.get("auth_token") or action.get("token") or self._remote_auth_token
        if not token:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "remote_auth_token_not_configured", "endpoint": endpoint_norm}

        method = str(action.get("method") or "step").strip().lower()
        timeout_s = float(action.get("timeout_s", self._remote_default_timeout_s))

        client = RemoteToolClient(host, port, auth_token=str(token), timeout_s=timeout_s)

        if method == "reset":
            try:
                remote_observation = client.reset(timeout_s=timeout_s)
            except RemoteToolAuthError:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "remote_auth_failed", "endpoint": endpoint_norm}
            except (RemoteToolConnectionError, RemoteToolProtocolError) as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "remote_reset_failed", "endpoint": endpoint_norm, "exception": str(exc)}
            except Exception as exc:
                return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "remote_reset_failed", "endpoint": endpoint_norm, "exception": repr(exc)}

            text = remote_observation.get("text", "") if isinstance(remote_observation, dict) else ""
            observation = {
                "text": text,
                "remote_observation": remote_observation,
                "tool_state": self._tool_state(),
            }
            info: Dict[str, Any] = {"endpoint": endpoint_norm, "method": "reset"}
            return observation, 0.05, False, info

        if method != "step":
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "unknown_remote_method", "method": method}

        remote_action = action.get("action") or action.get("payload") or action.get("remote_action")
        if not isinstance(remote_action, dict):
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "missing_remote_action"}

        remote_action_type = str(remote_action.get("type") or "").strip().lower()
        if not remote_action_type:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "remote_action_missing_type"}

        if self._allowed_remote_action_types is not None and remote_action_type not in self._allowed_remote_action_types:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {
                "blocked": True,
                "reason": "remote_action_type_not_allowed",
                "endpoint": endpoint_norm,
                "action_type": remote_action_type,
            }

        try:
            response = client.step(remote_action, timeout_s=timeout_s)
        except RemoteToolAuthError:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"blocked": True, "reason": "remote_auth_failed", "endpoint": endpoint_norm}
        except (RemoteToolConnectionError, RemoteToolProtocolError) as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "remote_step_failed", "endpoint": endpoint_norm, "exception": str(exc)}
        except Exception as exc:
            return {"text": "", "tool_state": self._tool_state()}, -1.0, False, {"error": "remote_step_failed", "endpoint": endpoint_norm, "exception": repr(exc)}

        remote_observation = response.observation
        text = remote_observation.get("text", "") if isinstance(remote_observation, dict) else ""
        observation = {
            "text": text,
            "remote_observation": remote_observation,
            "tool_state": self._tool_state(),
        }
        info = {"endpoint": endpoint_norm, "method": "step", "remote_info": response.info}
        return observation, float(response.reward), bool(response.terminated), info
