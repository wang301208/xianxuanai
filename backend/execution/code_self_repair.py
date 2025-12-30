from __future__ import annotations

"""Opt-in code self-repair workflow (proposal -> validate -> human review).

This module is intentionally conservative and disabled by default.

High-level flow:
1) Detect a repeated failure (optional) OR accept an explicit `self_repair.request`
2) (Optional) Ask an LLM to propose a minimal unified diff patch
3) Validate the patch in an isolated git worktree and run tests
4) Publish results for humans/agents to review and merge

By default, the manager NEVER auto-merges; it only prepares a branch/worktree.
"""

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

try:  # optional in some deployments
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EventBus = None  # type: ignore

from .task_manager import TaskManager, TaskPriority

logger = logging.getLogger(__name__)

LLMCallable = Callable[[str], str]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _truncate(text: Any, *, max_chars: int) -> str:
    blob = str(text or "")
    if max_chars <= 0 or len(blob) <= max_chars:
        return blob
    return blob[: max(0, max_chars - 3)] + "..."


def _parse_prefixes(value: str | None, *, default: Sequence[str]) -> tuple[str, ...]:
    raw = str(value or "").strip()
    if not raw:
        return tuple(str(p) for p in default if str(p))
    parts = [p.strip() for p in raw.split(",")]
    out = []
    for part in parts:
        token = part.replace("\\", "/").strip().lstrip("/")
        if not token:
            continue
        if not token.endswith("/"):
            token += "/"
        out.append(token)
    return tuple(out) if out else tuple(str(p) for p in default if str(p))


def _split_command(value: str | Sequence[str] | None, *, default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x)]
    text = str(value or "").strip()
    if not text:
        return list(default)
    try:
        return [str(x) for x in shlex.split(text) if str(x)]
    except Exception:
        return list(default)


_DIFF_GIT_RE = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+?)\s*$")
_DIFF_FILE_RE = re.compile(r"^(?P<kind>---|\+\+\+)\s+(?P<path>.+?)\s*$")
_TRACEBACK_FILE_RE = re.compile(r'^\s*File\s+"(?P<path>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<func>.+?)\s*$')


def _normalise_patch_path(raw: str) -> str | None:
    path = str(raw or "").strip()
    if not path or path == "/dev/null":
        return None
    path = path.strip('"').replace("\\", "/")
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    if ":" in path.split("/", 1)[0]:
        return None
    posix = PurePosixPath(path)
    if posix.is_absolute():
        return None
    parts = posix.parts
    if any(part in {"..", ""} for part in parts):
        return None
    norm = str(posix)
    if norm.startswith("../") or "/../" in norm or norm == "..":
        return None
    return norm


def extract_patch_paths(diff_text: str) -> tuple[str, ...]:
    paths: set[str] = set()
    for line in str(diff_text or "").splitlines():
        m = _DIFF_GIT_RE.match(line)
        if m:
            norm = _normalise_patch_path(m.group("b"))
            if norm:
                paths.add(norm)
            continue
        m = _DIFF_FILE_RE.match(line)
        if m:
            path = m.group("path")
            if path.startswith("a/") or path.startswith("b/"):
                path = path[2:]
            norm = _normalise_patch_path(path)
            if norm:
                paths.add(norm)
    return tuple(sorted(paths))


def validate_patch(
    diff_text: str,
    *,
    max_chars: int,
    max_files: int,
    allowed_prefixes: Sequence[str],
    protected_prefixes: Sequence[str],
) -> tuple[bool, str, tuple[str, ...]]:
    if not diff_text or not str(diff_text).strip():
        return False, "empty_patch", ()
    if int(max_chars) > 0 and len(diff_text) > int(max_chars):
        return False, "patch_too_large", ()
    paths = extract_patch_paths(diff_text)
    if not paths:
        return False, "no_paths_detected", ()
    if int(max_files) > 0 and len(paths) > int(max_files):
        return False, "too_many_files", paths

    allow = tuple(allowed_prefixes)
    protect = tuple(protected_prefixes)
    for path in paths:
        norm = path.replace("\\", "/").lstrip("/")
        if not norm or norm.startswith(".git/"):
            return False, "disallowed_path", paths
        if protect and any(norm.startswith(prefix) for prefix in protect):
            return False, "protected_path", paths
        if allow and not any(norm.startswith(prefix) for prefix in allow):
            return False, "outside_allowed_roots", paths
    return True, "ok", paths


def _extract_traceback_frames(traceback_text: str) -> list[tuple[str, int, str]]:
    frames: list[tuple[str, int, str]] = []
    for line in str(traceback_text or "").splitlines():
        match = _TRACEBACK_FILE_RE.match(line)
        if not match:
            continue
        path = match.group("path")
        try:
            lineno = int(match.group("line"))
        except Exception:
            continue
        func = match.group("func")
        frames.append((path, lineno, func))
    return frames


def _pick_repo_frame(
    frames: Sequence[tuple[str, int, str]],
    *,
    repo_root: Path,
    protected_prefixes: Sequence[str],
) -> tuple[str, int] | None:
    root = Path(repo_root).resolve()
    protect = tuple(p.replace("\\", "/").lstrip("/") for p in protected_prefixes)
    for raw_path, lineno, _func in frames:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        else:
            try:
                candidate = candidate.resolve()
            except Exception:
                continue
        try:
            rel = candidate.relative_to(root)
        except Exception:
            continue
        rel_posix = str(rel).replace("\\", "/")
        if protect and any(rel_posix.startswith(prefix) for prefix in protect):
            continue
        return rel_posix, int(lineno)
    return None


def _render_snippet(path: Path, *, line: int, context: int = 40, max_chars: int = 10_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    target = max(1, int(line))
    start = max(1, target - int(context))
    end = min(len(lines), target + int(context))
    out: list[str] = []
    for i in range(start, end + 1):
        marker = ">>" if i == target else "  "
        out.append(f"{marker}{i:4d}: {lines[i - 1]}")
    return _truncate("\n".join(out), max_chars=max_chars)


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidate = (text or "").strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _openai_chat_completion(*, model: str, temperature: float) -> LLMCallable:
    def _call(prompt: str) -> str:
        from openai import OpenAI  # local import: optional dependency / lazy loading

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return strict JSON only. No prose."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    return _call


@dataclass(frozen=True)
class PatchProposal:
    proposal_id: str
    metric: str
    summary: str
    diff: str
    risk: str = "low"  # low|medium|high
    requires_human_review: bool = True
    paths: tuple[str, ...] = ()
    source: str = "self_repair"


@dataclass(frozen=True)
class PatchValidationResult:
    proposal_id: str
    metric: str
    success: bool
    branch: str | None = None
    worktree: str | None = None
    commit: str | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    reason: str = ""


class PatchExecutor:
    """Apply a patch in an isolated sandbox and run tests."""

    def validate(self, proposal: PatchProposal) -> PatchValidationResult:  # pragma: no cover - interface
        raise NotImplementedError


class GitWorktreeExecutor(PatchExecutor):
    """Validate patches in a dedicated git worktree and branch."""

    def __init__(
        self,
        *,
        repo_root: str | os.PathLike[str] | None = None,
        worktree_base: str | os.PathLike[str] | None = None,
        test_command: str | Sequence[str] | None = None,
        timeout_secs: float | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self._repo_root = Path(repo_root or os.getenv("WORKSPACE_ROOT") or Path.cwd())
        self._worktree_base = Path(worktree_base or (self._repo_root / ".self_repair" / "worktrees"))
        self._test_command = _split_command(
            test_command or os.getenv("SELF_REPAIR_TEST_COMMAND"),
            default=[os.getenv("PYTHON", "python"), "-m", "pytest", "-q", "tests/unit", "tests/integration"],
        )
        self._timeout = _env_float("SELF_REPAIR_TEST_TIMEOUT_SECS", 600.0) if timeout_secs is None else float(timeout_secs)
        self._logger = logger_ or logger

    def validate(self, proposal: PatchProposal) -> PatchValidationResult:
        repo_root = self._repo_root.resolve()
        proposal_id = proposal.proposal_id
        branch = f"self-repair/{proposal_id}"
        worktree_path = (self._worktree_base / proposal_id).resolve()
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        def _run_git(args: Sequence[str], *, cwd: Path, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", *args],
                cwd=str(cwd),
                input=input_text,
                text=True,
                capture_output=True,
                check=False,
            )

        # Create worktree/branch
        if worktree_path.exists():
            return PatchValidationResult(
                proposal_id=proposal_id,
                metric=proposal.metric,
                success=False,
                branch=branch,
                worktree=str(worktree_path),
                reason="worktree_path_exists",
            )

        created = _run_git(["worktree", "add", "-b", branch, str(worktree_path)], cwd=repo_root)
        if created.returncode != 0:
            return PatchValidationResult(
                proposal_id=proposal_id,
                metric=proposal.metric,
                success=False,
                branch=branch,
                worktree=str(worktree_path),
                stderr=_truncate(created.stderr, max_chars=8_000),
                reason="worktree_add_failed",
            )

        # Apply patch
        applied = _run_git(["apply", "--whitespace=nowarn", "-"], cwd=worktree_path, input_text=proposal.diff)
        if applied.returncode != 0:
            return PatchValidationResult(
                proposal_id=proposal_id,
                metric=proposal.metric,
                success=False,
                branch=branch,
                worktree=str(worktree_path),
                stdout=_truncate(applied.stdout, max_chars=8_000),
                stderr=_truncate(applied.stderr, max_chars=8_000),
                reason="git_apply_failed",
            )

        # Configure git identity locally so commits succeed in CI/dev environments.
        _run_git(["config", "user.name", "self-repair"], cwd=worktree_path)
        _run_git(["config", "user.email", "self-repair@local"], cwd=worktree_path)

        status = _run_git(["status", "--porcelain"], cwd=worktree_path)
        if not status.stdout.strip():
            return PatchValidationResult(
                proposal_id=proposal_id,
                metric=proposal.metric,
                success=False,
                branch=branch,
                worktree=str(worktree_path),
                reason="no_changes_after_patch",
            )

        _run_git(["add", "-A"], cwd=worktree_path)
        message = f"self-repair: {proposal.summary}".strip()
        if len(message) > 72:
            message = message[:69] + "..."
        committed = _run_git(["commit", "-m", message], cwd=worktree_path)
        if committed.returncode != 0:
            return PatchValidationResult(
                proposal_id=proposal_id,
                metric=proposal.metric,
                success=False,
                branch=branch,
                worktree=str(worktree_path),
                stdout=_truncate(committed.stdout, max_chars=8_000),
                stderr=_truncate(committed.stderr, max_chars=8_000),
                reason="git_commit_failed",
            )
        rev = _run_git(["rev-parse", "HEAD"], cwd=worktree_path)
        commit_hash = rev.stdout.strip() if rev.returncode == 0 else None

        # Run tests
        try:
            proc = subprocess.run(
                list(self._test_command),
                cwd=str(worktree_path),
                capture_output=True,
                text=True,
                timeout=max(1.0, float(self._timeout)),
                check=False,
            )
        except Exception as exc:  # pragma: no cover - OS/process failures
            return PatchValidationResult(
                proposal_id=proposal_id,
                metric=proposal.metric,
                success=False,
                branch=branch,
                worktree=str(worktree_path),
                commit=commit_hash,
                stderr=_truncate(repr(exc), max_chars=8_000),
                reason="test_run_failed",
            )

        return PatchValidationResult(
            proposal_id=proposal_id,
            metric=proposal.metric,
            success=proc.returncode == 0,
            branch=branch,
            worktree=str(worktree_path),
            commit=commit_hash,
            returncode=int(proc.returncode),
            stdout=_truncate(proc.stdout, max_chars=20_000),
            stderr=_truncate(proc.stderr, max_chars=20_000),
            reason="ok" if proc.returncode == 0 else "tests_failed",
        )


class CodeSelfRepairManager:
    """Event-driven self-repair scaffold for low-risk code fixes."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        task_manager: TaskManager,
        executor: PatchExecutor | None = None,
        llm: LLMCallable | None = None,
        enabled: bool | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")
        self._bus = event_bus
        self._tasks = task_manager
        self._logger = logger_ or logger

        self._enabled = _env_bool("SELF_REPAIR_ENABLED", False) if enabled is None else bool(enabled)
        self._on_task_failure = _env_bool("SELF_REPAIR_ON_TASK_FAILURE", False)
        self._auto_validate = _env_bool("SELF_REPAIR_AUTO_VALIDATE", False)
        self._cooldown_secs = _env_float("SELF_REPAIR_COOLDOWN_SECS", 900.0)
        self._window_secs = _env_float("SELF_REPAIR_WINDOW_SECS", 300.0)
        self._min_repeats = _env_int("SELF_REPAIR_MIN_REPEATS", 2)

        self._allowed_prefixes = _parse_prefixes(
            os.getenv("SELF_REPAIR_ALLOWED_PREFIXES"),
            default=("backend/", "BrainSimulationSystem/", "modules/", "common/", "tests/"),
        )
        self._protected_prefixes = _parse_prefixes(
            os.getenv("SELF_REPAIR_PROTECTED_PREFIXES"),
            default=(".git/", ".github/", "third_party/", "docker/", "deploy/"),
        )
        self._max_patch_chars = _env_int("SELF_REPAIR_MAX_PATCH_CHARS", 40_000)
        self._max_files = _env_int("SELF_REPAIR_MAX_FILES", 2)
        self._max_event_chars = _env_int("SELF_REPAIR_MAX_EVENT_CHARS", 8_000)

        self._llm = llm or self._maybe_default_llm()
        self._llm_model = os.getenv("SELF_REPAIR_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self._llm_temperature = _env_float("SELF_REPAIR_TEMPERATURE", 0.0)

        self._executor = executor or GitWorktreeExecutor(logger_=self._logger)
        self._failures: Dict[str, list[float]] = {}
        self._last_trigger_ts: float | None = None
        self._inflight: set[str] = set()

        self._subscriptions: list[Callable[[], None]] = [self._bus.subscribe("self_repair.request", self._on_request)]
        if self._on_task_failure:
            self._subscriptions.append(self._bus.subscribe("task_manager.task_completed", self._on_task_completed))

    def close(self) -> None:
        subs = list(self._subscriptions)
        self._subscriptions.clear()
        for cancel in subs:
            try:
                cancel()
            except Exception:
                continue

    # ------------------------------------------------------------------ handlers
    async def _on_request(self, event: Dict[str, Any]) -> None:
        if not self._enabled or not isinstance(event, Mapping):
            return
        metric = str(event.get("metric") or "code_health").strip() or "code_health"
        summary = str(event.get("summary") or event.get("reason") or "Self-repair proposal").strip()
        diff = event.get("diff") or event.get("patch")
        if not isinstance(diff, str) or not diff.strip():
            return

        proposal_id = str(event.get("proposal_id") or "").strip() or uuid.uuid4().hex
        ok, reason, paths = validate_patch(
            diff,
            max_chars=self._max_patch_chars,
            max_files=self._max_files,
            allowed_prefixes=self._allowed_prefixes,
            protected_prefixes=self._protected_prefixes,
        )
        if not ok:
            self._publish(
                "self_repair.patch_rejected",
                {
                    "time": time.time(),
                    "metric": metric,
                    "proposal_id": proposal_id,
                    "reason": reason,
                    "paths": list(paths),
                },
            )
            return

        proposal = PatchProposal(
            proposal_id=proposal_id,
            metric=metric,
            summary=summary,
            diff=diff,
            risk=str(event.get("risk") or "low"),
            requires_human_review=bool(event.get("requires_human_review", True)),
            paths=paths,
            source=str(event.get("source") or "self_repair.request"),
        )
        self._publish(
            "self_repair.patch_proposed",
            {
                "time": time.time(),
                "metric": metric,
                "proposal_id": proposal_id,
                "summary": summary,
                "paths": list(paths),
                "risk": proposal.risk,
                "requires_human_review": bool(proposal.requires_human_review),
                "diff": _truncate(diff, max_chars=self._max_event_chars),
                "source": proposal.source,
            },
        )

        if not self._auto_validate:
            return
        if proposal_id in self._inflight:
            return
        self._inflight.add(proposal_id)
        try:
            self._tasks.submit(
                self._validate_and_publish,
                proposal,
                priority=TaskPriority.LOW,
                category="self_repair",
                deadline=time.time() + 1800.0,
                metadata={"proposal_id": proposal_id, "metric": metric},
            )
        except Exception:
            self._inflight.discard(proposal_id)

    async def _on_task_completed(self, event: Dict[str, Any]) -> None:
        if not self._enabled or not isinstance(event, Mapping):
            return
        if str(event.get("status") or "").lower() == "completed":
            return
        if not self._llm:
            return
        if self._cooldown_active(time.time()):
            return
        traceback_text = ""
        autofix = event.get("autofix")
        if isinstance(autofix, Mapping):
            analysis = autofix.get("analysis")
            if isinstance(analysis, Mapping):
                traceback_text = str(analysis.get("traceback") or "")
        if not traceback_text:
            return

        signature = str(event.get("error") or event.get("name") or "failure").strip()
        now = time.time()
        window = max(1.0, float(self._window_secs))
        timestamps = self._failures.setdefault(signature, [])
        timestamps.append(float(now))
        self._failures[signature] = [t for t in timestamps if (now - t) <= window]
        if len(self._failures[signature]) < max(1, int(self._min_repeats)):
            return

        self._last_trigger_ts = float(now)
        metric = "code_health"
        payload = self._generate_patch_from_failure(event, traceback_text=traceback_text)
        if not payload:
            return
        try:
            self._bus.publish("self_repair.request", payload)
        except Exception:
            pass

    # ------------------------------------------------------------------ internals
    def _cooldown_active(self, now: float) -> bool:
        if self._cooldown_secs <= 0:
            return False
        if self._last_trigger_ts is None:
            return False
        return (float(now) - float(self._last_trigger_ts)) < float(self._cooldown_secs)

    def _maybe_default_llm(self) -> LLMCallable | None:
        if not _env_bool("SELF_REPAIR_LLM_ENABLED", False):
            return None
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("AZURE_OPENAI_API_KEY"):
            return None
        return _openai_chat_completion(model=os.getenv("SELF_REPAIR_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini", temperature=_env_float("SELF_REPAIR_TEMPERATURE", 0.0))

    def _generate_patch_from_failure(self, event: Mapping[str, Any], *, traceback_text: str) -> Dict[str, Any] | None:
        if not self._llm:
            return None
        repo_root = Path(os.getenv("WORKSPACE_ROOT") or Path.cwd())
        frames = _extract_traceback_frames(traceback_text)
        picked = _pick_repo_frame(frames, repo_root=repo_root, protected_prefixes=self._protected_prefixes)
        if picked is None:
            return None
        rel_path, lineno = picked
        snippet = _render_snippet(repo_root / rel_path, line=lineno, context=40, max_chars=8_000)
        if not snippet:
            return None

        error = _truncate(event.get("error") or "", max_chars=800)
        prompt = (
            "You are a safe self-repair module for a Python repository.\n"
            "Given a repeated failure and a code snippet, propose the MINIMAL patch to fix it.\n"
            "Return STRICT JSON only.\n\n"
            "Constraints:\n"
            f"- ONLY modify file: {rel_path}\n"
            "- Use a unified diff with --- a/<path> and +++ b/<path> headers.\n"
            "- Keep changes small and low-risk; no new dependencies.\n"
            "- If you cannot propose a safe patch, return diff as an empty string.\n\n"
            "Schema:\n"
            "{\n"
            '  \"summary\": str,\n'
            '  \"risk\": \"low\"|\"medium\"|\"high\",\n'
            '  \"requires_human_review\": bool,\n'
            '  \"diff\": str\n'
            "}\n\n"
            f"Error: {error}\n\n"
            f"Traceback (truncated):\n{_truncate(traceback_text, max_chars=2000)}\n\n"
            f"File: {rel_path}\n"
            f"Line: {lineno}\n"
            f"Code:\n{snippet}\n"
        )
        raw = ""
        data: Dict[str, Any] | None = None
        try:
            raw = self._llm(prompt)
            data = _parse_json_object(raw) or None
        except Exception:
            self._logger.debug("Self-repair LLM call failed", exc_info=True)
            return None
        if data is None:
            return None
        diff = data.get("diff")
        if not isinstance(diff, str):
            return None
        summary = str(data.get("summary") or "LLM patch proposal").strip()
        risk = str(data.get("risk") or "low").strip().lower()
        requires_review = bool(data.get("requires_human_review", True))

        if not diff.strip():
            self._publish(
                "self_repair.no_patch",
                {
                    "time": time.time(),
                    "summary": summary,
                    "risk": risk,
                    "requires_human_review": requires_review,
                    "file": rel_path,
                    "line": lineno,
                    "source": "self_repair.llm",
                },
            )
            return None

        return {
            "time": time.time(),
            "metric": "code_health",
            "proposal_id": uuid.uuid4().hex,
            "summary": summary,
            "risk": risk,
            "requires_human_review": requires_review,
            "diff": diff,
            "source": "self_repair.llm",
        }

    def _validate_and_publish(self, proposal: PatchProposal) -> None:
        proposal_id = proposal.proposal_id
        try:
            result = self._executor.validate(proposal)
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.debug("Self-repair validation failed", exc_info=True)
            result = PatchValidationResult(
                proposal_id=proposal_id,
                metric=proposal.metric,
                success=False,
                stderr=_truncate(repr(exc), max_chars=8_000),
                reason="validation_exception",
            )
        finally:
            self._inflight.discard(proposal_id)

        self._publish(
            "self_repair.patch_validated",
            {
                "time": time.time(),
                "metric": result.metric,
                "proposal_id": result.proposal_id,
                "success": bool(result.success),
                "branch": result.branch,
                "worktree": result.worktree,
                "commit": result.commit,
                "returncode": result.returncode,
                "stdout": _truncate(result.stdout, max_chars=self._max_event_chars),
                "stderr": _truncate(result.stderr, max_chars=self._max_event_chars),
                "reason": result.reason,
            },
        )

        if result.success:
            self._publish(
                "self_repair.review_required",
                {
                    "time": time.time(),
                    "metric": result.metric,
                    "proposal_id": result.proposal_id,
                    "branch": result.branch,
                    "commit": result.commit,
                    "message": "Patch validated. Please review and merge manually.",
                },
            )

    def _publish(self, topic: str, event: Dict[str, Any]) -> None:
        try:
            self._bus.publish(topic, event)
        except Exception:
            pass


__all__ = [
    "CodeSelfRepairManager",
    "GitWorktreeExecutor",
    "PatchExecutor",
    "PatchProposal",
    "PatchValidationResult",
    "extract_patch_paths",
    "validate_patch",
]

