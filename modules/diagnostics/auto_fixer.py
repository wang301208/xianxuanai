"""Automatic error analysis + fix-plan generation.

This module is intentionally **opt-in**. It is designed to help agents close the
loop when a task fails by:

1) extracting a structured error report (exception + traceback + context)
2) asking an LLM for a machine-readable fix plan
3) optionally applying the plan (e.g., patching a provided code string) and retrying

The higher-level integration is done by task runners (e.g. `modules.execution.TaskAdapter`)
by passing an `autofix` config in task metadata.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("diagnostics.autofix")

LLMCallable = Callable[[str], str]

_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)?\s*\n(?P<body>[\s\S]*?)\n```", re.MULTILINE)
_HUNK_RE = re.compile(
    r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_len>\d+))?\s+\+(?P<new_start>\d+)(?:,(?P<new_len>\d+))?\s+@@"
)


@dataclass(frozen=True)
class ErrorAnalysis:
    error_type: str
    message: str
    traceback: str
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback,
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class FixPlan:
    """LLM-produced fix plan.

    - `raw` preserves the original model output.
    - `data` is best-effort parsed JSON (if present).
    """

    raw: str
    data: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw,
            "data": dict(self.data or {}),
            "model": self.model,
            "created_at": self.created_at,
        }


class AutoFixFailed(RuntimeError):
    """Raised when autofix ran but could not repair the failing task."""

    def payload(self) -> Dict[str, Any]:
        if self.args and isinstance(self.args[0], dict):
            return dict(self.args[0])
        return {"error": str(self)}


def _is_truthy_env(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _truncate(value: Any, *, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def extract_fenced_block(text: str, *, lang: str) -> Optional[str]:
    """Extract the last fenced block matching `lang` (case-insensitive)."""

    if not text:
        return None
    target = (lang or "").strip().lower()
    found: Optional[str] = None
    for match in _FENCE_RE.finditer(text):
        fence_lang = (match.group("lang") or "").strip().lower()
        if fence_lang == target:
            found = (match.group("body") or "").strip()
    return found


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    fenced = extract_fenced_block(text, lang="json")
    if fenced:
        try:
            parsed = json.loads(fenced)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

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


def apply_unified_diff(original: str, diff_text: str) -> Optional[str]:
    """Apply a unified diff to `original` using exact (non-fuzzy) matching.

    Supports a single-file patch; returns None if it cannot be applied cleanly.
    """

    if not diff_text:
        return None

    original_has_trailing_newline = original.endswith("\n")
    lines = original.splitlines()
    diff_lines = diff_text.splitlines()

    i = 0
    while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
        i += 1
    if i >= len(diff_lines):
        return None

    offset = 0
    while i < len(diff_lines):
        header = diff_lines[i]
        if not header.startswith("@@"):
            i += 1
            continue

        match = _HUNK_RE.match(header)
        if not match:
            return None
        old_start = int(match.group("old_start"))
        start_index = old_start - 1 + offset
        if start_index < 0:
            return None

        i += 1
        idx = start_index
        new_chunk: list[str] = []

        while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
            hline = diff_lines[i]
            if hline.startswith(("diff ", "index ", "--- ", "+++ ")):
                i += 1
                continue
            if not hline:
                # Empty lines can be either context, add, or delete; in unified diff they are prefixed.
                return None

            prefix = hline[0]
            content = hline[1:] if len(hline) > 1 else ""

            if prefix == " ":
                if idx >= len(lines) or lines[idx] != content:
                    return None
                new_chunk.append(lines[idx])
                idx += 1
            elif prefix == "-":
                if idx >= len(lines) or lines[idx] != content:
                    return None
                idx += 1
            elif prefix == "+":
                new_chunk.append(content)
            elif hline.startswith("\\ No newline at end of file"):
                # ignore meta line
                pass
            else:
                return None
            i += 1

        consumed = idx - start_index
        lines = lines[:start_index] + new_chunk + lines[start_index + consumed :]
        offset += len(new_chunk) - consumed

    patched = "\n".join(lines)
    if original_has_trailing_newline:
        patched += "\n"
    return patched


def apply_fix_to_code(code: str, fix_plan: FixPlan) -> Optional[str]:
    """Best-effort code update from a fix plan.

    Supports either:
    - JSON plan with `fix.kind == "replace_code"` and `fix.code`
    - JSON plan with `fix.kind == "code_diff"` and `fix.diff` (unified diff)
    - fenced ```python``` block (full replacement)
    - fenced ```diff``` block (unified diff)
    """

    if not isinstance(code, str) or not code:
        return None

    data = fix_plan.data or {}
    fix = data.get("fix") if isinstance(data.get("fix"), dict) else {}
    kind = str(fix.get("kind") or "").strip().lower()
    if kind == "replace_code" and isinstance(fix.get("code"), str) and fix["code"].strip():
        return str(fix["code"]).rstrip() + ("\n" if code.endswith("\n") else "")
    if kind == "code_diff" and isinstance(fix.get("diff"), str) and fix["diff"].strip():
        return apply_unified_diff(code, str(fix["diff"]))

    replacement = extract_fenced_block(fix_plan.raw, lang="python")
    if replacement:
        return replacement.rstrip() + ("\n" if code.endswith("\n") else "")

    diff_block = extract_fenced_block(fix_plan.raw, lang="diff")
    if diff_block:
        return apply_unified_diff(code, diff_block)

    return None


def extract_retry_kwargs(fix_plan: FixPlan) -> Optional[Dict[str, Any]]:
    data = fix_plan.data or {}
    fix = data.get("fix") if isinstance(data.get("fix"), dict) else {}
    retry_kwargs = fix.get("retry_kwargs")
    if isinstance(retry_kwargs, dict):
        return dict(retry_kwargs)
    return None


class AutoFixer:
    """Generate fix plans for exceptions (optionally via an LLM)."""

    def __init__(
        self,
        *,
        llm: Optional[LLMCallable] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_context_chars: int = 8_000,
    ) -> None:
        self._llm = llm
        self._model = model
        self._temperature = float(temperature)
        self._max_context_chars = int(max_context_chars)

    @property
    def model(self) -> Optional[str]:
        return self._model

    @classmethod
    def from_env(cls) -> "AutoFixer | None":
        if not _is_truthy_env("AUTOFIX_ENABLED"):
            return None
        model = os.getenv("AUTOFIX_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
        temperature = float(os.getenv("AUTOFIX_TEMPERATURE") or 0.0)

        # Avoid accidental network calls when not configured.
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("AZURE_OPENAI_API_KEY"):
            logger.debug("AUTOFIX_ENABLED is set but no API key found; autofix disabled.")
            return None

        llm = _openai_chat_completion(model=model, temperature=temperature)
        return cls(llm=llm, model=model, temperature=temperature)

    def analyze_error(self, error: Exception, context: Dict[str, Any] | None = None) -> ErrorAnalysis:
        context_dict: Dict[str, Any] = dict(context or {})
        tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        return ErrorAnalysis(
            error_type=type(error).__name__,
            message=str(error),
            traceback=tb,
            context=context_dict,
        )

    def generate_fix_plan(self, analysis: ErrorAnalysis) -> Optional[FixPlan]:
        if self._llm is None:
            return None

        context_blob = _truncate(
            json.dumps(analysis.context, ensure_ascii=False, sort_keys=True, default=str),
            max_chars=self._max_context_chars,
        )
        tb_blob = _truncate(analysis.traceback, max_chars=self._max_context_chars)

        prompt = (
            "You are an expert engineer helping an agent self-heal after a failed task.\n"
            "Given the exception and context, produce a SAFE minimal fix plan.\n\n"
            "Output requirements:\n"
            "- Return ONLY a JSON object (no prose), preferably in a ```json``` fenced block.\n"
            "- Schema:\n"
            "  {\n"
            '    \"analysis\": {\"likely_root_cause\": str, \"confidence\": 0..1},\n'
            "    \"fix\": {\n"
            '      \"kind\": \"replace_code\"|\"code_diff\"|\"retry_kwargs\"|\"instructions\"|\"none\",\n'
            "      \"code\": str|null,\n"
            "      \"diff\": str|null,\n"
            "      \"retry_kwargs\": object|null,\n"
            "      \"instructions\": str|null\n"
            "    },\n"
            '    \"safety\": {\"requires_human_review\": bool, \"risk\": \"low\"|\"medium\"|\"high\"}\n'
            "  }\n\n"
            "Exception traceback:\n"
            f"{tb_blob}\n\n"
            "Context (JSON):\n"
            f"{context_blob}\n"
        )

        try:
            raw = (self._llm(prompt) or "").strip()
        except Exception:  # pragma: no cover - best effort
            logger.debug("AutoFix LLM call failed.", exc_info=True)
            return None

        data = extract_json_object(raw)
        return FixPlan(raw=raw, data=data, model=self._model, created_at=time.time())

    def auto_fix_error(self, error: Exception, context: Dict[str, Any] | None = None) -> Optional[FixPlan]:
        analysis = self.analyze_error(error, context=context)
        return self.generate_fix_plan(analysis)


def _openai_chat_completion(*, model: str, temperature: float) -> LLMCallable:
    def _call(prompt: str) -> str:
        from openai import OpenAI  # local import: optional dependency / lazy loading

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You output strict JSON only. No markdown besides optional ```json``` fencing.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content or ""

    return _call


def execute_with_autofix(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
    autofix: Dict[str, Any] | bool | None,
    task_context: Dict[str, Any] | None = None,
) -> Any:
    """Execute a callable with an autofix retry loop (opt-in).

    Expected autofix config keys (all optional):
    - enabled: bool (default True)
    - max_attempts: int (default 1)  # number of *fix* attempts after the initial failure
    - code_arg_index: int           # where a code string lives in args
    - code_kwarg: str               # where a code string lives in kwargs
    - context: dict                 # extra context to pass to the AutoFixer
    - llm: callable(prompt)->str    # optional override (useful for tests)
    - model, temperature            # optional LLM params
    """

    if not autofix:
        return func(*args, **kwargs)

    config: Dict[str, Any]
    if isinstance(autofix, dict):
        config = dict(autofix)
    else:
        config = {}

    if not bool(config.get("enabled", True)):
        return func(*args, **kwargs)

    max_attempts = int(config.get("max_attempts", 1))
    max_attempts = max(0, max_attempts)

    llm = config.get("llm") if callable(config.get("llm")) else None
    model = config.get("model") or os.getenv("AUTOFIX_MODEL") or os.getenv("OPENAI_MODEL") or None
    temperature = float(config.get("temperature") or os.getenv("AUTOFIX_TEMPERATURE") or 0.0)

    autofixer: AutoFixer | None
    if llm is not None:
        autofixer = AutoFixer(llm=llm, model=model, temperature=temperature)
    else:
        # Fall back to env-driven configuration (and keep it disabled unless explicitly enabled).
        autofixer = AutoFixer.from_env()

    if autofixer is None or max_attempts <= 0:
        return func(*args, **kwargs)

    code_arg_index = config.get("code_arg_index")
    code_kwarg = config.get("code_kwarg")
    extra_context = config.get("context") if isinstance(config.get("context"), dict) else {}

    current_args = tuple(args)
    current_kwargs = dict(kwargs)
    fix_history: list[Dict[str, Any]] = []
    last_exc: Exception | None = None
    last_analysis: ErrorAnalysis | None = None

    def _get_code() -> Optional[str]:
        nonlocal current_args, current_kwargs
        if code_kwarg and isinstance(current_kwargs.get(code_kwarg), str):
            return current_kwargs.get(code_kwarg)
        if isinstance(code_arg_index, int) and 0 <= code_arg_index < len(current_args):
            if isinstance(current_args[code_arg_index], str):
                return current_args[code_arg_index]
        return None

    def _set_code(new_code: str) -> bool:
        nonlocal current_args, current_kwargs
        updated = False
        if code_kwarg:
            if isinstance(current_kwargs.get(code_kwarg), str):
                current_kwargs[code_kwarg] = new_code
                updated = True
        if isinstance(code_arg_index, int) and 0 <= code_arg_index < len(current_args):
            if isinstance(current_args[code_arg_index], str):
                args_list = list(current_args)
                args_list[code_arg_index] = new_code
                current_args = tuple(args_list)
                updated = True
        return updated

    for attempt in range(max_attempts + 1):
        try:
            result = func(*current_args, **current_kwargs)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            return result
        except Exception as exc:
            last_exc = exc
            ctx: Dict[str, Any] = dict(extra_context)
            ctx.update(
                {
                    "task": dict(task_context or {}),
                    "attempt": attempt,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
            )
            code = _get_code()
            if code is not None:
                ctx.setdefault("code", _truncate(code, max_chars=12_000))

            last_analysis = autofixer.analyze_error(exc, context=ctx)
            if attempt >= max_attempts:
                break

            plan = autofixer.generate_fix_plan(last_analysis)
            if plan is None:
                raise
            fix_history.append(plan.to_dict())

            updated = False
            retry_kwargs = extract_retry_kwargs(plan)
            if retry_kwargs:
                current_kwargs.update(retry_kwargs)
                updated = True

            if code is not None:
                patched = apply_fix_to_code(code, plan)
                if isinstance(patched, str) and patched and patched != code:
                    updated = _set_code(patched) or updated

            if not updated:
                break

    if not fix_history:
        raise last_exc  # type: ignore[misc]

    payload = {
        "task": dict(task_context or {}),
        "original_exception": repr(last_exc) if last_exc is not None else "",
        "analysis": last_analysis.to_dict() if last_analysis is not None else {},
        "fix_history": list(fix_history),
    }
    raise AutoFixFailed(payload) from last_exc
