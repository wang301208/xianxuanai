from __future__ import annotations

import pytest

from modules.diagnostics.auto_fixer import (
    AutoFixFailed,
    apply_unified_diff,
    execute_with_autofix,
)


def test_apply_unified_diff_applies_hunk() -> None:
    original = "a\nb\nc\n"
    diff = "@@ -1,3 +1,3 @@\n a\n-b\n+d\n c\n"

    patched = apply_unified_diff(original, diff)

    assert patched == "a\nd\nc\n"


def test_execute_with_autofix_retries_with_retry_kwargs() -> None:
    calls: list[str] = []

    def llm_stub(prompt: str) -> str:
        calls.append(prompt)
        return (
            "```json\n"
            '{\n  "analysis": {"likely_root_cause": "division by zero", "confidence": 0.9},\n'
            '  "fix": {"kind": "retry_kwargs", "retry_kwargs": {"b": 1}},\n'
            '  "safety": {"requires_human_review": false, "risk": "low"}\n'
            "}\n"
            "```"
        )

    def divide(a: float, *, b: float) -> float:
        return a / b

    result = execute_with_autofix(
        divide,
        (2.0,),
        {"b": 0.0},
        {"enabled": True, "max_attempts": 1, "llm": llm_stub},
        {"name": "divide"},
    )

    assert result == 2.0
    assert len(calls) == 1


def test_execute_with_autofix_patches_code_arg() -> None:
    def llm_stub(_: str) -> str:
        return (
            "```json\n"
            '{\n  "analysis": {"likely_root_cause": "NameError", "confidence": 0.8},\n'
            '  "fix": {"kind": "replace_code", "code": "value = 1 + 2"},\n'
            '  "safety": {"requires_human_review": false, "risk": "low"}\n'
            "}\n"
            "```"
        )

    def run_code(code: str) -> int:
        scope: dict[str, object] = {}
        exec(code, {}, scope)
        return int(scope["value"])  # type: ignore[arg-type]

    bad_code = "value = 1 + unknown\n"
    result = execute_with_autofix(
        run_code,
        (bad_code,),
        {},
        {"enabled": True, "max_attempts": 1, "llm": llm_stub, "code_arg_index": 0},
        {"name": "run_code"},
    )

    assert result == 3


def test_execute_with_autofix_does_not_run_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOFIX_ENABLED", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    def boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        execute_with_autofix(boom, (), {}, {"enabled": True, "max_attempts": 1}, {"name": "boom"})


def test_autofix_failed_payload_roundtrips() -> None:
    err = AutoFixFailed({"hello": "world"})
    assert err.payload()["hello"] == "world"

