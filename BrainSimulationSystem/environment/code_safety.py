"""Lightweight static safety scanning for untrusted code snippets/scripts.

This module is intentionally dependency-free (stdlib only). It performs a
best-effort scan for *obviously dangerous* operations before code is executed
via tools like ``run_script``.

It is not a substitute for a full sandbox, but it reduces risk by blocking
common primitives used by malicious snippets (e.g. shell execution, recursive
deletes, dynamic eval/exec).
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_SEVERITY_ORDER: Dict[str, int] = {"low": 10, "medium": 20, "high": 30, "critical": 40}


def _max_severity(findings: Sequence["CodeScanFinding"]) -> str:
    best = "low"
    best_val = 0
    for finding in findings:
        sev = str(getattr(finding, "severity", "low") or "low").lower()
        val = _SEVERITY_ORDER.get(sev, 0)
        if val > best_val:
            best = sev
            best_val = val
    return best


def _is_truthy_env(value: str | None) -> bool:
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CodeScanFinding:
    severity: str
    rule: str
    message: str
    line: Optional[int] = None
    col: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "severity": str(self.severity),
            "rule": str(self.rule),
            "message": str(self.message),
        }
        if self.line is not None:
            payload["line"] = int(self.line)
        if self.col is not None:
            payload["col"] = int(self.col)
        return payload


@dataclass(frozen=True)
class CodeScanResult:
    ok: bool
    language: str
    path: str
    sha256: str
    scanned_chars: int
    max_severity: str
    findings: List[CodeScanFinding]

    def as_dict(self, *, max_findings: int = 12) -> Dict[str, Any]:
        limit = max(0, int(max_findings))
        items = self.findings[:limit] if limit else list(self.findings)
        return {
            "ok": bool(self.ok),
            "language": str(self.language),
            "path": str(self.path),
            "sha256": str(self.sha256),
            "scanned_chars": int(self.scanned_chars),
            "max_severity": str(self.max_severity),
            "findings": [f.as_dict() for f in items],
            "findings_truncated": bool(limit and len(self.findings) > limit),
        }


_PY_DANGEROUS_CALLS: Dict[str, Tuple[str, str, str]] = {
    # Shell / process execution.
    "os.system": ("critical", "python.os.system", "Shell execution via os.system"),
    "os.popen": ("critical", "python.os.popen", "Shell execution via os.popen"),
    "subprocess.Popen": ("critical", "python.subprocess.popen", "Process spawn via subprocess.Popen"),
    "subprocess.run": ("critical", "python.subprocess.run", "Process spawn via subprocess.run"),
    "subprocess.call": ("critical", "python.subprocess.call", "Process spawn via subprocess.call"),
    "subprocess.check_call": ("critical", "python.subprocess.check_call", "Process spawn via subprocess.check_call"),
    "subprocess.check_output": ("critical", "python.subprocess.check_output", "Process spawn via subprocess.check_output"),
    # Dynamic code execution.
    "eval": ("critical", "python.eval", "Dynamic code execution via eval"),
    "exec": ("critical", "python.exec", "Dynamic code execution via exec"),
    "compile": ("high", "python.compile", "Dynamic code compilation via compile"),
    "__import__": ("high", "python.__import__", "Dynamic import via __import__"),
    # Destructive filesystem operations.
    "shutil.rmtree": ("high", "python.shutil.rmtree", "Recursive delete via shutil.rmtree"),
    "os.remove": ("high", "python.os.remove", "File deletion via os.remove"),
    "os.unlink": ("high", "python.os.unlink", "File deletion via os.unlink"),
    "os.rmdir": ("high", "python.os.rmdir", "Directory deletion via os.rmdir"),
}

_PY_DANGEROUS_METHODS: Dict[str, Tuple[str, str, str]] = {
    # Methods that are commonly destructive even when receiver type is unknown.
    "unlink": ("high", "python.method.unlink", "File deletion via .unlink()"),
    "rmdir": ("high", "python.method.rmdir", "Directory deletion via .rmdir()"),
    "rmtree": ("high", "python.method.rmtree", "Recursive delete via .rmtree()"),
}

_PS1_PATTERNS: Sequence[Tuple[re.Pattern[str], Tuple[str, str, str]]] = (
    (re.compile(r"\b(?:iex|invoke-expression)\b", flags=re.IGNORECASE), ("critical", "ps1.iex", "Dynamic eval via Invoke-Expression/IEX")),
    (re.compile(r"\bremove-item\b.*\b(-recurse|-force)\b", flags=re.IGNORECASE), ("high", "ps1.remove_item", "Destructive delete via Remove-Item -Recurse/-Force")),
    (re.compile(r"\bstart-process\b", flags=re.IGNORECASE), ("high", "ps1.start_process", "Process spawn via Start-Process")),
)

_SH_PATTERNS: Sequence[Tuple[re.Pattern[str], Tuple[str, str, str]]] = (
    (re.compile(r"\brm\s+-rf\b", flags=re.IGNORECASE), ("critical", "sh.rm_rf", "Destructive delete via rm -rf")),
    (re.compile(r"\b(?:curl|wget)\b[^\n]*\|\s*(?:bash|sh)\b", flags=re.IGNORECASE), ("critical", "sh.curl_pipe_sh", "Remote code execution via curl/wget | sh")),
    (re.compile(r"\bmkfs\b", flags=re.IGNORECASE), ("critical", "sh.mkfs", "Filesystem formatting via mkfs")),
)

_BAT_PATTERNS: Sequence[Tuple[re.Pattern[str], Tuple[str, str, str]]] = (
    (re.compile(r"\bdel\b\s+/[fsq]", flags=re.IGNORECASE), ("high", "bat.del_force", "Destructive delete via del /f|/s|/q")),
    (re.compile(r"\brmdir\b\s+/s\s+/q", flags=re.IGNORECASE), ("high", "bat.rmdir_sq", "Destructive delete via rmdir /s /q")),
    (re.compile(r"\bformat\b", flags=re.IGNORECASE), ("critical", "bat.format", "Disk format via format")),
)


def scan_script_file(path: Path, *, max_chars: int = 200_000) -> CodeScanResult:
    resolved = Path(path).resolve()
    try:
        raw = resolved.read_bytes()
    except Exception:
        raw = b""
    if max_chars > 0 and len(raw) > max_chars:
        raw = raw[:max_chars]
    text = raw.decode("utf-8", errors="replace")
    sha256 = hashlib.sha256(raw).hexdigest()

    suffix = resolved.suffix.lower()
    language = {
        ".py": "python",
        ".ps1": "powershell",
        ".sh": "shell",
        ".bat": "batch",
        ".cmd": "batch",
    }.get(suffix, suffix.lstrip(".") or "unknown")

    findings: List[CodeScanFinding] = []
    if suffix == ".py":
        findings.extend(_scan_python(text))
    elif suffix == ".ps1":
        findings.extend(_scan_regex(text, _PS1_PATTERNS))
    elif suffix == ".sh":
        findings.extend(_scan_regex(text, _SH_PATTERNS))
    elif suffix in {".bat", ".cmd"}:
        findings.extend(_scan_regex(text, _BAT_PATTERNS))

    max_sev = _max_severity(findings) if findings else "low"
    ok = _SEVERITY_ORDER.get(max_sev, 0) < _SEVERITY_ORDER["high"]
    return CodeScanResult(
        ok=ok,
        language=language,
        path=str(resolved),
        sha256=sha256,
        scanned_chars=len(text),
        max_severity=max_sev,
        findings=findings,
    )


def _scan_regex(
    text: str,
    patterns: Sequence[Tuple[re.Pattern[str], Tuple[str, str, str]]],
) -> List[CodeScanFinding]:
    findings: List[CodeScanFinding] = []
    for rx, meta in patterns:
        match = rx.search(text or "")
        if not match:
            continue
        severity, rule, message = meta
        # Best-effort line number.
        line = None
        try:
            line = 1 + (text[: match.start()].count("\n"))
        except Exception:
            line = None
        findings.append(CodeScanFinding(severity=severity, rule=rule, message=message, line=line, col=None))
    return findings


def _scan_python(source: str) -> List[CodeScanFinding]:
    text = str(source or "")
    findings: List[CodeScanFinding] = []

    try:
        tree = ast.parse(text)
    except Exception:
        # Fallback to a shallow regex scan when parsing fails.
        rx_map: Sequence[Tuple[re.Pattern[str], Tuple[str, str, str]]] = tuple(
            (re.compile(re.escape(key)), meta) for key, meta in _PY_DANGEROUS_CALLS.items()
        )
        return _scan_regex(text, rx_map)

    aliases = _collect_import_aliases(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        callee = _python_qualname(func, aliases)
        if callee:
            meta = _PY_DANGEROUS_CALLS.get(callee)
            if meta:
                severity, rule, msg = meta
                findings.append(
                    CodeScanFinding(
                        severity=severity,
                        rule=rule,
                        message=msg,
                        line=getattr(node, "lineno", None),
                        col=getattr(node, "col_offset", None),
                    )
                )
                continue

        # Fall back to matching common destructive method names (receiver unknown).
        if isinstance(func, ast.Attribute):
            meta = _PY_DANGEROUS_METHODS.get(str(func.attr or ""))
            if meta:
                severity, rule, msg = meta
                findings.append(
                    CodeScanFinding(
                        severity=severity,
                        rule=rule,
                        message=msg,
                        line=getattr(node, "lineno", None),
                        col=getattr(node, "col_offset", None),
                    )
                )

    # De-duplicate (rule+line) to avoid noisy outputs.
    seen: set[Tuple[str, Optional[int]]] = set()
    unique: List[CodeScanFinding] = []
    for item in findings:
        key = (item.rule, item.line)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _collect_import_aliases(tree: ast.AST) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = str(alias.name or "")
                if not name:
                    continue
                key = str(alias.asname or name.split(".", 1)[0])
                aliases[key] = name
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            for alias in node.names:
                name = str(alias.name or "")
                if not name:
                    continue
                key = str(alias.asname or name)
                aliases[key] = f"{module}.{name}" if module else name
    return aliases


def _python_qualname(expr: ast.AST, aliases: Dict[str, str]) -> Optional[str]:
    if isinstance(expr, ast.Name):
        name = str(expr.id or "")
        if not name:
            return None
        return aliases.get(name, name)
    if isinstance(expr, ast.Attribute):
        base = _python_qualname(expr.value, aliases)
        attr = str(expr.attr or "")
        if not attr:
            return base
        if base:
            return f"{base}.{attr}"
        return attr
    return None


__all__ = ["CodeScanFinding", "CodeScanResult", "scan_script_file"]

