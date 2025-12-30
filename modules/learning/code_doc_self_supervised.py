from __future__ import annotations

"""Self-supervised dataset builder for code <-> docs <-> usage alignment.

This module extracts training examples from local corpora (e.g. repositories
ingested via ``github_repo_ingest``) to support offline self-supervised learning.

Design goals:
- Dependency-light (stdlib only).
- Safe by default: limits files/examples/bytes and clips prompts.
- Produces JSONL-friendly records that can be used to train:
  - Small helper LMs (prompt -> completion)
  - Dual-encoders / retrievers (code/doc pairing)
"""

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


_DEFAULT_EXCLUDE_DIRS: Tuple[str, ...] = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "node_modules",
    ".bss_sandbox",
)

_DEFAULT_SUFFIXES: Tuple[str, ...] = (".py", ".md", ".txt", ".rst")

_PY_DEF_RE = re.compile(r"^(?P<prefix>\s*(?:async\s+def|def)\s+)(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_MD_FENCE_RE = re.compile(r"^\s{0,3}```(?P<lang>[A-Za-z0-9_+-]*)\s*$")


def _clip(text: str, *, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars]


def _safe_read_text(path: Path, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    return _clip(text, max_chars=max_chars)


def _node_span(node: ast.AST) -> Tuple[int | None, int | None]:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if end is None:
        end = start
    return (int(start) if start is not None else None, int(end) if end is not None else None)


def _extract_source_segment(lines: Sequence[str], start_line: int | None, end_line: int | None) -> str:
    if start_line is None:
        return ""
    start = max(0, int(start_line) - 1)
    end = int(end_line or start_line)
    end = max(start + 1, end)
    return "\n".join(lines[start:end])


def _mask_python_def_name(signature_line: str) -> Tuple[str, str]:
    """Return (masked_signature, original_name) if the line is a def."""

    line = str(signature_line or "")
    match = _PY_DEF_RE.match(line)
    if not match:
        return line, ""
    name = match.group("name")
    masked = match.group("prefix") + "<FUNC_NAME>" + line[match.end("name") :]
    return masked, name


def _dedupe_preserve_order(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        try:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


@dataclass(frozen=True)
class CodeDocDatasetConfig:
    include_suffixes: Sequence[str] = _DEFAULT_SUFFIXES
    exclude_dirs: Sequence[str] = _DEFAULT_EXCLUDE_DIRS
    max_files: int = 1200
    max_chars_per_file: int = 180_000
    max_examples: int = 10_000
    max_input_chars: int = 2800
    max_output_chars: int = 1400
    tasks: Sequence[str] = ("python_function_name", "python_docstring", "markdown_code_to_heading", "code_doc_pair")


def iter_corpus_files(
    roots: Sequence[str | os.PathLike[str]],
    *,
    include_suffixes: Sequence[str],
    exclude_dirs: Sequence[str],
    max_files: int,
) -> Iterator[Path]:
    suffix_set = {str(s).lower() for s in include_suffixes if str(s).strip()}
    excluded = {str(d) for d in exclude_dirs if str(d).strip()}
    emitted = 0
    for root in roots:
        base = Path(root).resolve()
        if not base.exists():
            continue
        if base.is_file():
            if base.suffix.lower() in suffix_set:
                yield base
                emitted += 1
            if emitted >= max_files:
                return
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in excluded]
            for fname in filenames:
                candidate = Path(dirpath) / fname
                if candidate.suffix.lower() not in suffix_set:
                    continue
                yield candidate
                emitted += 1
                if emitted >= max_files:
                    return


def _python_examples_from_tree(
    *,
    text: str,
    path: Path,
    task_set: set[str],
    max_input_chars: int,
    max_output_chars: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    if not text.strip():
        return examples

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return examples

    lines = text.splitlines()

    def _emit(task: str, input_text: str, output_text: str, meta: Dict[str, Any]) -> None:
        examples.append(
            {
                "task": task,
                "input": _clip(input_text, max_chars=max_input_chars).strip(),
                "output": _clip(output_text, max_chars=max_output_chars).strip(),
                "meta": dict(meta),
            }
        )

    def _handle_function(node: ast.AST, *, owner: str | None = None) -> None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        start, end = _node_span(node)
        segment = _extract_source_segment(lines, start, end)
        if not segment:
            return
        first_line = segment.splitlines()[0] if segment else ""
        masked, original = _mask_python_def_name(first_line)
        if not original:
            return
        docstring = ast.get_docstring(node) or ""
        meta = {
            "path": str(path),
            "relative_path": path.as_posix(),
            "symbol": f"{owner}.{original}" if owner else original,
            "kind": "function",
            "lineno": start,
            "end_lineno": end,
        }

        if "python_function_name" in task_set:
            prompt_parts = [
                "Task: recover the function name.",
                f"Signature: {masked}",
            ]
            if docstring:
                prompt_parts.extend(["Docstring:", docstring])
            else:
                # Provide a short body preview when no docstring exists.
                body_preview = "\n".join(segment.splitlines()[1:10]).strip()
                if body_preview:
                    prompt_parts.extend(["Body:", body_preview])
            _emit("python_function_name", "\n".join(prompt_parts), original, meta)

        if docstring and "python_docstring" in task_set:
            # Build a code context without the docstring block when possible.
            body_lines = segment.splitlines()
            doc_node = None
            if getattr(node, "body", None):
                first_stmt = node.body[0]
                if isinstance(first_stmt, ast.Expr) and isinstance(getattr(first_stmt, "value", None), ast.Constant):
                    if isinstance(first_stmt.value.value, str):
                        doc_node = first_stmt
            if doc_node is not None:
                d_start, d_end = _node_span(doc_node)
                if d_start is not None:
                    seg_lines = segment.splitlines()
                    local_start = 0
                    local_doc_start = max(0, int(d_start) - int(start or d_start) )
                    local_doc_end = max(local_doc_start, int(d_end or d_start) - int(start or d_start))
                    filtered = seg_lines[: max(1, local_doc_start)] + seg_lines[local_doc_end + 1 :]
                    body_lines = filtered

            context = "\n".join(body_lines[:40]).strip()
            prompt = "\n".join(
                [
                    "Task: write a docstring for the following Python code.",
                    "Code:",
                    context,
                ]
            )
            _emit("python_docstring", prompt, docstring, meta)

        if docstring and "code_doc_pair" in task_set:
            context = "\n".join(segment.splitlines()[:50]).strip()
            _emit("code_doc_pair", context, docstring, meta)

    for node in list(getattr(tree, "body", []) or []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _handle_function(node)
        elif isinstance(node, ast.ClassDef):
            owner = node.name
            for child in list(getattr(node, "body", []) or []):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _handle_function(child, owner=owner)

    return examples


def _markdown_examples(
    *,
    text: str,
    path: Path,
    task_set: set[str],
    max_input_chars: int,
    max_output_chars: int,
) -> List[Dict[str, Any]]:
    if "markdown_code_to_heading" not in task_set:
        return []
    raw = str(text or "")
    if not raw.strip():
        return []

    lines = raw.replace("\r\n", "\n").splitlines()
    headings: List[Tuple[int, str, int]] = []  # (level, title, line_index)
    for idx, line in enumerate(lines):
        match = _MD_HEADING_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        if title:
            headings.append((level, title, idx))

    def _nearest_heading(line_idx: int) -> str:
        candidates = [h for h in headings if h[2] < line_idx]
        if not candidates:
            return ""
        return candidates[-1][1]

    examples: List[Dict[str, Any]] = []
    in_fence = False
    fence_lang = ""
    fence_start = 0
    fence_lines: List[str] = []

    for idx, line in enumerate(lines):
        fence = _MD_FENCE_RE.match(line)
        if fence:
            if not in_fence:
                in_fence = True
                fence_lang = str(fence.group("lang") or "").strip().lower()
                fence_start = idx
                fence_lines = []
            else:
                # closing
                in_fence = False
                code = "\n".join(fence_lines).strip()
                heading = _nearest_heading(fence_start)
                if code and heading:
                    meta = {
                        "path": str(path),
                        "relative_path": path.as_posix(),
                        "kind": "markdown_code_block",
                        "lineno": fence_start + 1,
                        "end_lineno": idx + 1,
                        "language": fence_lang or None,
                    }
                    prompt = "\n".join(
                        [
                            "Task: identify the documentation section title for this example code.",
                            f"Code ({fence_lang or 'text'}):",
                            _clip(code, max_chars=max_input_chars),
                        ]
                    )
                    examples.append(
                        {
                            "task": "markdown_code_to_heading",
                            "input": prompt.strip(),
                            "output": _clip(heading, max_chars=max_output_chars).strip(),
                            "meta": meta,
                        }
                    )
                fence_lang = ""
                fence_lines = []
            continue

        if in_fence:
            fence_lines.append(line)

    return examples


def build_self_supervised_examples(
    roots: Sequence[str | os.PathLike[str]],
    *,
    config: CodeDocDatasetConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or CodeDocDatasetConfig()
    task_set = {str(t) for t in (cfg.tasks or []) if str(t).strip()}

    files_scanned = 0
    examples_built: List[Dict[str, Any]] = []
    skipped_files = 0

    for path in iter_corpus_files(
        roots,
        include_suffixes=cfg.include_suffixes,
        exclude_dirs=cfg.exclude_dirs,
        max_files=max(1, int(cfg.max_files)),
    ):
        files_scanned += 1
        text = _safe_read_text(path, max_chars=max(1, int(cfg.max_chars_per_file)))
        if not text.strip():
            skipped_files += 1
            continue

        suffix = path.suffix.lower()
        if suffix == ".py":
            examples_built.extend(
                _python_examples_from_tree(
                    text=text,
                    path=path,
                    task_set=task_set,
                    max_input_chars=int(cfg.max_input_chars),
                    max_output_chars=int(cfg.max_output_chars),
                )
            )
        elif suffix in {".md", ".rst", ".txt"}:
            # Only markdown supports fenced code blocks reliably; .rst/.txt still yield headings-less tasks (none).
            if suffix == ".md":
                examples_built.extend(
                    _markdown_examples(
                        text=text,
                        path=path,
                        task_set=task_set,
                        max_input_chars=int(cfg.max_input_chars),
                        max_output_chars=int(cfg.max_output_chars),
                    )
                )

        if len(examples_built) >= int(cfg.max_examples):
            examples_built = examples_built[: int(cfg.max_examples)]
            break

    examples_built = _dedupe_preserve_order(examples_built)
    if len(examples_built) > int(cfg.max_examples):
        examples_built = examples_built[: int(cfg.max_examples)]

    return {
        "stats": {
            "roots": [str(Path(r).resolve()) for r in roots],
            "files_scanned": int(files_scanned),
            "skipped_files": int(skipped_files),
            "examples": int(len(examples_built)),
            "tasks": sorted(task_set),
        },
        "examples": examples_built,
    }


def write_jsonl(examples: Sequence[Dict[str, Any]], output_path: str | Path) -> Dict[str, Any]:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8", errors="replace", newline="\n") as handle:
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            handle.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1
    return {"output_path": str(target), "written": int(written)}


__all__ = ["CodeDocDatasetConfig", "build_self_supervised_examples", "iter_corpus_files", "write_jsonl"]

