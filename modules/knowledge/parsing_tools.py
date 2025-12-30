from __future__ import annotations

"""Dependency-light parsing helpers for heterogeneous external inputs.

This module is used by tool-chain actions such as:
- ``parse_code``: AST-based parsing for Python source code.
- ``summarize_doc``: lightweight document summarisation for text/markdown/PDF.

The goal is to give the agent structured, prompt-friendly representations of
documents and code without requiring an LLM or heavyweight runtime services.
"""

import ast
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_LATEX_INLINE_RE = re.compile(r"\$(.+?)\$")
_LATEX_ENV_RE = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?)\}(.+?)\\end\{\1\}", re.DOTALL)


def _clip(text: str, *, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars]


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node).strip()
    except Exception:
        return ""


def _node_span(node: ast.AST) -> Tuple[int | None, int | None]:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if end is None:
        end = start
    return (int(start) if start is not None else None, int(end) if end is not None else None)


def _format_arguments(args: ast.arguments) -> Dict[str, Any]:
    defaults = list(args.defaults or [])
    kw_defaults = list(args.kw_defaults or [])

    def _arg_entry(arg: ast.arg, default_node: ast.AST | None = None) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"name": arg.arg}
        ann = _safe_unparse(arg.annotation)
        if ann:
            entry["annotation"] = ann
        if default_node is not None:
            default_text = _safe_unparse(default_node)
            if default_text:
                entry["default"] = default_text
        return entry

    posonly = list(args.posonlyargs or [])
    pos = list(args.args or [])

    pos_defaults: List[ast.AST | None] = [None] * max(0, len(posonly) + len(pos) - len(defaults)) + defaults
    pos_entries: List[Dict[str, Any]] = []
    combined = posonly + pos
    for arg, default_node in zip(combined, pos_defaults):
        pos_entries.append(_arg_entry(arg, default_node))

    kwonly_entries: List[Dict[str, Any]] = []
    for arg, default_node in zip(list(args.kwonlyargs or []), kw_defaults + [None] * 50):
        kwonly_entries.append(_arg_entry(arg, default_node))

    out: Dict[str, Any] = {"positional": pos_entries, "kwonly": kwonly_entries}
    if args.vararg is not None:
        out["vararg"] = _arg_entry(args.vararg)
    if args.kwarg is not None:
        out["kwarg"] = _arg_entry(args.kwarg)
    if posonly:
        out["posonly_count"] = len(posonly)
    return out


def _cyclomatic_hint(node: ast.AST) -> int:
    """Very lightweight cyclomatic-complexity hint."""

    branches = 0
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.Try,
                ast.With,
                ast.AsyncWith,
                ast.Match,
                ast.BoolOp,
                ast.ExceptHandler,
            ),
        ):
            branches += 1
    return 1 + branches


def parse_python_code(
    code: str,
    *,
    filename: str | None = None,
    include_docstrings: bool = True,
    max_items: int = 200,
) -> Dict[str, Any]:
    """Parse Python code into a structured, prompt-friendly representation."""

    text = str(code or "")
    if not text.strip():
        return {"language": "python", "error": "empty_input"}

    try:
        tree = ast.parse(text, filename=str(filename or "<code>"))
    except SyntaxError as exc:
        return {
            "language": "python",
            "error": "syntax_error",
            "message": str(exc),
            "lineno": getattr(exc, "lineno", None),
            "offset": getattr(exc, "offset", None),
        }

    module_doc = ast.get_docstring(tree) if include_docstrings else None

    imports: List[Dict[str, Any]] = []
    globals_: List[Dict[str, Any]] = []
    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []

    for node in list(getattr(tree, "body", []) or [])[: max(1, int(max_items))]:
        if isinstance(node, ast.Import):
            start, end = _node_span(node)
            imports.append(
                {
                    "type": "import",
                    "names": [alias.name for alias in node.names],
                    "asnames": [alias.asname for alias in node.names],
                    "lineno": start,
                    "end_lineno": end,
                }
            )
        elif isinstance(node, ast.ImportFrom):
            start, end = _node_span(node)
            imports.append(
                {
                    "type": "from_import",
                    "module": node.module,
                    "level": int(node.level or 0),
                    "names": [alias.name for alias in node.names],
                    "asnames": [alias.asname for alias in node.names],
                    "lineno": start,
                    "end_lineno": end,
                }
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            start, end = _node_span(node)
            targets: List[str] = []
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    name = _safe_unparse(tgt)
                    if name:
                        targets.append(name)
                value = _safe_unparse(node.value)
            else:
                name = _safe_unparse(node.target)
                if name:
                    targets.append(name)
                value = _safe_unparse(node.value)
            if targets:
                entry: Dict[str, Any] = {"targets": targets, "lineno": start, "end_lineno": end}
                if value:
                    entry["value"] = value
                globals_.append(entry)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _node_span(node)
            decorators = [_safe_unparse(d) for d in (node.decorator_list or [])]
            decorators = [d for d in decorators if d]
            entry: Dict[str, Any] = {
                "name": node.name,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "args": _format_arguments(node.args),
                "returns": _safe_unparse(node.returns) or None,
                "decorators": decorators,
                "lineno": start,
                "end_lineno": end,
                "complexity_hint": _cyclomatic_hint(node),
            }
            if include_docstrings:
                entry["docstring"] = ast.get_docstring(node) or ""
            functions.append(entry)
        elif isinstance(node, ast.ClassDef):
            start, end = _node_span(node)
            decorators = [_safe_unparse(d) for d in (node.decorator_list or [])]
            decorators = [d for d in decorators if d]
            bases = [_safe_unparse(b) for b in (node.bases or [])]
            bases = [b for b in bases if b]

            methods: List[Dict[str, Any]] = []
            class_vars: List[Dict[str, Any]] = []
            for child in list(node.body or [])[: max(1, int(max_items))]:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    c_start, c_end = _node_span(child)
                    deco = [_safe_unparse(d) for d in (child.decorator_list or [])]
                    deco = [d for d in deco if d]
                    m_entry: Dict[str, Any] = {
                        "name": child.name,
                        "is_async": isinstance(child, ast.AsyncFunctionDef),
                        "args": _format_arguments(child.args),
                        "returns": _safe_unparse(child.returns) or None,
                        "decorators": deco,
                        "lineno": c_start,
                        "end_lineno": c_end,
                        "complexity_hint": _cyclomatic_hint(child),
                    }
                    if include_docstrings:
                        m_entry["docstring"] = ast.get_docstring(child) or ""
                    methods.append(m_entry)
                elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                    c_start, c_end = _node_span(child)
                    targets: List[str] = []
                    if isinstance(child, ast.Assign):
                        for tgt in child.targets:
                            name = _safe_unparse(tgt)
                            if name:
                                targets.append(name)
                        value = _safe_unparse(child.value)
                    else:
                        name = _safe_unparse(child.target)
                        if name:
                            targets.append(name)
                        value = _safe_unparse(child.value)
                    if targets:
                        cv: Dict[str, Any] = {"targets": targets, "lineno": c_start, "end_lineno": c_end}
                        if value:
                            cv["value"] = value
                        class_vars.append(cv)

            c_entry: Dict[str, Any] = {
                "name": node.name,
                "bases": bases,
                "decorators": decorators,
                "lineno": start,
                "end_lineno": end,
                "methods": methods,
                "class_vars": class_vars,
            }
            if include_docstrings:
                c_entry["docstring"] = ast.get_docstring(node) or ""
            classes.append(c_entry)

    return {
        "language": "python",
        "module": {
            "docstring": module_doc or "",
            "imports": imports,
            "globals": globals_,
            "functions": functions,
            "classes": classes,
        },
    }


def extract_markdown_outline(text: str, *, max_headings: int = 12) -> List[str]:
    headings: List[str] = []
    for line in str(text or "").splitlines():
        match = _MD_HEADING_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        if not title:
            continue
        headings.append(f"{level}:{title}")
        if len(headings) >= max(1, int(max_headings)):
            break
    return headings


def extract_formulas(text: str, *, max_formulas: int = 8) -> List[str]:
    raw = str(text or "")
    formulas: List[str] = []
    formulas.extend(_LATEX_INLINE_RE.findall(raw))

    for _env, body in _LATEX_ENV_RE.findall(raw):
        body_clean = " ".join(body.split())
        if body_clean:
            formulas.append(body_clean)

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(sym in stripped for sym in ("=", "≈", "≤", "≥", "→", "∈", "∑", "∫")) and len(stripped) <= 160:
            formulas.append(stripped)

    return _dedupe_preserve_order(formulas)[: max(1, int(max_formulas))]


def extract_keywords(text: str, *, max_keywords: int = 12) -> List[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(str(text or ""))]
    if not tokens:
        return []
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "are",
        "was",
        "were",
        "into",
        "then",
        "than",
        "else",
        "import",
        "return",
        "class",
        "def",
    }
    freq: Dict[str, int] = {}
    for tok in tokens:
        if tok in stop:
            continue
        if len(tok) <= 2:
            continue
        freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return [k for k, _ in ranked[: max(1, int(max_keywords))]]


def summarize_text(
    text: str,
    *,
    max_summary_chars: int = 800,
    max_headings: int = 12,
    max_keywords: int = 12,
    max_formulas: int = 8,
) -> Dict[str, Any]:
    raw = str(text or "")
    headings = extract_markdown_outline(raw, max_headings=max_headings)
    formulas = extract_formulas(raw, max_formulas=max_formulas)
    keywords = extract_keywords(raw, max_keywords=max_keywords)

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw.replace("\r\n", "\n")) if p.strip()]
    summary = paragraphs[0] if paragraphs else raw.strip()
    summary = " ".join(summary.split())
    summary = _clip(summary, max_chars=max_summary_chars)

    return {
        "summary": summary,
        "headings": headings,
        "keywords": keywords,
        "formulas": formulas,
    }


def _pdf_to_text(pdf_bytes: bytes, *, max_pages: int = 6) -> Tuple[str, Dict[str, Any]]:
    """Best-effort PDF->text extraction with optional dependencies."""

    used = None
    text_parts: List[str] = []

    # Try pypdf / PyPDF2 first.
    reader_cls = None
    for mod_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(mod_name, fromlist=["PdfReader"])
            reader_cls = getattr(module, "PdfReader", None)
        except Exception:
            reader_cls = None
        if reader_cls is not None:
            used = mod_name
            break

    if reader_cls is not None:
        try:
            reader = reader_cls(io.BytesIO(pdf_bytes))
            pages = getattr(reader, "pages", []) or []
            for page in list(pages)[: max(1, int(max_pages))]:
                try:
                    extracted = page.extract_text() or ""
                except Exception:
                    extracted = ""
                if extracted:
                    text_parts.append(extracted)
            return "\n".join(text_parts).strip(), {"parser": used, "pages": min(len(pages), int(max_pages))}
        except Exception:
            pass

    # Fallback to pdfminer.six.
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
    except Exception as exc:
        return "", {"error": "pdf_parser_unavailable", "exception": repr(exc)}

    try:
        text = pdfminer_extract_text(io.BytesIO(pdf_bytes), maxpages=max(1, int(max_pages))) or ""
    except Exception as exc:
        return "", {"error": "pdf_extract_failed", "exception": repr(exc)}
    return str(text).strip(), {"parser": "pdfminer.six", "pages": int(max_pages)}


def summarize_document(
    *,
    text: str | None = None,
    path: str | Path | None = None,
    pdf_bytes: bytes | None = None,
    max_chars: int = 120_000,
    max_summary_chars: int = 900,
    max_headings: int = 12,
    max_keywords: int = 12,
    max_formulas: int = 8,
) -> Dict[str, Any]:
    """Summarize a document by text or by path.

    For PDFs, pass ``pdf_bytes`` (preferred) or rely on callers to extract bytes.
    """

    suffix = ""
    if path is not None:
        suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        if pdf_bytes is None:
            return {"error": "missing_pdf_bytes"}
        extracted, meta = _pdf_to_text(pdf_bytes, max_pages=6)
        if not extracted:
            return {"error": meta.get("error") or "pdf_extract_failed", "meta": meta}
        doc = summarize_text(
            _clip(extracted, max_chars=max_chars),
            max_summary_chars=max_summary_chars,
            max_headings=max_headings,
            max_keywords=max_keywords,
            max_formulas=max_formulas,
        )
        doc["meta"] = meta
        return doc

    if text is None:
        return {"error": "missing_text"}

    raw = _clip(str(text), max_chars=max_chars)
    if suffix == ".py":
        parsed = parse_python_code(raw, filename=str(path) if path is not None else None, include_docstrings=True)
        if parsed.get("error"):
            return {"error": parsed["error"], "details": parsed}
        module = parsed.get("module") if isinstance(parsed.get("module"), Mapping) else {}
        func_names = [f.get("name") for f in (module.get("functions") or []) if isinstance(f, Mapping)]
        class_names = [c.get("name") for c in (module.get("classes") or []) if isinstance(c, Mapping)]
        headline = []
        if class_names:
            headline.append("classes: " + ", ".join(str(n) for n in class_names[:8]))
        if func_names:
            headline.append("functions: " + ", ".join(str(n) for n in func_names[:10]))
        summary = "; ".join(headline) or "Python module"
        return {
            "summary": _clip(summary, max_chars=max_summary_chars),
            "headings": [],
            "keywords": extract_keywords(raw, max_keywords=max_keywords),
            "formulas": extract_formulas(raw, max_formulas=max_formulas),
            "parsed_code": parsed,
        }

    return summarize_text(
        raw,
        max_summary_chars=max_summary_chars,
        max_headings=max_headings,
        max_keywords=max_keywords,
        max_formulas=max_formulas,
    )


__all__ = [
    "parse_python_code",
    "summarize_document",
    "summarize_text",
    "extract_markdown_outline",
    "extract_formulas",
    "extract_keywords",
]

