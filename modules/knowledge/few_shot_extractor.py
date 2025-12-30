from __future__ import annotations

"""Extract prompt-friendly few-shot material from documentation sources."""

import re
from typing import Any, Dict, Iterable, Mapping, Sequence


_DEF_RE = re.compile(
    r"^(?:async\s+def|def)\s+[A-Za-z_][A-Za-z0-9_]*\s*\(.*\)\s*(?:->\s*[^:]+)?\s*:",
    re.IGNORECASE,
)
_CLASS_RE = re.compile(r"^class\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:\(|:)", re.IGNORECASE)


def _clip(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 1] + "…"


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _first_nonempty_line(block: str) -> str:
    for line in str(block or "").splitlines():
        value = line.strip()
        if value:
            return value
    return ""


def extract_few_shot_material(
    sources_or_observation: Any,
    *,
    max_signatures: int = 8,
    max_snippets: int = 6,
    max_chars_per_snippet: int = 900,
    max_total_chars: int = 3500,
) -> Dict[str, Any]:
    """Extract signatures + pseudocode/code snippets from documentation sources.

    Parameters accept either:
    - documentation_tool observation dict (expects key ``sources``)
    - raw ``sources`` list
    """

    sources = None
    if isinstance(sources_or_observation, Mapping):
        sources = sources_or_observation.get("sources")
    else:
        sources = sources_or_observation

    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes, bytearray)):
        return {}

    signatures: list[str] = []
    snippets: list[str] = []
    source_refs: list[Dict[str, str]] = []
    source_seen: set[tuple[str, str]] = set()

    for entry in sources:
        if not isinstance(entry, Mapping):
            continue
        search = entry.get("search") if isinstance(entry.get("search"), Mapping) else {}
        page = entry.get("page") if isinstance(entry.get("page"), Mapping) else {}
        title = str(page.get("title") or search.get("title") or "").strip()
        url = str(page.get("final_url") or page.get("url") or search.get("url") or "").strip()
        if url:
            key = (url, title)
            if key not in source_seen:
                source_seen.add(key)
                ref = {"url": url}
                if title:
                    ref["title"] = title
                source_refs.append(ref)

        blocks = page.get("code_blocks")
        if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes, bytearray)):
            continue

        for block in blocks:
            text = str(block or "").strip()
            if not text:
                continue
            first = _first_nonempty_line(text)
            if first and (_DEF_RE.match(first) or _CLASS_RE.match(first)):
                signatures.append(first)
                continue
            if len(snippets) < max(0, int(max_snippets)):
                snippets.append(_clip(text, int(max_chars_per_snippet)))

    signatures = _dedupe(signatures)[: max(0, int(max_signatures))]
    snippets = _dedupe(snippets)[: max(0, int(max_snippets))]
    material: Dict[str, Any] = {}
    if signatures:
        material["signatures"] = signatures
    if snippets:
        material["snippets"] = snippets
    if source_refs:
        material["sources"] = source_refs

    if max_total_chars > 0:
        # Ensure payload stays prompt-friendly.
        total = 0
        trimmed: Dict[str, Any] = {}
        for key in ("signatures", "snippets"):
            values = material.get(key)
            if not isinstance(values, list):
                continue
            kept: list[str] = []
            for item in values:
                part = str(item)
                if total + len(part) > max_total_chars:
                    break
                kept.append(part)
                total += len(part)
            if kept:
                trimmed[key] = kept
        if "sources" in material:
            trimmed["sources"] = material["sources"]
        material = trimmed

    return material


__all__ = ["extract_few_shot_material"]
