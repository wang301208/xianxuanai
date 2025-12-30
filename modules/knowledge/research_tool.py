from __future__ import annotations

"""Lightweight research helpers (web search + local docs search).

This module is designed to be used by higher-level planners such as
`modules.knowledge.problem_analyzer.ProblemAnalyzer`.

All features are best-effort and intentionally conservative:
- Web search uses `duckduckgo_search.DDGS` when available (the repo includes a
  deterministic stub for tests).
- Local doc search limits traversal and read sizes to avoid scanning the entire
  workspace by accident.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSearchHit:
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo"

    def to_dict(self) -> Dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "source": self.source}


@dataclass(frozen=True)
class DocHit:
    path: str
    snippet: str

    def to_dict(self) -> Dict[str, str]:
        return {"path": self.path, "snippet": self.snippet}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class ResearchTool:
    """Simple research toolbelt with minimal dependencies."""

    def __init__(
        self,
        *,
        workspace_root: str | os.PathLike[str] | None = None,
        docs_roots: Sequence[str | os.PathLike[str]] | None = None,
        include_suffixes: Sequence[str] = (".md", ".rst", ".txt"),
        exclude_dirs: Sequence[str] = (".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"),
        max_scan_files: int = 300,
        max_file_chars: int = 50_000,
    ) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.docs_roots = tuple(Path(p).resolve() for p in (docs_roots or (self.workspace_root,)))
        self.include_suffixes = tuple(str(s).lower() for s in include_suffixes)
        self.exclude_dirs = set(str(d) for d in exclude_dirs)
        self.max_scan_files = max(1, int(max_scan_files))
        self.max_file_chars = max(1, int(max_file_chars))

    # ------------------------------------------------------------------ web search
    def search_web(self, query: str, *, max_results: int = 5) -> List[WebSearchHit]:
        query = _safe_text(query)
        if not query:
            return []

        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            logger.debug("duckduckgo_search not available; web search disabled.")
            return []

        try:
            raw = DDGS().text(query, max_results=max(1, int(max_results)))
        except Exception:  # pragma: no cover - network/client failures should not crash
            logger.debug("web search failed for query=%s", query, exc_info=True)
            return []

        hits: List[WebSearchHit] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            title = _safe_text(item.get("title"))
            url = _safe_text(item.get("href") or item.get("url"))
            snippet = _safe_text(item.get("body") or item.get("snippet"))
            if not (title or url or snippet):
                continue
            hits.append(WebSearchHit(title=title or url or query, url=url, snippet=snippet))
        return hits[: max(1, int(max_results))]

    def render_web_hits(self, hits: Iterable[WebSearchHit]) -> str:
        lines: List[str] = []
        for idx, hit in enumerate(hits, start=1):
            title = _compact_whitespace(hit.title)
            snippet = _compact_whitespace(hit.snippet)
            url = _compact_whitespace(hit.url)
            if url:
                lines.append(f"{idx}. {title} ({url})")
            else:
                lines.append(f"{idx}. {title}")
            if snippet:
                lines.append(f"   - {snippet}")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------ docs search
    def query_docs(
        self,
        keyword: str,
        *,
        roots: Sequence[str | os.PathLike[str]] | None = None,
        max_results: int = 5,
        case_sensitive: bool = False,
    ) -> List[DocHit]:
        keyword = _safe_text(keyword)
        if not keyword:
            return []

        roots_resolved = tuple(Path(p).resolve() for p in (roots or self.docs_roots))
        max_results = max(1, int(max_results))

        needle = keyword if case_sensitive else keyword.lower()
        hits: List[DocHit] = []
        scanned = 0

        def _search_file(path: Path) -> None:
            nonlocal hits, scanned
            if len(hits) >= max_results or scanned >= self.max_scan_files:
                return
            scanned += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return
            if self.max_file_chars and len(text) > self.max_file_chars:
                text = text[: self.max_file_chars]
            haystack = text if case_sensitive else text.lower()
            idx = haystack.find(needle)
            if idx == -1:
                return
            start = max(0, idx - 120)
            end = min(len(text), idx + len(keyword) + 240)
            snippet = _compact_whitespace(text[start:end])
            hits.append(DocHit(path=str(path), snippet=snippet))

        for root in roots_resolved:
            if len(hits) >= max_results or scanned >= self.max_scan_files:
                break
            if root.is_file():
                if root.suffix.lower() in self.include_suffixes:
                    _search_file(root)
                continue
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                # prune excluded dirs
                dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs]
                if len(hits) >= max_results or scanned >= self.max_scan_files:
                    break
                base = Path(dirpath)
                for fname in filenames:
                    if len(hits) >= max_results or scanned >= self.max_scan_files:
                        break
                    candidate = base / fname
                    if candidate.suffix.lower() not in self.include_suffixes:
                        continue
                    _search_file(candidate)

        return hits

    def render_doc_hits(self, hits: Iterable[DocHit]) -> str:
        lines: List[str] = []
        for idx, hit in enumerate(hits, start=1):
            lines.append(f"{idx}. {hit.path}")
            if hit.snippet:
                lines.append(f"   - {hit.snippet}")
        return "\n".join(lines).strip()


__all__ = ["ResearchTool", "WebSearchHit", "DocHit"]

