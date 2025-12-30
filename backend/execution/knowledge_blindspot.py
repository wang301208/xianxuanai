from __future__ import annotations

"""Knowledge blindspot detection utilities.

This module provides a lightweight "knowledge coverage" probe used to decide
whether the agent should:

- proactively retrieve more information (docs / memory / web search), and/or
- transparently declare low knowledge ("I don't know enough about this topic").

The detector is intentionally defensive and dependency-light. It can work with:
- a MemoryRouter-like object exposing `query(text, top_k=...) -> list[dict]`
  that may include `similarity` scores; and
- optional local docs search via `modules.knowledge.research_tool.ResearchTool`.
"""

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if number != number or number in (float("inf"), float("-inf")):
        return float(default)
    return float(number)


def _unique_tokens(tokens: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for token in tokens:
        key = str(token or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _extract_keywords(text: str) -> List[str]:
    tokens = re.split(r"[,;\\s]+", str(text or ""))
    cleaned: List[str] = []
    for token in tokens:
        token = token.strip().lower()
        if len(token) < 3:
            continue
        cleaned.append(token)
    return _unique_tokens(cleaned)


@dataclass(frozen=True)
class KnowledgeBlindspotConfig:
    """Tunable knobs for blindspot detection."""

    enabled: bool = False
    # Similarity threshold when counting memory hits.
    min_similarity: float = 0.25
    # `min_support` counts "evidence units" across memory hits + doc hits.
    # Values <= 1 detect only empty coverage; values >= 2 treat "few" hits as blindspots.
    min_support: int = 2
    # MemoryRouter query parameters.
    top_k: int = 5
    # Local docs probing.
    docs_enabled: bool = True
    docs_max_keywords: int = 4
    docs_max_results_per_keyword: int = 1

    # Declaration copy (can be surfaced to users / upstream agents).
    declaration_zh: str = "我目前缺乏这方面知识"
    declaration_en: str = "I currently lack sufficient knowledge about this topic."

    @classmethod
    def from_env(cls) -> "KnowledgeBlindspotConfig":
        return cls(
            enabled=_env_bool("KNOWLEDGE_BLINDSPOT_ENABLED", False),
            min_similarity=_env_float("KNOWLEDGE_BLINDSPOT_MIN_SIMILARITY", 0.25),
            min_support=_env_int("KNOWLEDGE_BLINDSPOT_MIN_SUPPORT", 2),
            top_k=_env_int("KNOWLEDGE_BLINDSPOT_TOP_K", 5),
            docs_enabled=_env_bool("KNOWLEDGE_BLINDSPOT_DOCS_ENABLED", True),
            docs_max_keywords=_env_int("KNOWLEDGE_BLINDSPOT_DOCS_MAX_KEYWORDS", 4),
            docs_max_results_per_keyword=_env_int(
                "KNOWLEDGE_BLINDSPOT_DOCS_MAX_RESULTS_PER_KEYWORD", 1
            ),
            declaration_zh=str(os.getenv("KNOWLEDGE_BLINDSPOT_DECLARATION_ZH", cls.declaration_zh)).strip()
            or cls.declaration_zh,
            declaration_en=str(os.getenv("KNOWLEDGE_BLINDSPOT_DECLARATION_EN", cls.declaration_en)).strip()
            or cls.declaration_en,
        )


@dataclass(frozen=True)
class KnowledgeBlindspotAssessment:
    """Structured result from a blindspot probe."""

    enabled: bool
    blindspot: bool
    level: str  # ok|sparse|empty|disabled
    support: int
    memory_hits: int
    doc_hits: int
    keywords: List[str] = field(default_factory=list)
    query_preview: str = ""
    declaration_zh: str = ""
    declaration_en: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KnowledgeBlindspotDetector:
    """Probe knowledge coverage using memory retrieval + optional local docs search."""

    def __init__(
        self,
        *,
        memory_router: Any | None = None,
        config: KnowledgeBlindspotConfig | None = None,
        workspace_root: str | None = None,
    ) -> None:
        self._router = memory_router
        self.config = config or KnowledgeBlindspotConfig.from_env()
        self._workspace_root = workspace_root or os.getenv("WORKSPACE_ROOT") or os.getcwd()

    def assess(
        self,
        *,
        query_text: str,
        keywords: Sequence[str] | None = None,
    ) -> KnowledgeBlindspotAssessment:
        cfg = self.config
        if not cfg.enabled:
            return KnowledgeBlindspotAssessment(
                enabled=False,
                blindspot=False,
                level="disabled",
                support=0,
                memory_hits=0,
                doc_hits=0,
                keywords=list(keywords or []),
                query_preview=str(query_text or "")[:200],
            )

        query = str(query_text or "").strip()
        candidate_keywords = list(keywords or [])
        if not candidate_keywords:
            candidate_keywords = _extract_keywords(query)

        memory_hits = self._memory_support(query)
        doc_hits = self._doc_support(candidate_keywords)
        support = int(memory_hits + doc_hits)
        min_support = max(0, int(cfg.min_support))
        blindspot = support < min_support if min_support > 0 else False
        level = "ok"
        if blindspot:
            level = "empty" if support <= 0 else "sparse"

        return KnowledgeBlindspotAssessment(
            enabled=True,
            blindspot=bool(blindspot),
            level=level,
            support=support,
            memory_hits=int(memory_hits),
            doc_hits=int(doc_hits),
            keywords=list(candidate_keywords),
            query_preview=query[:200],
            declaration_zh=cfg.declaration_zh,
            declaration_en=cfg.declaration_en,
        )

    # ------------------------------------------------------------------
    def _memory_support(self, query: str) -> int:
        cfg = self.config
        router = self._router
        if router is None or not hasattr(router, "query"):
            return 0
        try:
            results = router.query(query, top_k=max(1, int(cfg.top_k)))  # type: ignore[misc]
        except Exception:
            return 0
        hits = 0
        for item in results or []:
            if not isinstance(item, Mapping):
                continue
            similarity = _safe_float(item.get("similarity"), 0.0)
            if similarity >= float(cfg.min_similarity):
                hits += 1
        return int(hits)

    def _doc_support(self, keywords: Sequence[str]) -> int:
        cfg = self.config
        if not cfg.docs_enabled:
            return 0
        try:
            from modules.knowledge.research_tool import ResearchTool  # type: ignore
        except Exception:
            return 0

        max_keywords = max(0, int(cfg.docs_max_keywords))
        max_results = max(1, int(cfg.docs_max_results_per_keyword))
        chosen = [kw for kw in _unique_tokens(keywords) if kw][:max_keywords]
        if not chosen:
            return 0

        tool = ResearchTool(workspace_root=self._workspace_root)
        hits = 0
        for kw in chosen:
            try:
                found = tool.query_docs(kw, max_results=max_results)
            except Exception:
                found = []
            hits += len(found)
        return int(hits)


__all__ = [
    "KnowledgeBlindspotConfig",
    "KnowledgeBlindspotAssessment",
    "KnowledgeBlindspotDetector",
]

