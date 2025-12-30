"""
Heuristic advisor for automatic memory retrieval queries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _normalize_sequence(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence):
        result: List[str] = []
        for item in value:
            if isinstance(item, str):
                token = item.strip()
            elif isinstance(item, dict):
                token = (
                    item.get("goal")
                    or item.get("action")
                    or item.get("description")
                    or item.get("summary")
                    or ""
                )
                token = str(token).strip()
            else:
                token = str(item).strip()
            if token:
                result.append(token)
        return result
    return [str(value).strip()]


class MemoryRetrievalAdvisor:
    """Construct memory retrieval queries from high-level task context."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.max_keywords = int(cfg.get("max_keywords", 6))
        self.min_query_length = int(cfg.get("min_query_length", 3))
        self.priority_tags = tuple(cfg.get("priority_tags", ["goal", "action"]))

    def build_query(
        self,
        *,
        goals: Any = None,
        planner: Optional[Dict[str, Any]] = None,
        dialogue_state: Optional[Dict[str, Any]] = None,
        summary: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        keywords = self._collect_keywords(goals, planner, dialogue_state)
        if summary:
            keywords.append(summary.strip())
        keywords = [token for token in keywords if token]
        if not keywords:
            return None

        query_text = "; ".join(keywords[: self.max_keywords])
        if len(query_text) < self.min_query_length:
            return None

        return {
            "memory_type": "SEMANTIC",
            "query": {
                "query": query_text,
                "keywords": keywords[: self.max_keywords],
            },
            "context": {
                "tags": list(self.priority_tags),
                "source": "auto_goal_retrieval",
            },
        }

    def _collect_keywords(
        self,
        goals: Any,
        planner: Optional[Dict[str, Any]],
        dialogue_state: Optional[Dict[str, Any]],
    ) -> List[str]:
        keywords: List[str] = []
        keywords.extend(_normalize_sequence(goals))

        if planner:
            for candidate in planner.get("candidates") or []:
                action = candidate.get("action") if isinstance(candidate, dict) else candidate
                if action:
                    keywords.append(str(action))
                if isinstance(candidate, dict):
                    keywords.extend(candidate.get("justification", []))

        if dialogue_state:
            keywords.extend(dialogue_state.get("topics", []) or [])
            keywords.extend(dialogue_state.get("entities", []) or [])

        return keywords[: self.max_keywords * 2]


__all__ = ["MemoryRetrievalAdvisor"]

