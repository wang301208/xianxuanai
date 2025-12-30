"""Utilities for fusing knowledge graph context into LLM prompts."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
from concurrent.futures import Executor
from typing import Any, Dict, Iterable, List, Optional

from backend.knowledge.registry import get_graph_store_instance
from backend.monitoring.global_workspace import WorkspaceMessage, global_workspace

LOGGER = logging.getLogger(__name__)


def collect_knowledge_context(
    query: str,
    *,
    knowledge_base: Any | None = None,
    top_k: int = 5,
    relation_limit: int = 3,
    max_line_length: int = 220,
) -> List[str]:
    """Return formatted knowledge snippets relevant to ``query``."""

    query = (query or "").strip()
    if not query:
        return []

    snippets: List[str] = []

    if knowledge_base is not None:
        try:
            kb_result = knowledge_base.query(query, semantic=True, top_k=top_k)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Knowledge base query failed", exc_info=True)
        else:
            snippets = _format_kb_result(
                kb_result,
                top_k=top_k,
                relation_limit=relation_limit,
                max_line=max_line_length,
            )

    if not snippets:
        try:
            snippets = _lookup_graph(
                query,
                top_k=top_k,
                relation_limit=relation_limit,
                max_line=max_line_length,
            )
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Knowledge graph lookup failed", exc_info=True)
            snippets = []

    if top_k > 0:
        limit = top_k * max(1, relation_limit + 1)
        if len(snippets) > limit:
            snippets = snippets[:limit]
    if snippets:
        _broadcast_workspace_knowledge(query, snippets)
    return snippets


async def collect_knowledge_context_async(
    query: str,
    *,
    knowledge_base: Any | None = None,
    top_k: int = 5,
    relation_limit: int = 3,
    max_line_length: int = 220,
    timeout: float | None = 6.0,
    executor: Executor | None = None,
) -> List[str]:
    """Asynchronous wrapper around :func:`collect_knowledge_context`.

    This helper executes the synchronous collection routine in a background
    executor, enforcing the provided timeout to ensure the calling coroutine is
    not blocked indefinitely when external stores are slow to respond.
    """

    loop = asyncio.get_running_loop()
    bound = functools.partial(
        collect_knowledge_context,
        query,
        knowledge_base=knowledge_base,
        top_k=top_k,
        relation_limit=relation_limit,
        max_line_length=max_line_length,
    )

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, bound),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        LOGGER.warning(
            "Knowledge context lookup timed out after %.1fs for query '%s'.",
            timeout or 0.0,
            query,
        )
        return []
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug("Async knowledge context lookup failed", exc_info=True)
        return []


def _broadcast_workspace_knowledge(query: str, snippets: List[str]) -> None:
    preview = snippets[0] if snippets else ""
    summary = preview if len(snippets) == 1 else f"{len(snippets)} snippets retrieved"
    try:
        global_workspace.publish_message(
            WorkspaceMessage(
                type="knowledge.retrieval",
                source="knowledge",
                payload={"query": query, "snippets": snippets},
                summary=summary,
                tags=("knowledge", "retrieval"),
                importance=min(1.0, 0.25 + 0.05 * len(snippets)),
            ),
            propagate=True,
        )
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug("Failed to publish knowledge retrieval to workspace.", exc_info=True)


# --------------------------------------------------------------------------- #
# Knowledge base helpers
# --------------------------------------------------------------------------- #


def _format_kb_result(
    result: Any,
    *,
    top_k: int,
    relation_limit: int,
    max_line: int,
) -> List[str]:
    if not isinstance(result, dict):
        return []

    snippets: List[str] = []
    causal = result.get("causal_relations")
    entries = [(k, v) for k, v in result.items() if k != "causal_relations"]

    for idx, (key, value) in enumerate(entries):
        if top_k and idx >= top_k:
            break
        label = key.replace(":", " -> ")
        summary = _to_single_line(value, max_line)
        if summary:
            snippets.append(f"{label}: {summary}")

    if isinstance(causal, Iterable):
        for index, relation in enumerate(causal):
            if index >= relation_limit:
                break
            cause, effect, weight = _extract_relation(relation)
            if not cause and not effect:
                continue
            line = f"{cause} -> {effect}" if effect else cause
            if weight is not None:
                if isinstance(weight, (int, float)):
                    line += f" (weight={weight:.2f})"
                else:
                    line += f" (weight={weight})"
            snippets.append(line)

    return snippets


def _extract_relation(relation: Any) -> tuple[str, str, Any]:
    if hasattr(relation, "cause") or hasattr(relation, "effect"):
        return (
            str(getattr(relation, "cause", "")).strip(),
            str(getattr(relation, "effect", "")).strip(),
            getattr(relation, "weight", None),
        )
    if isinstance(relation, dict):
        return (
            str(relation.get("cause", "")).strip(),
            str(relation.get("effect", "")).strip(),
            relation.get("weight"),
        )
    return str(relation), "", None


# --------------------------------------------------------------------------- #
# Graph lookup helpers
# --------------------------------------------------------------------------- #


def _lookup_graph(
    query: str,
    *,
    top_k: int,
    relation_limit: int,
    max_line: int,
) -> List[str]:
    graph = get_graph_store_instance()
    snapshot = graph.get_snapshot()
    nodes = snapshot.get("nodes", [])
    if not nodes:
        return []

    node_map = {node.id: node for node in nodes}
    edges = snapshot.get("edges", [])
    terms = _tokenize(query)

    scored: List[tuple[int, Any]] = []
    for node in nodes:
        label = _label_for_node(node)
        text_parts = [label.lower()]
        for value in (node.properties or {}).values():
            if isinstance(value, str):
                text_parts.append(value.lower())
        text = " ".join(text_parts)
        score = sum(1 for term in terms if term in text)
        if score:
            scored.append((score, node))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    snippets: List[str] = []

    for score, node in scored[:top_k]:
        label = _label_for_node(node)
        summary = _summarize_properties(node.properties, max_line=max_line)
        node_type = getattr(node.type, "name", str(node.type))
        base_line = f"{label} [{node_type}]"
        if summary:
            base_line += f": {summary}"
        snippets.append(base_line)
        snippets.extend(
            _format_edges(
                node.id,
                label,
                edges,
                node_map,
                relation_limit=relation_limit,
                max_line=max_line,
            )
        )
    return snippets


def _format_edges(
    node_id: str,
    node_label: str,
    edges: Iterable[Any],
    node_map: Dict[str, Any],
    *,
    relation_limit: int,
    max_line: int,
) -> List[str]:
    formatted: List[tuple[float, str]] = []
    for edge in edges:
        if edge.source != node_id and edge.target != node_id:
            continue
        other_id = edge.target if edge.source == node_id else edge.source
        other_label = _label_for_node(node_map.get(other_id), default=other_id)
        rel_type = getattr(edge.type, "value", None)
        if rel_type is None:
            rel_type = getattr(edge.type, "name", str(edge.type))
        arrow = "->" if edge.source == node_id else "<-"
        line = f"  - {node_label} {arrow} {rel_type} {other_label}"
        weight = edge.properties.get("weight") if isinstance(edge.properties, dict) else None
        if weight is not None:
            line += f" (weight={weight})"
        score = float(weight) if isinstance(weight, (int, float)) else 0.0
        formatted.append((score, _truncate(line, max_line)))

    formatted.sort(key=lambda item: item[0], reverse=True)
    return [line for _, line in formatted[:relation_limit]]


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def _summarize_properties(properties: Optional[Dict[str, Any]], *, max_line: int) -> str:
    if not properties:
        return ""
    parts: List[str] = []
    for key, value in properties.items():
        if key in {"label", "name"}:
            continue
        if isinstance(value, (str, int, float)):
            text = _truncate(" ".join(str(value).split()), max_line // 2)
            parts.append(f"{key}={text}")
        elif isinstance(value, list):
            parts.append(f"{key}={len(value)} items")
        elif isinstance(value, dict):
            parts.append(f"{key}=object")
        if len(parts) >= 3:
            break
    return "; ".join(parts)


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[: max_length - 3].rstrip() + "..."


def _label_for_node(node: Any, default: str | None = None) -> str:
    if node is None:
        return default or ""
    properties = getattr(node, "properties", {}) or {}
    return str(
        properties.get("label")
        or properties.get("name")
        or getattr(node, "id", default)
        or default
        or ""
    )


def _tokenize(query: str) -> List[str]:
    tokens = [token.lower() for token in re.split(r"\W+", query) if len(token) > 2]
    if not tokens:
        tokens = [query.lower()]
    return tokens


def _to_single_line(value: Any, max_length: int) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = " ".join(value.split())
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
    return _truncate(text, max_length)
