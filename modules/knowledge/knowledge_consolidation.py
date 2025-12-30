"""External knowledge consolidation helpers.

This module converts short-lived knowledge-acquisition payloads (web search,
documentation snippets, code-index hits, etc.) into persistent memory:

- Symbolic: store lightweight `KnowledgeFact` relations in the knowledge graph.
- Vector: store a compact episode/learning summary in the vector memory store.

Design goals:
- Dependency-light: works without an LLM; summaries are heuristic by default.
- Safe & bounded: aggressively clips content and metadata to avoid runaway growth.
- Pluggable: callers may inject a `LongTermMemoryCoordinator`.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

try:  # pragma: no cover - prefer full stack when available
    from .long_term_memory import LongTermMemoryCoordinator
    from .runtime_importer import KnowledgeFact
except Exception:  # pragma: no cover - fallback exports from modules.knowledge.__init__
    from modules.knowledge import KnowledgeFact, LongTermMemoryCoordinator  # type: ignore


def _clip_text(value: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _stable_id(text: str, *, prefix: str = "task") -> str:
    payload = (text or "").encode("utf-8", errors="ignore")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quality_level_from_score(score: float | None, *, has_signals: bool) -> str:
    if not has_signals or score is None:
        return "unknown"
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _host_from_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        host = str(urlparse(raw).hostname or "")
    except Exception:
        host = ""
    return host.strip().lower().strip(".")


def _compute_source_quality(
    acquisitions: Sequence[Dict[str, Any]],
    references: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute a lightweight quality signal to help prevent knowledge pollution.

    Heuristic signals (best-effort):
    - Web trust: average trust_score from web_search results, or tool-level avg_trust.
    - Doc consensus: documentation_tool consensus metadata (host diversity + similarity).
    """

    web_avg_trusts: List[float] = []
    web_unique_hosts: List[int] = []
    web_warnings: set[str] = set()

    doc_levels: List[str] = []
    doc_similarity: List[float] = []
    doc_avg_trusts: List[float] = []
    doc_unique_hosts: List[int] = []
    doc_warnings: set[str] = set()

    for payload in acquisitions or []:
        web_search = payload.get("web_search") if isinstance(payload, dict) else None
        if isinstance(web_search, Mapping):
            info = web_search.get("info")
            if isinstance(info, Mapping):
                avg_trust = _safe_float(info.get("avg_trust"))
                if avg_trust is not None:
                    web_avg_trusts.append(max(0.0, min(1.0, avg_trust)))
                unique_hosts = info.get("unique_hosts")
                try:
                    web_unique_hosts.append(int(unique_hosts))
                except Exception:
                    pass
                trust_counts = info.get("trust_counts")
                if isinstance(trust_counts, Mapping):
                    try:
                        if int(trust_counts.get("low", 0)) > 0:
                            web_warnings.add("low_trust_source")
                    except Exception:
                        pass
            if web_unique_hosts and max(web_unique_hosts) < 2:
                web_warnings.add("single_host")

        doc = payload.get("web") if isinstance(payload, dict) else None
        if isinstance(doc, Mapping):
            info = doc.get("info")
            consensus = info.get("consensus") if isinstance(info, Mapping) else None
            if isinstance(consensus, Mapping):
                level = str(consensus.get("level") or "").strip().lower()
                if level:
                    doc_levels.append(level)
                sim = _safe_float(consensus.get("similarity_avg"))
                if sim is not None:
                    doc_similarity.append(sim)
                avg_trust = _safe_float(consensus.get("avg_trust"))
                if avg_trust is not None:
                    doc_avg_trusts.append(max(0.0, min(1.0, avg_trust)))
                unique_hosts = consensus.get("unique_hosts")
                try:
                    doc_unique_hosts.append(int(unique_hosts))
                except Exception:
                    pass
                warnings = consensus.get("warnings")
                if isinstance(warnings, list):
                    for item in warnings:
                        tag = str(item or "").strip()
                        if tag:
                            doc_warnings.add(tag)
                if consensus.get("needs_verification"):
                    doc_warnings.add("needs_verification")

    ref_trust_scores: List[float] = []
    ref_hosts: set[str] = set()
    for ref in references or []:
        if not isinstance(ref, Mapping):
            continue
        host = str(ref.get("host") or "").strip().lower().strip(".")
        if not host:
            host = _host_from_url(str(ref.get("url") or ""))
        if host:
            ref_hosts.add(host)
        trust_score = _safe_float(ref.get("trust_score"))
        if trust_score is not None:
            ref_trust_scores.append(max(0.0, min(1.0, trust_score)))

    trust_avg: float | None = None
    if ref_trust_scores:
        trust_avg = sum(ref_trust_scores) / len(ref_trust_scores)
    elif web_avg_trusts:
        trust_avg = sum(web_avg_trusts) / len(web_avg_trusts)
    elif doc_avg_trusts:
        trust_avg = sum(doc_avg_trusts) / len(doc_avg_trusts)

    has_signals = bool(ref_trust_scores or web_avg_trusts or doc_levels or doc_avg_trusts)
    warnings: set[str] = set()
    warnings.update(web_warnings)
    warnings.update(doc_warnings)
    if trust_avg is not None and trust_avg < 0.4:
        warnings.add("low_trust_source")
    if (doc_unique_hosts and max(doc_unique_hosts) < 2) or (web_unique_hosts and max(web_unique_hosts) < 2):
        warnings.add("single_host")
    if len(ref_hosts) == 1 and has_signals:
        warnings.add("single_host")

    # Worst-case consensus level (if any).
    level_rank = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
    worst_doc_level = None
    if doc_levels:
        unique_levels = {lvl if lvl in level_rank else "unknown" for lvl in doc_levels}
        worst_doc_level = min(unique_levels, key=lambda lvl: level_rank.get(lvl, 0))

    score = trust_avg if trust_avg is not None else 0.5
    if worst_doc_level is not None:
        if worst_doc_level == "high":
            score = max(score, 0.75)
        elif worst_doc_level == "medium":
            score = max(score, 0.6)
        elif worst_doc_level == "unknown":
            score = min(score, 0.5)
        else:  # low
            score = min(score, 0.35)

    needs_verification = bool(warnings)
    if needs_verification:
        score = min(score, 0.45)
    if trust_avg is not None and trust_avg < 0.4:
        score = min(score, 0.35)
    score = max(0.0, min(1.0, float(score)))

    level = _quality_level_from_score(score, has_signals=has_signals)
    return {
        "level": level,
        "score": round(score, 3),
        "needs_verification": bool(needs_verification),
        "warnings": sorted({w for w in warnings if w}),
        "signals": {
            "reference_trust_avg": round(trust_avg, 3) if trust_avg is not None else None,
            "doc_consensus_level": worst_doc_level,
            "doc_similarity_avg": round(sum(doc_similarity) / len(doc_similarity), 3) if doc_similarity else None,
            "unique_hosts": int(len(ref_hosts)) if ref_hosts else None,
        },
    }


@dataclass(frozen=True)
class KnowledgeConsolidationConfig:
    enabled: bool = True
    max_summary_chars: int = 6000
    max_context_chars: int = 4000
    max_references: int = 12
    max_channels: int = 6
    ingest_graph: bool = True
    store_vector_summary: bool = True


class ExternalKnowledgeConsolidator:
    """Persist external knowledge acquisition outcomes into long-term memory."""

    def __init__(
        self,
        *,
        memory: LongTermMemoryCoordinator | None = None,
        config: KnowledgeConsolidationConfig | None = None,
    ) -> None:
        self.config = config or KnowledgeConsolidationConfig()
        self.memory = memory or LongTermMemoryCoordinator()

    def consolidate(
        self,
        *,
        goal: str,
        knowledge_acquisition: Sequence[Mapping[str, Any]] | None,
        success: bool,
        human_reward: float | None = None,
        task_metadata: Mapping[str, Any] | None = None,
        source: str = "external_knowledge",
    ) -> Dict[str, Any]:
        """Consolidate a batch of knowledge-acquisition payloads.

        Parameters
        ----------
        goal:
            The task goal text.
        knowledge_acquisition:
            Payloads produced by `AutonomousTaskExecutor` knowledge acquisition.
        success:
            Whether the overall task succeeded.
        human_reward:
            Optional RLHF-style score in [0, 1] (best-effort).
        task_metadata:
            Optional extra metadata to store alongside the summary.
        """

        cfg = self.config
        if not cfg.enabled:
            return {"stored": False, "reason": "disabled"}

        acquisitions = [dict(p) for p in (knowledge_acquisition or []) if isinstance(p, Mapping)]
        if not acquisitions:
            return {"stored": False, "reason": "no_knowledge_acquisition"}

        run_id = uuid.uuid4().hex[:12]
        task_id = _stable_id(goal, prefix="task")
        now = time.time()

        summary_text, summary_meta, facts = self._build_summary_and_facts(
            goal=goal,
            task_id=task_id,
            run_id=run_id,
            acquisitions=acquisitions,
            success=success,
            human_reward=human_reward,
            task_metadata=task_metadata,
            timestamp=now,
            source=source,
        )

        errors: List[str] = []
        vector_id: str | None = None
        if cfg.store_vector_summary and summary_text:
            try:
                vector_id = self.memory.vector_store.add_text(summary_text, summary_meta)
            except Exception as exc:  # pragma: no cover - best effort
                errors.append(f"vector_store_failed:{exc!r}")

        facts_imported = 0
        if cfg.ingest_graph and facts:
            try:
                # Keep vector storage separate: embed only the summary text to
                # reduce vector-store growth.
                self.memory.record_facts(
                    facts,
                    embed=False,
                    base_metadata=dict(summary_meta),
                )
                facts_imported = len(facts)
            except Exception as exc:  # pragma: no cover - best effort
                errors.append(f"graph_ingest_failed:{exc!r}")

        return {
            "stored": bool(vector_id or facts_imported),
            "task_id": task_id,
            "run_id": run_id,
            "vector_id": vector_id,
            "facts_imported": int(facts_imported),
            "summary_chars": int(len(summary_text)),
            "errors": errors,
        }

    # ------------------------------------------------------------------
    def _build_summary_and_facts(
        self,
        *,
        goal: str,
        task_id: str,
        run_id: str,
        acquisitions: Sequence[Dict[str, Any]],
        success: bool,
        human_reward: float | None,
        task_metadata: Mapping[str, Any] | None,
        timestamp: float,
        source: str,
    ) -> tuple[str, Dict[str, Any], List[KnowledgeFact]]:
        cfg = self.config
        goal_text = str(goal or "").strip()
        goal_clip = _clip_text(goal_text, max_chars=700)

        all_channels: List[str] = []
        all_refs: List[Dict[str, Any]] = []
        queries: List[str] = []
        contexts: List[str] = []
        domains: List[str] = []

        for payload in acquisitions:
            q = str(payload.get("query") or "").strip()
            if q:
                queries.append(_clip_text(q, max_chars=700))

            channels = payload.get("channels_used")
            for channel in _as_list(channels):
                ch = str(channel or "").strip()
                if ch and ch not in all_channels:
                    all_channels.append(ch)

            refs = payload.get("references")
            for ref in _as_list(refs):
                if isinstance(ref, Mapping) and ref.get("url"):
                    all_refs.append(dict(ref))

            ctx = str(payload.get("retrieval_context") or "").strip()
            if ctx:
                contexts.append(_clip_text(ctx, max_chars=cfg.max_context_chars))

            meta_policy = payload.get("meta_policy")
            if isinstance(meta_policy, Mapping) and meta_policy.get("domain"):
                domains.append(str(meta_policy.get("domain")))

        channels_used = all_channels[: cfg.max_channels]
        references = all_refs[: cfg.max_references]
        source_quality = _compute_source_quality(acquisitions, references)

        domain = next((d for d in domains if d), "general")
        primary_query = next((q for q in queries if q), goal_clip)

        success_text = "success" if bool(success) else "failure"
        reward_unit: float | None = None
        if human_reward is not None:
            try:
                reward_unit = max(0.0, min(1.0, float(human_reward)))
            except Exception:
                reward_unit = None

        ref_lines: List[str] = []
        for ref in references:
            url = str(ref.get("url") or "").strip()
            title = str(ref.get("title") or "").strip()
            if not url:
                continue
            label = title or url
            ref_lines.append(f"- {label} ({url})")

        summary_parts: List[str] = [
            f"Goal: {goal_clip}",
            f"Query: {primary_query}",
            f"Outcome: {success_text}",
            f"Domain: {domain}",
        ]
        if isinstance(source_quality, dict) and (
            bool(source_quality.get("needs_verification"))
            or str(source_quality.get("level") or "").strip().lower() not in {"", "unknown"}
        ):
            quality_line = (
                f"SourceQuality: {str(source_quality.get('level') or 'unknown')}"
                f" score={float(source_quality.get('score', 0.5)):.3f}"
            )
            if source_quality.get("needs_verification"):
                quality_line += " needs_verification"
            warns = source_quality.get("warnings")
            if isinstance(warns, list) and warns:
                clipped = ",".join(str(item) for item in warns[:4] if item)
                if clipped:
                    quality_line += f" warnings={clipped}"
            summary_parts.append(quality_line)
        if reward_unit is not None:
            summary_parts.append(f"HumanReward: {reward_unit:.3f}")
        if channels_used:
            summary_parts.append(f"Channels: {', '.join(channels_used)}")
        if ref_lines:
            summary_parts.append("References:\n" + "\n".join(ref_lines))
        if contexts:
            summary_parts.append("Context:\n" + "\n\n".join(contexts[:2]))

        summary_text = _clip_text("\n".join(summary_parts).strip(), max_chars=cfg.max_summary_chars)

        summary_meta: Dict[str, Any] = {
            "source": str(source),
            "task_id": task_id,
            "run_id": run_id,
            "goal": goal_clip,
            "query": primary_query,
            "domain": domain,
            "success": bool(success),
            "timestamp": float(timestamp),
            "channels_used": list(channels_used),
            "references": [
                {
                    "url": str(ref.get("url") or ""),
                    "title": str(ref.get("title") or ""),
                    "source": str(ref.get("source") or ""),
                    **{
                        key: ref.get(key)
                        for key in (
                            "host",
                            "trust",
                            "trust_score",
                            "license_spdx",
                            "license_copyleft",
                        )
                        if key in ref and isinstance(ref.get(key), (str, int, float, bool))
                    },
                }
                for ref in references
            ],
        }
        if isinstance(source_quality, dict) and source_quality:
            summary_meta["source_quality"] = dict(source_quality)
            if source_quality.get("needs_verification"):
                summary_meta["needs_verification"] = True
                if not bool(success):
                    summary_meta["archived"] = True
                    summary_meta["archived_reason"] = "needs_verification_on_failure"
        if reward_unit is not None:
            summary_meta["human_reward"] = float(reward_unit)
        if task_metadata:
            # Keep it shallow and bounded.
            extra: Dict[str, Any] = {}
            for key, value in dict(task_metadata).items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    extra[str(key)] = value
            if extra:
                summary_meta["task_meta"] = extra

        facts: List[KnowledgeFact] = []
        summary_node = f"external_summary:{task_id}:{run_id}"
        confidence_unit = reward_unit
        if confidence_unit is None and isinstance(source_quality, dict):
            try:
                confidence_unit = float(source_quality.get("score"))
            except Exception:
                confidence_unit = None

        facts.append(
            KnowledgeFact(
                subject=primary_query,
                predicate="external_knowledge_summary",
                obj=summary_node,
                subject_description=goal_clip,
                object_description=_clip_text(summary_text, max_chars=min(cfg.max_summary_chars, 24_000)),
                metadata={
                    "domain": domain,
                    "success": bool(success),
                    "human_reward": reward_unit,
                    "channels_used": list(channels_used),
                    "reference_count": len(references),
                    "needs_verification": bool(summary_meta.get("needs_verification")),
                },
                confidence=confidence_unit,
                source=str(source),
                context=goal_clip,
                timestamp=float(timestamp),
            )
        )

        facts.append(
            KnowledgeFact(
                subject=primary_query,
                predicate="external_task_outcome",
                obj=success_text,
                subject_description=goal_clip,
                metadata={"domain": domain},
                confidence=confidence_unit,
                source=str(source),
                context=goal_clip,
                timestamp=float(timestamp),
            )
        )

        for channel in channels_used:
            facts.append(
                KnowledgeFact(
                    subject=primary_query,
                    predicate="external_retrieval_channel",
                    obj=str(channel),
                    subject_description=goal_clip,
                    metadata={"domain": domain},
                    confidence=confidence_unit,
                    source=str(source),
                    context=goal_clip,
                    timestamp=float(timestamp),
                )
            )

        for ref in references:
            url = str(ref.get("url") or "").strip()
            if not url:
                continue
            title = str(ref.get("title") or "").strip()
            ref_source = str(ref.get("source") or "").strip() or "reference"
            facts.append(
                KnowledgeFact(
                    subject=primary_query,
                    predicate="external_reference",
                    obj=url,
                    subject_description=goal_clip,
                    object_description=title,
                    metadata={"domain": domain, "source_kind": ref_source},
                    confidence=confidence_unit,
                    source=str(source),
                    context=goal_clip,
                    timestamp=float(timestamp),
                )
            )

        return summary_text, summary_meta, facts


__all__ = [
    "KnowledgeConsolidationConfig",
    "ExternalKnowledgeConsolidator",
]
