from __future__ import annotations

import logging
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
    from backend.autogpt.autogpt.core.ability.schema import AbilityResult as AbilityResultType
else:  # pragma: no cover - runtime fallback avoids heavy imports
    AbilityResultType = Any

logger = logging.getLogger(__name__)

_SUMMARY_LIMIT = 240


def _summarise_text(text: str, limit: int = _SUMMARY_LIMIT) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= limit:
        return compact
    suffix = "..."
    cutoff = max(0, int(limit) - len(suffix))
    return compact[:cutoff].rstrip() + suffix


def _normalise_sequence(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (str, bytes)):
        return [value.decode() if isinstance(value, bytes) else value]
    try:
        return [str(item) for item in value if item]
    except TypeError:
        return [str(value)]


@dataclass
class SearchStep:
    channel: str
    keywords: list[str]
    ability: Optional[str] = None
    status: str = "pending"

    def mark(self, status: str) -> None:
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "keywords": list(self.keywords),
            "ability": self.ability,
            "status": self.status,
        }


@dataclass
class KnowledgeAcquisitionPlan:
    session_id: str
    trigger: str
    query: str
    steps: list[SearchStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "trigger": self.trigger,
            "query": self.query,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class KnowledgeAcquisitionSession:
    plan: KnowledgeAcquisitionPlan
    ability_name: str
    ability_arguments: Dict[str, Any]
    task_snapshot: Dict[str, Any]
    metadata_snapshot: Dict[str, Any]
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    success: Optional[bool] = None
    response: str = ""
    evidence: list[Dict[str, Any]] = field(default_factory=list)

    def mark_started(self) -> None:
        self.started_at = time.time()
        for step in self.plan.steps:
            step.mark("in_progress")

    def complete(self, *, success: bool, response: str) -> None:
        self.completed_at = time.time()
        self.success = success
        self.response = response or ""
        for step in self.plan.steps:
            step.mark("completed" if success else "failed")

    def to_log(self) -> Dict[str, Any]:
        return {
            "event": "knowledge_acquisition_completed",
            "session_id": self.plan.session_id,
            "ability": self.ability_name,
            "ability_arguments": dict(self.ability_arguments),
            "plan": self.plan.to_dict(),
            "task": dict(self.task_snapshot),
            "success": self.success,
            "response": self.response,
            "response_summary": _summarise_text(self.response),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "evidence": list(self.evidence),
            "metadata": dict(self.metadata_snapshot),
        }


class KnowledgeAcquisitionManager:
    """Decide when and how to trigger knowledge acquisition abilities."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.45,
        search_keywords: Sequence[str] | None = None,
        fallback_keywords: Sequence[str] | None = None,
        history_size: int = 128,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._search_keywords = tuple(search_keywords or ("search", "lookup", "web", "browser"))
        self._fallback_keywords = tuple(fallback_keywords or ("request_human", "ask_human", "human_feedback"))
        self._sessions: Dict[str, KnowledgeAcquisitionSession] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=max(1, history_size))

    def maybe_acquire(
        self,
        *,
        metadata: Optional[Dict[str, Any]],
        ability_specs: Sequence[Any],
        task: Any,
        current_selection: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        metadata_dict = self._normalise_metadata(metadata)
        if not metadata_dict:
            return None
        if current_selection and current_selection.get("knowledge_acquisition"):
            return None

        needs = bool(metadata_dict.get("needs_knowledge"))
        confidence_value = self._extract_confidence(metadata_dict.get("confidence"))

        if not needs:
            if confidence_value is None:
                return None
            if confidence_value >= self._confidence_threshold:
                return None

        query = (
            metadata_dict.get("knowledge_query")
            or metadata_dict.get("question")
            or getattr(task, "description", None)
            or getattr(task, "objective", "")
        )
        query = str(query or "").strip() or "Current task context"

        url = self._extract_url(metadata_dict, query)
        ability = None
        channel = "search"
        if url:
            ability = self._find_ability(
                ability_specs,
                ("scrape", "fetch", "download", "crawl", "web"),
                require_any_params=("url",),
            )
            channel = "scrape"

        if ability is None:
            ability = self._find_ability(
                ability_specs,
                self._search_keywords,
                require_any_params=("query", "prompt", "question", "input", "text"),
            )
            channel = "search"

        if ability is None:
            ability = self._find_ability(
                ability_specs,
                self._fallback_keywords,
                require_any_params=("query", "prompt", "question", "input", "text"),
            )
            channel = "human"

        if ability is None:
            logger.debug("Knowledge acquisition requested but no search/human ability available.")
            return None

        ability_name = self._ability_name(ability)
        arguments = self._build_arguments(ability, query, metadata_dict, url=url or None)
        reason = metadata_dict.get("reason") or self._build_reason(needs, confidence_value)

        session = self._initialise_session(
            ability_spec=ability,
            ability_args=arguments,
            metadata=metadata_dict,
            task=task,
            reason=reason,
            query=query,
            channel=channel,
        )

        return {
            "backend": "knowledge_acquisition",
            "next_ability": ability_name,
            "ability_arguments": arguments,
            "knowledge_acquisition": True,
            "reason": reason,
            "knowledge_session_id": session.plan.session_id,
            "knowledge_plan": session.plan.to_dict(),
        }

    def mark_session_started(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.mark_started()

    def record_evidence(
        self,
        session_id: str,
        *,
        source: str,
        content: str,
        confidence: Optional[float] = None,
    ) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.evidence.append(
            {
                "source": source,
                "content": content,
                "confidence": confidence,
                "captured_at": time.time(),
            }
        )
        return True

    def complete_session(
        self,
        session_id: str,
        ability_result: AbilityResultType | Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        memory: Any = None,
    ) -> Optional[Dict[str, Any]]:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return None

        success = bool(getattr(ability_result, "success", False))
        response = getattr(ability_result, "message", "") or ""
        session.complete(success=success, response=response)
        log_entry = session.to_log()
        log_entry["result_metadata"] = self._normalise_metadata(metadata)

        knowledge = getattr(ability_result, "new_knowledge", None)
        if knowledge is not None:
            try:
                log_entry["new_knowledge"] = knowledge.dict()
            except Exception:  # pragma: no cover - defensive, depends on pydantic internals
                log_entry["new_knowledge"] = {"content": getattr(knowledge, "content", None)}

        self._history.append(log_entry)

        if memory is not None:
            try:
                memory.add(self._render_memory_line(log_entry))
            except Exception:  # pragma: no cover - memory backend failures should not crash agent
                logger.debug("Unable to record knowledge acquisition in memory.", exc_info=True)
            knowledge_content = getattr(knowledge, "content", None)
            if isinstance(knowledge_content, str) and knowledge_content.strip():
                try:
                    source = None
                    meta = getattr(knowledge, "content_metadata", None)
                    if isinstance(meta, dict):
                        source = meta.get("source") or meta.get("url")
                    snippet = _summarise_text(knowledge_content, limit=600)
                    if snippet:
                        prefix = f"External knowledge ({source}): " if source else "External knowledge: "
                        memory.add(prefix + snippet)
                except Exception:  # pragma: no cover - never crash on memory issues
                    logger.debug("Unable to record knowledge payload in memory.", exc_info=True)

        return log_entry

    def history(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        if limit is None or limit >= len(self._history):
            return list(self._history)
        return list(self._history)[-limit:]

    # ------------------------------------------------------------------ helpers
    def _find_ability(
        self,
        ability_specs: Sequence[Any],
        keywords: Sequence[str],
        *,
        require_any_params: Sequence[str] | None = None,
    ) -> Optional[Any]:
        best_spec: Optional[Any] = None
        best_score = 0.0
        for spec in ability_specs:
            name = self._ability_name(spec)
            lowered = name.lower()

            if require_any_params:
                params = self._ability_parameter_keys(spec)
                if params and not any(p in params for p in require_any_params):
                    continue

            score = self._score_keyword_match(lowered, keywords)
            if score <= 0:
                continue

            if "search" in lowered and "scrape" in lowered:
                score += 2.0

            tags = getattr(spec, "tags", None)
            if tags:
                try:
                    tags_text = " ".join(str(tag).lower() for tag in tags)
                except Exception:
                    tags_text = ""
                score += self._score_keyword_match(tags_text, keywords) * 0.25

            if score > best_score:
                best_score = score
                best_spec = spec

        return best_spec

    def _score_keyword_match(self, haystack: str, keywords: Sequence[str]) -> float:
        if not haystack:
            return 0.0
        if not keywords:
            return 0.0
        normalized = re.sub(r"[^a-z0-9_]+", " ", haystack.lower())
        tokens = set(token for token in re.split(r"[\s_]+", normalized) if token)
        score = 0.0
        total = max(1, len(keywords))
        for idx, keyword in enumerate(keywords):
            kw = str(keyword).lower().strip()
            if not kw:
                continue
            weight = float(total - idx)
            if kw in tokens:
                score += weight
            elif kw in haystack:
                score += weight * 0.6
        return score

    def _ability_name(self, spec: Any) -> str:
        name = getattr(spec, "name", None)
        if isinstance(name, str):
            return name
        if isinstance(spec, dict):
            value = spec.get("name")
            if isinstance(value, str):
                return value
        return str(spec)

    def _build_arguments(
        self,
        spec: Any,
        query: str,
        metadata: Dict[str, Any],
        *,
        url: str | None = None,
    ) -> Dict[str, Any]:
        candidates = ["query", "prompt", "question", "input", "text"]
        parameters = getattr(spec, "parameters", None) or (
            spec.get("parameters") if isinstance(spec, dict) else None
        )
        schema = parameters if isinstance(parameters, dict) else {}
        keys = {str(k) for k in schema.keys()}
        required_keys = {str(k) for k, v in schema.items() if getattr(v, "required", False) is True}

        args: Dict[str, Any] = {}

        if url and "url" in keys:
            args["url"] = url
        else:
            for key in candidates:
                if key in keys:
                    args[key] = query
                    break
            else:
                if "query" in keys:
                    args["query"] = query
                elif len(required_keys) == 1:
                    args[next(iter(required_keys))] = query
                elif len(keys) == 1:
                    args[next(iter(keys))] = query
                else:
                    args["query"] = query

        if "context" in keys:
            context = metadata.get("context") or {}
            if isinstance(context, dict):
                args.setdefault("context", context)
            else:
                args.setdefault("context", metadata)
        return args

    def _ability_parameter_keys(self, spec: Any) -> set[str]:
        parameters = getattr(spec, "parameters", None)
        if isinstance(parameters, dict):
            return {str(k) for k in parameters.keys()}
        if isinstance(spec, dict) and isinstance(spec.get("parameters"), dict):
            return {str(k) for k in spec.get("parameters", {}).keys()}
        return set()

    def _extract_url(self, metadata: Dict[str, Any], query: str) -> str:
        candidates = [
            metadata.get("knowledge_url"),
            metadata.get("url"),
            metadata.get("source_url"),
            metadata.get("link"),
        ]
        context = metadata.get("context")
        if isinstance(context, dict):
            candidates.extend([context.get("url"), context.get("source_url")])
        for value in candidates:
            url = str(value or "").strip()
            if url.startswith("http://") or url.startswith("https://"):
                return url
        query_text = str(query or "").strip()
        if query_text.startswith("http://") or query_text.startswith("https://"):
            return query_text
        return ""

    def _initialise_session(
        self,
        *,
        ability_spec: Any,
        ability_args: Dict[str, Any],
        metadata: Dict[str, Any],
        task: Any,
        reason: str,
        query: str,
        channel: str,
    ) -> KnowledgeAcquisitionSession:
        session_id = uuid.uuid4().hex
        ability_name = self._ability_name(ability_spec)
        keywords = self._derive_keywords(query, metadata)
        step = SearchStep(channel=channel, keywords=keywords, ability=ability_name)
        plan_metadata = {
            "confidence": self._extract_confidence(metadata.get("confidence")),
            "needs_knowledge": bool(metadata.get("needs_knowledge")),
        }
        if "concept" in metadata:
            plan_metadata["concept"] = metadata["concept"]
        plan = KnowledgeAcquisitionPlan(
            session_id=session_id,
            trigger=reason,
            query=query,
            steps=[step],
            metadata=plan_metadata,
        )
        task_snapshot = self._serialise_task(task)
        session = KnowledgeAcquisitionSession(
            plan=plan,
            ability_name=ability_name,
            ability_arguments=dict(ability_args),
            task_snapshot=task_snapshot,
            metadata_snapshot=dict(metadata),
        )
        self._sessions[session_id] = session
        return session

    def _build_reason(self, needs: bool, confidence: Optional[float]) -> str:
        if needs:
            return "Knowledge gap flagged by upstream module."
        if confidence is None:
            return "Unable to score confidence; gathering external knowledge."
        return f"Confidence {confidence:.2f} below threshold {self._confidence_threshold:.2f}."

    def _derive_keywords(self, query: str, metadata: Dict[str, Any]) -> list[str]:
        keywords = []
        for key in ("keywords", "key_terms", "tags"):
            value = metadata.get(key)
            if value:
                keywords.extend(_normalise_sequence(value))
        if not keywords and query:
            cleaned = re.sub(r"[^\w\s]", " ", query)
            keywords = [token for token in cleaned.split() if len(token) > 3][:6]
        return [kw.lower() for kw in keywords[:10]]

    def _serialise_task(self, task: Any) -> Dict[str, Any]:
        if task is None:
            return {}
        snapshot: Dict[str, Any] = {}
        task_id = getattr(task, "id", None)
        if task_id is not None:
            snapshot["id"] = str(task_id)
        description = getattr(task, "description", None) or getattr(task, "objective", None)
        if description:
            snapshot["description"] = str(description)
        objective = getattr(task, "objective", None)
        if objective:
            snapshot["objective"] = str(objective)
        context = getattr(task, "context", None)
        if context is not None:
            for attr in ("status", "cycle_count"):
                value = getattr(context, attr, None)
                if value is not None:
                    snapshot[attr] = value
        return snapshot

    def _extract_confidence(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalise_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(metadata, dict):
            return metadata
        if hasattr(metadata, "items"):
            try:
                return dict(metadata)
            except Exception:  # pragma: no cover
                return {}
        return {}

    def _render_memory_line(self, log_entry: Dict[str, Any]) -> str:
        status = "succeeded" if log_entry.get("success") else "failed"
        plan = log_entry.get("plan", {})
        query = plan.get("query", "unknown query")
        ability = log_entry.get("ability", "unknown ability")
        summary = log_entry.get("response_summary") or ""
        return f"Knowledge acquisition {status}: {query} via {ability}. {summary}"
