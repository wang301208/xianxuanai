from __future__ import annotations

"""Automatic self-correction orchestration.

The project already emits rich failure signals (task failures, action outcomes,
replan directives). Historically, these signals required humans to read logs and
manually decide how to react.

`SelfCorrectionManager` closes that gap by:
- turning failure bursts into structured *remediation cases*
- automatically triggering follow-up actions (knowledge lookup, strategy hints,
  background learning requests) via the existing event bus

It is deliberately conservative: it does not patch code by default; it publishes
diagnostic context and requests an explicit plan that can be executed by an
agent/human loop.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

try:  # optional in some deployments
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EventBus = None  # type: ignore

try:  # optional dependency chain
    from modules.knowledge import ProblemAnalyzer, ResearchTool  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ProblemAnalyzer = None  # type: ignore[assignment]
    ResearchTool = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


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


def _compact(text: Any, limit: int = 240) -> str:
    blob = re.sub(r"\s+", " ", str(text or "")).strip()
    if limit <= 0 or len(blob) <= limit:
        return blob
    return blob[: max(0, limit - 3)] + "..."


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        token = str(item or "").strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out


_EXC_RE = re.compile(r"\b([A-Za-z_]+Error|Exception)\b")
_MISSING_MODULE_RE = re.compile(r"No module named ['\"](?P<name>[^'\"]+)['\"]")


@dataclass(frozen=True)
class CorrectionConfig:
    enabled: bool = True
    cooldown_secs: float = 180.0
    publish_plan: bool = True
    publish_learning_request: bool = True
    docs_search: bool = True
    web_search: bool = False
    max_doc_hits: int = 4
    max_subquestions: int = 5
    add_capability_hints: bool = True


class SelfCorrectionManager:
    """Analyze failure signals and automatically schedule remediation."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        memory_router: Any | None = None,
        performance_monitor: Any | None = None,
        workspace_root: str | os.PathLike[str] | None = None,
        docs_roots: Sequence[str | os.PathLike[str]] | None = None,
        config: Optional[CorrectionConfig] = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")
        self._bus = event_bus
        self._memory_router = memory_router
        self._performance_monitor = performance_monitor
        self._logger = logger_ or logger

        cfg = config or CorrectionConfig(
            enabled=_env_bool("SELF_CORRECTION_ENABLED", True),
            cooldown_secs=_env_float("SELF_CORRECTION_COOLDOWN_SECS", 180.0),
            publish_plan=_env_bool("SELF_CORRECTION_PUBLISH_PLAN", True),
            publish_learning_request=_env_bool("SELF_CORRECTION_REQUEST_LEARNING", True),
            docs_search=_env_bool("SELF_CORRECTION_DOCS_SEARCH", True),
            web_search=_env_bool("SELF_CORRECTION_WEB_SEARCH", False),
            max_doc_hits=_env_int("SELF_CORRECTION_MAX_DOC_HITS", 4),
            max_subquestions=_env_int("SELF_CORRECTION_MAX_SUBQUESTIONS", 5),
            add_capability_hints=_env_bool("SELF_CORRECTION_CAPABILITY_HINTS", True),
        )
        self._config = cfg

        self._workspace_root = Path(workspace_root or os.getenv("SELF_CORRECTION_WORKSPACE_ROOT") or Path.cwd()).resolve()
        self._docs_roots = tuple(Path(p).resolve() for p in (docs_roots or self._parse_docs_roots_env() or (self._workspace_root,)))
        self._last_trigger_ts: float | None = None

        self._subscriptions: list[Callable[[], None]] = [
            self._bus.subscribe("diagnostics.self_debug", self._on_self_debug_case),
            self._bus.subscribe("task_manager.task_completed", self._on_task_manager_completed),
        ]

    def close(self) -> None:
        subs = list(self._subscriptions)
        self._subscriptions.clear()
        for cancel in subs:
            try:
                cancel()
            except Exception:
                continue

    # ------------------------------------------------------------------
    async def _on_self_debug_case(self, event: Dict[str, Any]) -> None:
        if not self._config.enabled or not isinstance(event, Mapping):
            return
        now = float(event.get("time", time.time()) or time.time())
        self._handle_case(dict(event), now=now, source="self_debug")

    async def _on_task_manager_completed(self, event: Dict[str, Any]) -> None:
        if not self._config.enabled or not isinstance(event, Mapping):
            return
        status = str(event.get("status") or "").lower()
        if status in {"completed", "success"}:
            return
        now = float(event.get("time", time.time()) or time.time())
        pseudo_case = {
            "time": now,
            "trigger": "task_failure",
            "hint": "task_manager",
            "counts": {"failures": 1, "replans": 0, "bad_plans": 0},
            "last_failure": {
                "time": now,
                "source": "task_manager",
                "task_id": event.get("task_id"),
                "name": event.get("name"),
                "category": event.get("category"),
                "reason": event.get("error") or status,
                "autofix": event.get("autofix"),
            },
            "evidence": [
                {
                    "time": now,
                    "source": "task_manager",
                    "task_id": event.get("task_id"),
                    "name": event.get("name"),
                    "category": event.get("category"),
                    "reason": event.get("error") or status,
                    "autofix": event.get("autofix"),
                }
            ],
        }
        self._handle_case(pseudo_case, now=now, source="task_manager")

    # ------------------------------------------------------------------
    def _handle_case(self, case: Dict[str, Any], *, now: float, source: str) -> None:
        if self._cooldown_active(now):
            return
        remediation = self._build_remediation(case, now=now, source=source)
        self._persist(remediation)
        self._publish(remediation)
        self._maybe_request_learning(remediation)
        self._maybe_publish_plan(remediation)
        self._last_trigger_ts = float(now)

    def _cooldown_active(self, now: float) -> bool:
        if self._config.cooldown_secs <= 0:
            return False
        if self._last_trigger_ts is None:
            return False
        return (float(now) - float(self._last_trigger_ts)) < float(self._config.cooldown_secs)

    def _build_remediation(self, case: Mapping[str, Any], *, now: float, source: str) -> Dict[str, Any]:
        trigger = str(case.get("trigger") or "unknown")
        last_failure = case.get("last_failure")
        reason = ""
        action = ""
        autofix = None
        if isinstance(last_failure, Mapping):
            reason = str(last_failure.get("reason") or "")
            action = str(last_failure.get("action") or last_failure.get("command") or "")
            autofix = last_failure.get("autofix")

        classification = self._classify(trigger=trigger, reason=reason, autofix=autofix)
        capabilities = self._capability_hints(trigger=trigger, reason=reason, classification=classification)

        query = _compact(" ".join(token for token in (reason, action, trigger) if token), limit=320)
        evidence: list[dict] = []
        doc_hits: list[dict] = []
        web_hits: list[dict] = []
        sub_questions: list[str] = []

        if isinstance(case.get("evidence"), list):
            for entry in case.get("evidence")[-10:]:
                if isinstance(entry, Mapping):
                    evidence.append(dict(entry))

        if self._config.docs_search and ResearchTool is not None:
            try:
                tool = ResearchTool(workspace_root=self._workspace_root, docs_roots=self._docs_roots)
                doc_hits = self._search_docs(tool, query=query, reason=reason, max_hits=self._config.max_doc_hits)
            except Exception:
                self._logger.debug("Self-correction docs search failed", exc_info=True)

        if self._config.web_search and ResearchTool is not None:
            try:
                tool = ResearchTool(workspace_root=self._workspace_root, docs_roots=self._docs_roots)
                hits = tool.search_web(query, max_results=3)
                web_hits = [hit.to_dict() for hit in hits]
            except Exception:
                self._logger.debug("Self-correction web search failed", exc_info=True)

        if ProblemAnalyzer is not None:
            try:
                analyzer = ProblemAnalyzer.from_env() if hasattr(ProblemAnalyzer, "from_env") else ProblemAnalyzer()  # type: ignore[call-arg]
                if analyzer is None:
                    analyzer = ProblemAnalyzer()  # type: ignore[call-arg]
                sub_questions = analyzer.analyze_problem(
                    query or "Investigate repeated failures",
                    context={"case": dict(case), "classification": classification, "capabilities": capabilities},
                    max_subquestions=int(max(1, self._config.max_subquestions)),
                )
            except Exception:
                sub_questions = []

        return {
            "time": float(now),
            "source": str(source),
            "trigger": trigger,
            "classification": classification,
            "capabilities": list(capabilities),
            "query": query,
            "reason": _compact(reason, 480),
            "action": _compact(action, 240),
            "evidence": evidence,
            "doc_hits": doc_hits,
            "web_hits": web_hits,
            "sub_questions": list(sub_questions or []),
            "case": dict(case),
        }

    def _search_docs(self, tool: Any, *, query: str, reason: str, max_hits: int) -> list[dict]:
        keywords = self._keywords_from_reason(reason) + self._keywords_from_query(query)
        keywords = _unique_preserve_order(keywords)[: max(1, int(max_hits))]
        hits: list[dict] = []
        seen: set[str] = set()
        for kw in keywords:
            for hit in tool.query_docs(kw, max_results=2):
                payload = hit.to_dict() if hasattr(hit, "to_dict") else {"path": getattr(hit, "path", ""), "snippet": getattr(hit, "snippet", "")}
                path = str(payload.get("path") or "")
                if not path or path in seen:
                    continue
                seen.add(path)
                hits.append(payload)
                if len(hits) >= int(max_hits):
                    return hits
        return hits

    def _keywords_from_reason(self, reason: str) -> list[str]:
        text = str(reason or "")
        keywords: list[str] = []
        match = _MISSING_MODULE_RE.search(text)
        if match:
            keywords.append(match.group("name"))
        keywords.extend(_EXC_RE.findall(text))
        lowered = text.lower()
        for token in ("permission", "timeout", "rate limit", "not found", "keyerror", "attributeerror", "typeerror"):
            if token in lowered:
                keywords.append(token)
        return keywords

    def _keywords_from_query(self, query: str) -> list[str]:
        query = str(query or "")
        if not query:
            return []
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", query)
        return [w for w in words if w.lower() not in {"error", "failed", "failure", "task", "trigger"}][:6]

    def _classify(self, *, trigger: str, reason: str, autofix: Any) -> str:
        reason_lower = str(reason or "").lower()
        trigger_lower = str(trigger or "").lower()
        if trigger_lower == "action_loop":
            return "strategy_action_loop"
        if trigger_lower in {"replan_storm", "bad_plan_burst"}:
            return "planning_quality"
        if "no module named" in reason_lower:
            return "missing_dependency"
        if "permission" in reason_lower or "permissionerror" in reason_lower:
            return "filesystem_permission"
        if "timeout" in reason_lower or "timed out" in reason_lower:
            return "timeout_or_latency"
        if "rate limit" in reason_lower:
            return "rate_limit"
        if isinstance(autofix, Mapping) and autofix.get("analysis"):
            return "autofix_available"
        return "generic_failure"

    def _capability_hints(self, *, trigger: str, reason: str, classification: str) -> Sequence[str]:
        if not self._config.add_capability_hints:
            return ()
        caps: list[str] = []
        reason_lower = str(reason or "").lower()
        trigger_lower = str(trigger or "").lower()

        if classification in {"planning_quality"} or trigger_lower in {"replan_storm", "bad_plan_burst"}:
            caps.extend(["planning", "reasoning"])
        if "permission" in reason_lower:
            caps.append("filesystem")
        if "no module named" in reason_lower:
            caps.extend(["python", "dependency_management"])
        if "timeout" in reason_lower or "rate limit" in reason_lower:
            caps.append("reliability")
        if any(tok in reason_lower for tok in ("typeerror", "attributeerror", "keyerror", "traceback")):
            caps.append("debugging")
        if trigger_lower == "action_loop":
            caps.append("strategy_selection")
        if self._config.docs_search or self._config.web_search:
            caps.append("research")
        return tuple(_unique_preserve_order(caps)[:6])

    # ------------------------------------------------------------------ publishing / persistence
    def _persist(self, remediation: Mapping[str, Any]) -> None:
        router = self._memory_router
        if router is None or not hasattr(router, "add_observation"):
            return
        try:
            summary = (
                f"self_correction classification={remediation.get('classification')} "
                f"trigger={remediation.get('trigger')} reason={remediation.get('reason')}"
            )
            router.add_observation(summary, source="self_correction", metadata=dict(remediation))
        except Exception:
            self._logger.debug("Failed to persist self-correction remediation", exc_info=True)

    def _publish(self, remediation: Mapping[str, Any]) -> None:
        try:
            self._bus.publish("diagnostics.self_correction", dict(remediation))
        except Exception:
            pass
        if self._performance_monitor is not None and hasattr(self._performance_monitor, "log_snapshot"):
            try:
                self._performance_monitor.log_snapshot({"self_correction_trigger": 1.0})
            except Exception:
                pass

    def _maybe_request_learning(self, remediation: Mapping[str, Any]) -> None:
        if not self._config.publish_learning_request:
            return
        try:
            self._bus.publish(
                "learning.request",
                {"time": time.time(), "reason": f"self_correction:{remediation.get('classification', 'unknown')}"},
            )
        except Exception:
            pass

    def _maybe_publish_plan(self, remediation: Mapping[str, Any]) -> None:
        if not self._config.publish_plan:
            return
        goal = (
            f"Self-correction ({remediation.get('classification')}): "
            f"resolve trigger={remediation.get('trigger')} reason={remediation.get('reason')}"
        )
        capabilities = remediation.get("capabilities")
        if isinstance(capabilities, list) and capabilities:
            caps_str = ",".join(str(c) for c in capabilities if str(c))
            if caps_str:
                goal = f"{goal} [capabilities:{caps_str}]"

        tasks: list[str] = [
            "Summarize failure evidence and identify the dominant root cause.",
            "Check whether a knowledge/skill gap exists; if yes, gather missing info and record it.",
            "Try an alternative strategy/module and define a small verification step.",
        ]
        sub_questions = remediation.get("sub_questions")
        if isinstance(sub_questions, list) and sub_questions:
            tasks.extend([f"Answer: {q}" for q in sub_questions[:3] if str(q).strip()])
        doc_hits = remediation.get("doc_hits")
        if isinstance(doc_hits, list) and doc_hits:
            top = [hit.get("path") for hit in doc_hits if isinstance(hit, Mapping) and hit.get("path")]
            if top:
                tasks.append(f"Review docs: {', '.join(str(p) for p in top[:2])}")

        try:
            self._bus.publish(
                "planner.plan_ready",
                {"goal": goal, "tasks": tasks, "source": "self_correction", "metadata": {"remediation": dict(remediation)}},
            )
        except Exception:
            pass

    def _parse_docs_roots_env(self) -> Sequence[Path] | None:
        raw = os.getenv("SELF_CORRECTION_DOCS_ROOTS")
        if not raw:
            return None
        roots: list[Path] = []
        for part in raw.split(os.pathsep):
            part = part.strip()
            if not part:
                continue
            try:
                roots.append(Path(part).resolve())
            except Exception:
                roots.append(Path(part))
        return tuple(roots)


__all__ = ["SelfCorrectionManager", "CorrectionConfig"]

