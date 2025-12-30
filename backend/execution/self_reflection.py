from __future__ import annotations

"""Post-task self-reflection loop.

This module implements a lightweight "reflect -> store -> retrieve" loop:
- when a task completes, summarize lessons learned and store them as a
  reflection note in short-term memory (MemoryRouter).
- when a new plan is created, retrieve relevant past reflections and publish
  them as context hints.

LLM usage is optional and controlled via env vars to avoid accidental network
calls in default deployments.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

try:  # optional in some deployments
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EventBus = None  # type: ignore

try:
    from modules.diagnostics.auto_fixer import extract_json_object  # type: ignore
except Exception:  # pragma: no cover - fallback parsing
    extract_json_object = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

LLMCallable = Callable[[str], str]

_CATEGORY = "autobiographical_reflection"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if number != number or number in (float("inf"), float("-inf")):
        return float(default)
    return float(number)


def _truncate(value: Any, *, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if extract_json_object is not None:
        try:
            return extract_json_object(text)
        except Exception:
            return None
    candidate = (text or "").strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _openai_chat_completion(*, model: str, temperature: float) -> LLMCallable:
    def _call(prompt: str) -> str:
        from openai import OpenAI  # local import: optional dependency / lazy loading

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You output strict JSON only. No prose. Optional ```json``` fencing is allowed.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content or ""

    return _call


@dataclass(frozen=True)
class SelfReflectionConfig:
    enabled: bool = False
    cooldown_secs: float = 60.0
    min_event_chars: int = 80
    max_context_chars: int = 8_000
    retrieve_top_k: int = 4
    retrieve_min_similarity: float = 0.25
    request_learning: bool = False
    temperature: float = 0.0


class SelfReflectionLoop:
    """Generate and persist a post-task reflection note."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        memory_router: Any,
        llm: LLMCallable | None = None,
        config: SelfReflectionConfig | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")
        self._bus = event_bus
        self._memory_router = memory_router
        self._llm = llm
        self._logger = logger_ or logger
        cfg = config or SelfReflectionConfig(
            enabled=_env_bool("SELF_REFLECTION_ENABLED", False),
            cooldown_secs=_env_float("SELF_REFLECTION_COOLDOWN_SECS", 60.0),
            min_event_chars=_env_int("SELF_REFLECTION_MIN_EVENT_CHARS", 80),
            max_context_chars=_env_int("SELF_REFLECTION_MAX_CONTEXT_CHARS", 8000),
            retrieve_top_k=_env_int("SELF_REFLECTION_RETRIEVE_TOP_K", 4),
            retrieve_min_similarity=_env_float("SELF_REFLECTION_MIN_SIMILARITY", 0.25),
            request_learning=_env_bool("SELF_REFLECTION_REQUEST_LEARNING", False),
            temperature=_env_float("SELF_REFLECTION_TEMPERATURE", 0.0),
        )
        self._config = cfg
        self._last_ts: float | None = None

    @classmethod
    def from_env(
        cls,
        *,
        event_bus: EventBus,
        memory_router: Any,
        logger_: Optional[logging.Logger] = None,
    ) -> "SelfReflectionLoop | None":
        if not _env_bool("SELF_REFLECTION_ENABLED", False):
            return None
        llm: LLMCallable | None = None
        model = os.getenv("SELF_REFLECTION_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        temperature = _env_float("SELF_REFLECTION_TEMPERATURE", 0.0)
        if os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"):
            try:
                llm = _openai_chat_completion(model=model, temperature=temperature)
            except Exception:
                llm = None
        return cls(
            event_bus=event_bus,
            memory_router=memory_router,
            llm=llm,
            logger_=logger_,
        )

    # ------------------------------------------------------------------ public API
    def reflect_task(
        self,
        event: Mapping[str, Any],
        *,
        trace: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate a reflection record and persist it into MemoryRouter."""

        if not self._config.enabled:
            return None
        if not isinstance(event, Mapping):
            return None

        now = float(event.get("time", time.time()) or time.time())
        if self._cooldown_active(now):
            return None

        status = str(event.get("status") or "").strip().lower()
        task_id = str(event.get("task_id") or event.get("task") or "").strip()
        agent_id = str(event.get("agent_id") or event.get("agent") or "").strip()
        summary = str(event.get("summary") or "").strip()
        detail = str(event.get("detail") or "").strip()

        raw_context = "\n".join(part for part in (summary, detail) if part)
        if len(raw_context) < self._config.min_event_chars:
            return None

        trace_payload = self._minify_trace(trace or [])
        prompt = self._build_prompt(
            task_id=task_id,
            agent_id=agent_id,
            status=status or "unknown",
            summary=summary,
            detail=detail,
            trace=trace_payload,
        )

        data: Dict[str, Any] | None = None
        used_llm = False
        raw = ""
        if self._llm is not None:
            try:
                raw = (self._llm(prompt) or "").strip()
                data = _parse_json_object(raw)
                used_llm = data is not None
            except Exception:
                data = None
                used_llm = False

        if data is None:
            data = self._fallback_reflection(status=status, summary=summary, detail=detail)

        record = {
            "time": now,
            "task_id": task_id or None,
            "agent_id": agent_id or None,
            "status": status or "unknown",
            "reflection": _compact_ws(data.get("reflection") or data.get("text") or ""),
            "lessons": [str(x).strip() for x in (data.get("lessons") or []) if str(x).strip()],
            "checklist": [str(x).strip() for x in (data.get("checklist") or []) if str(x).strip()],
            "tags": [str(x).strip() for x in (data.get("tags") or []) if str(x).strip()],
            "confidence": _safe_float(data.get("confidence"), default=0.0),
            "used_llm": used_llm,
            "source": "self_reflection",
        }

        note_text = self._render_note(record, summary=summary, detail=detail)
        memory_id = None
        try:
            if self._memory_router is not None and hasattr(self._memory_router, "add_observation"):
                source = f"reflection:{task_id}" if task_id else "reflection"
                memory_id = self._memory_router.add_observation(
                    note_text,
                    source=source,
                    metadata={
                        "category": _CATEGORY,
                        "task_id": task_id,
                        "agent_id": agent_id,
                        "status": status,
                        "tags": record["tags"],
                        "time": now,
                        "used_llm": used_llm,
                    },
                )
        except Exception:
            memory_id = None
        record["memory_id"] = memory_id

        try:
            self._bus.publish(
                "diagnostics.self_reflection",
                {
                    "time": now,
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "status": status,
                    "record": record,
                },
            )
        except Exception:
            pass

        if self._config.request_learning:
            try:
                self._bus.publish("learning.request", {"reason": "self_reflection", "time": now})
            except Exception:
                pass

        self._last_ts = now
        return record

    def retrieve_hints(self, query_text: str) -> List[Dict[str, Any]]:
        """Retrieve relevant past reflection notes for *query_text*."""

        if not self._config.enabled:
            return []
        query = str(query_text or "").strip()
        if not query:
            return []
        router = self._memory_router
        if router is None or not hasattr(router, "query"):
            return []

        top_k = max(1, int(self._config.retrieve_top_k))
        try:
            results = router.query(query, top_k=max(top_k, 8))
        except Exception:
            return []

        hints: List[Dict[str, Any]] = []
        min_sim = float(self._config.retrieve_min_similarity)
        for hit in results:
            if not isinstance(hit, Mapping):
                continue
            meta = hit.get("metadata")
            if not isinstance(meta, Mapping) or meta.get("category") != _CATEGORY:
                continue
            sim = _safe_float(hit.get("similarity"), default=0.0)
            if sim < min_sim:
                continue
            text = str(hit.get("text") or "").strip()
            if not text:
                continue
            hints.append(
                {
                    "text": _truncate(text, max_chars=1500),
                    "similarity": sim,
                    "source": hit.get("source"),
                    "metadata": dict(meta),
                }
            )
            if len(hints) >= top_k:
                break
        return hints

    # ------------------------------------------------------------------ internals
    def _cooldown_active(self, now: float) -> bool:
        if self._last_ts is None:
            return False
        return (now - float(self._last_ts)) < float(self._config.cooldown_secs)

    def _minify_trace(self, trace: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        if not trace:
            return []
        slim: List[Dict[str, Any]] = []
        for item in list(trace)[-10:]:
            if not isinstance(item, Mapping):
                continue
            slim.append(
                {
                    "type": item.get("type"),
                    "timestamp": item.get("timestamp"),
                    "goal": item.get("goal"),
                    "warning": item.get("warning"),
                    "reason": item.get("reason"),
                    "action": item.get("action") or item.get("command"),
                    "success": item.get("success"),
                }
            )
        return slim

    def _build_prompt(
        self,
        *,
        task_id: str,
        agent_id: str,
        status: str,
        summary: str,
        detail: str,
        trace: Sequence[Mapping[str, Any]],
    ) -> str:
        payload = {
            "task_id": task_id,
            "agent_id": agent_id,
            "status": status,
            "summary": summary,
            "detail": detail,
            "recent_trace": list(trace),
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        blob = _truncate(blob, max_chars=int(self._config.max_context_chars))
        return (
            "你是智能体的“自我反思模块”。请回顾刚才的任务执行过程，输出严格 JSON。\n"
            "目标：总结做得好的地方、失败原因/风险点、下次可复用的改进策略。\n\n"
            "输出 JSON 格式：\n"
            "{\n"
            '  "reflection": "一句话总评",\n'
            '  "lessons": ["经验教训1", "经验教训2"],\n'
            '  "checklist": ["下次执行前检查A", "执行中检查B"],\n'
            '  "tags": ["domain_or_error_type"],\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            "输入：\n"
            f"{blob}"
        )

    def _fallback_reflection(self, *, status: str, summary: str, detail: str) -> Dict[str, Any]:
        status = (status or "").lower()
        failed = status in {"failed", "error", "exception"}
        base_lessons = [
            "在行动前先明确成功标准与可验证的检查点。",
            "不确定时先检索/查询已有知识与历史经验，避免盲目推进。",
        ]
        base_checklist = [
            "先复述目标与约束，确认理解无误。",
            "对关键输入/输出做有效性检查（空值、格式、范围）。",
            "遇到异常先缩小复现范围，定位最小失败案例。",
        ]
        if failed:
            base_lessons.insert(0, "失败后先做归因：是知识缺口、工具使用错误，还是实现/环境问题。")
            base_checklist.insert(0, "先记录错误日志与上下文，再尝试修复。")
            reflection = "本次任务未达成预期，需要补齐知识/验证链路并改进执行步骤。"
            tags = ["failure", "debug"]
        else:
            reflection = "本次任务总体完成，但仍可通过更系统的检查与复盘提升稳定性。"
            tags = ["success", "process"]
        return {
            "reflection": reflection,
            "lessons": base_lessons,
            "checklist": base_checklist,
            "tags": tags,
            "confidence": 0.0,
            "summary": summary,
            "detail": detail,
        }

    def _render_note(self, record: Mapping[str, Any], *, summary: str, detail: str) -> str:
        lessons = record.get("lessons") if isinstance(record.get("lessons"), list) else []
        checklist = record.get("checklist") if isinstance(record.get("checklist"), list) else []
        tags = record.get("tags") if isinstance(record.get("tags"), list) else []

        lines: List[str] = []
        lines.append("【自我反思】")
        if summary:
            lines.append(f"任务摘要：{_compact_ws(summary)}")
        if detail:
            lines.append(f"任务详情：{_truncate(_compact_ws(detail), max_chars=800)}")
        lines.append(f"状态：{record.get('status')}")
        if record.get("reflection"):
            lines.append(f"总评：{record.get('reflection')}")
        if lessons:
            lines.append("经验教训：")
            for item in lessons[:6]:
                lines.append(f"- {item}")
        if checklist:
            lines.append("下次检查清单：")
            for item in checklist[:8]:
                lines.append(f"- {item}")
        if tags:
            lines.append("标签：" + ", ".join(str(t) for t in tags[:8]))
        return "\n".join(lines)


__all__ = ["SelfReflectionLoop", "SelfReflectionConfig"]

