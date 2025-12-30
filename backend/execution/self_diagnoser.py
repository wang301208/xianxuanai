from __future__ import annotations

"""LLM-driven failure diagnosis and remediation suggestions.

This module complements the existing heuristic self-debug/self-correction
managers by providing *semantic* analysis of failures using an LLM.

It is intentionally opt-in to avoid accidental network calls:
set `SELF_DIAGNOSER_ENABLED=1` and configure an API key (e.g. `OPENAI_API_KEY`).

Event inputs:
- `task_manager.task_completed` (failed tasks with optional `autofix` payload)
- `diagnostics.answer_mismatch` (question/answer/reference triples)

Event outputs (best-effort):
- `diagnostics.self_diagnosis` (structured diagnosis + recommended actions)
- `planner.plan_ready` (optional: convert actions into a concrete plan)
- `learning.request` (optional: request a background learning cycle)
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Sequence

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


def _is_truthy_env(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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


@dataclass(frozen=True)
class SelfDiagnoserConfig:
    enabled: bool = False
    cooldown_secs: float = 300.0
    publish_plan: bool = True
    publish_learning_request: bool = True
    temperature: float = 0.0
    max_context_chars: int = 8_000


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


class SelfDiagnoser:
    """LLM-based diagnoser that classifies failures and recommends actions."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        llm: LLMCallable | None = None,
        model: str | None = None,
        config: SelfDiagnoserConfig | None = None,
        memory_router: Any | None = None,
        performance_monitor: Any | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")
        self._bus = event_bus
        self._llm = llm
        self._model = model
        self._memory_router = memory_router
        self._performance_monitor = performance_monitor
        self._logger = logger_ or logger

        cfg = config or SelfDiagnoserConfig(
            enabled=_env_bool("SELF_DIAGNOSER_ENABLED", False),
            cooldown_secs=_env_float("SELF_DIAGNOSER_COOLDOWN_SECS", 300.0),
            publish_plan=_env_bool("SELF_DIAGNOSER_PUBLISH_PLAN", True),
            publish_learning_request=_env_bool("SELF_DIAGNOSER_REQUEST_LEARNING", True),
            temperature=float(os.getenv("SELF_DIAGNOSER_TEMPERATURE") or 0.0),
            max_context_chars=int(float(os.getenv("SELF_DIAGNOSER_MAX_CONTEXT_CHARS") or 8000)),
        )
        self._config = cfg
        self._last_ts: float | None = None

        self._subscriptions: list[Callable[[], None]] = [
            self._bus.subscribe("task_manager.task_completed", self._on_task_completed),
            self._bus.subscribe("diagnostics.answer_mismatch", self._on_answer_mismatch),
        ]

    @classmethod
    def from_env(
        cls,
        *,
        event_bus: EventBus,
        memory_router: Any | None = None,
        performance_monitor: Any | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> "SelfDiagnoser | None":
        if not _is_truthy_env("SELF_DIAGNOSER_ENABLED"):
            return None
        model = os.getenv("SELF_DIAGNOSER_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
        temperature = float(os.getenv("SELF_DIAGNOSER_TEMPERATURE") or 0.0)
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("AZURE_OPENAI_API_KEY"):
            logger.debug("SELF_DIAGNOSER_ENABLED is set but no API key found; disabled.")
            return None
        llm = _openai_chat_completion(model=model, temperature=temperature)
        cfg = SelfDiagnoserConfig(
            enabled=True,
            cooldown_secs=_env_float("SELF_DIAGNOSER_COOLDOWN_SECS", 300.0),
            publish_plan=_env_bool("SELF_DIAGNOSER_PUBLISH_PLAN", True),
            publish_learning_request=_env_bool("SELF_DIAGNOSER_REQUEST_LEARNING", True),
            temperature=temperature,
            max_context_chars=int(float(os.getenv("SELF_DIAGNOSER_MAX_CONTEXT_CHARS") or 8000)),
        )
        return cls(
            event_bus=event_bus,
            llm=llm,
            model=model,
            config=cfg,
            memory_router=memory_router,
            performance_monitor=performance_monitor,
            logger_=logger_,
        )

    def close(self) -> None:
        subs = list(self._subscriptions)
        self._subscriptions.clear()
        for cancel in subs:
            try:
                cancel()
            except Exception:
                continue

    # ------------------------------------------------------------------ event handlers
    async def _on_task_completed(self, event: Dict[str, Any]) -> None:
        if not self._config.enabled or not isinstance(event, Mapping):
            return
        status = str(event.get("status") or "").lower()
        if status in {"completed", "success"}:
            return
        now = float(event.get("time", time.time()) or time.time())
        if self._cooldown_active(now):
            return

        diagnosis = self.diagnose_exception_event(dict(event))
        self._publish(diagnosis)
        self._last_ts = now

    async def _on_answer_mismatch(self, event: Dict[str, Any]) -> None:
        if not self._config.enabled or not isinstance(event, Mapping):
            return
        now = float(event.get("time", time.time()) or time.time())
        if self._cooldown_active(now):
            return
        diagnosis = self.diagnose_answer_mismatch_event(dict(event))
        self._publish(diagnosis)
        self._last_ts = now

    # ------------------------------------------------------------------ diagnosis builders
    def diagnose_exception_event(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        now = float(event.get("time", time.time()) or time.time())
        payload = {
            "kind": "exception",
            "task_id": event.get("task_id"),
            "name": event.get("name"),
            "category": event.get("category"),
            "error": event.get("error"),
            "autofix": event.get("autofix"),
            "metadata": event.get("metadata") if isinstance(event.get("metadata"), Mapping) else {},
        }
        raw = ""
        data: Dict[str, Any] | None = None
        used_llm = False

        if self._llm is not None:
            prompt = self._build_exception_prompt(payload)
            raw, data = self._call_llm(prompt)
            used_llm = data is not None

        if data is None:
            # Fall back to existing autofix analysis/plan payload if present.
            data = self._fallback_from_autofix(payload.get("autofix"))

        return self._finalise(
            now=now,
            source="task_manager.task_completed",
            input_payload=payload,
            raw=raw,
            data=data or {},
            used_llm=used_llm,
        )

    def diagnose_answer_mismatch_event(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        now = float(event.get("time", time.time()) or time.time())
        payload = {
            "kind": "answer_mismatch",
            "question": event.get("question") or event.get("prompt") or event.get("input"),
            "answer": event.get("answer") or event.get("response") or event.get("output"),
            "reference": event.get("reference") or event.get("correct_answer") or event.get("expected"),
            "metadata": event.get("metadata") if isinstance(event.get("metadata"), Mapping) else {},
        }
        raw = ""
        data: Dict[str, Any] | None = None
        used_llm = False
        if self._llm is not None:
            prompt = self._build_answer_prompt(payload)
            raw, data = self._call_llm(prompt)
            used_llm = data is not None
        if data is None:
            data = {
                "category": "answer_incorrect",
                "confidence": 0.5,
                "summary": "Answer mismatch detected; manual review required.",
                "recommendations": [
                    "Verify assumptions and retrieve missing context before answering again.",
                    "If the task requires domain knowledge, update the knowledge base or request human feedback.",
                ],
                "actions": [
                    {"kind": "planner.replan", "instruction": "Re-answer with verified evidence and explicit assumptions."},
                    {"kind": "knowledge.update", "query": _compact_ws(payload.get("question") or "")},
                ],
            }
        return self._finalise(
            now=now,
            source="diagnostics.answer_mismatch",
            input_payload=payload,
            raw=raw,
            data=data or {},
            used_llm=used_llm,
        )

    # ------------------------------------------------------------------ internals
    def _cooldown_active(self, now: float) -> bool:
        if self._config.cooldown_secs <= 0:
            return False
        if self._last_ts is None:
            return False
        return (float(now) - float(self._last_ts)) < float(self._config.cooldown_secs)

    def _build_exception_prompt(self, payload: Mapping[str, Any]) -> str:
        max_chars = int(self._config.max_context_chars)
        autofix_blob = _truncate(
            json.dumps(payload.get("autofix") or {}, ensure_ascii=False, sort_keys=True, default=str),
            max_chars=max_chars,
        )
        meta_blob = _truncate(
            json.dumps(payload.get("metadata") or {}, ensure_ascii=False, sort_keys=True, default=str),
            max_chars=max_chars,
        )
        return (
            "You are a failure diagnosis module for an autonomous agent.\n"
            "Given a FAILED task event, classify the failure and propose concrete remediation actions.\n"
            "Return ONLY JSON (optionally inside a ```json``` fenced block). No prose.\n\n"
            "Schema:\n"
            "{\n"
            '  "category": str,                # e.g. "knowledge_gap","planning_error","code_bug","dependency","permission","timeout","rate_limit","tool_failure","unknown"\n'
            '  "confidence": 0..1,\n'
            '  "summary": str,\n'
            '  "recommendations": [str,...],\n'
            '  "actions": [\n'
            "    {\"kind\": \"knowledge.update\", \"query\": str},\n"
            "    {\"kind\": \"planner.replan\", \"instruction\": str},\n"
            "    {\"kind\": \"learning.request\", \"reason\": str},\n"
            "    {\"kind\": \"human.request\", \"question\": str}\n"
            "  ]\n"
            "}\n\n"
            f"Task name: {payload.get('name')}\n"
            f"Task category: {payload.get('category')}\n"
            f"Error (repr): {_truncate(payload.get('error'), max_chars=800)}\n"
            f"Task metadata (JSON): {meta_blob}\n"
            f"Autofix payload (JSON): {autofix_blob}\n"
        )

    def _build_answer_prompt(self, payload: Mapping[str, Any]) -> str:
        max_chars = int(self._config.max_context_chars)
        meta_blob = _truncate(
            json.dumps(payload.get("metadata") or {}, ensure_ascii=False, sort_keys=True, default=str),
            max_chars=max_chars,
        )
        return (
            "You are a failure diagnosis module for an autonomous agent.\n"
            "Given a question, the agent answer, and the reference answer (if provided),\n"
            "classify the error type and propose concrete remediation actions.\n"
            "Return ONLY JSON (optionally inside a ```json``` fenced block). No prose.\n\n"
            "Schema:\n"
            "{\n"
            '  "category": str,          # e.g. "knowledge_gap","reasoning_error","hallucination","format_error","answer_incorrect"\n'
            '  "confidence": 0..1,\n'
            '  "summary": str,\n'
            '  "recommendations": [str,...],\n'
            '  "actions": [\n'
            "    {\"kind\": \"knowledge.update\", \"query\": str},\n"
            "    {\"kind\": \"planner.replan\", \"instruction\": str},\n"
            "    {\"kind\": \"human.request\", \"question\": str}\n"
            "  ]\n"
            "}\n\n"
            f"Question: {_truncate(payload.get('question'), max_chars=1500)}\n"
            f"Agent answer: {_truncate(payload.get('answer'), max_chars=1500)}\n"
            f"Reference answer: {_truncate(payload.get('reference'), max_chars=1500)}\n"
            f"Metadata (JSON): {meta_blob}\n"
        )

    def _call_llm(self, prompt: str) -> tuple[str, Optional[Dict[str, Any]]]:
        try:
            raw = (self._llm(prompt) or "").strip() if self._llm is not None else ""
        except Exception:  # pragma: no cover - best effort
            self._logger.debug("SelfDiagnoser LLM call failed.", exc_info=True)
            return "", None
        data = _parse_json_object(raw)
        return raw, data

    def _fallback_from_autofix(self, autofix: Any) -> Dict[str, Any] | None:
        if not isinstance(autofix, Mapping):
            return None
        analysis = autofix.get("analysis") if isinstance(autofix.get("analysis"), Mapping) else {}
        fix_history = autofix.get("fix_history")
        fix = None
        if isinstance(fix_history, list) and fix_history:
            last = fix_history[-1]
            if isinstance(last, Mapping):
                data = last.get("data")
                if isinstance(data, Mapping):
                    fix = data.get("fix") if isinstance(data.get("fix"), Mapping) else None
        likely = str(analysis.get("message") or analysis.get("likely_root_cause") or "").strip()
        instructions = ""
        if isinstance(fix, Mapping):
            instructions = str(fix.get("instructions") or "").strip()
        summary = instructions or likely or "Task failed; autofix analysis available."
        return {
            "category": "autofix_available",
            "confidence": float(analysis.get("confidence", 0.55) or 0.55) if analysis else 0.55,
            "summary": summary,
            "recommendations": [r for r in [instructions, likely] if r],
            "actions": [
                {"kind": "planner.replan", "instruction": summary},
                {"kind": "learning.request", "reason": "autofix_feedback"},
            ],
        }

    def _finalise(
        self,
        *,
        now: float,
        source: str,
        input_payload: Mapping[str, Any],
        raw: str,
        data: Mapping[str, Any],
        used_llm: bool,
    ) -> Dict[str, Any]:
        category = str(data.get("category") or "unknown").strip() or "unknown"
        confidence = data.get("confidence")
        try:
            confidence_v = float(confidence) if confidence is not None else (0.7 if used_llm else 0.4)
        except Exception:
            confidence_v = 0.7 if used_llm else 0.4
        confidence_v = max(0.0, min(1.0, confidence_v))

        diagnosis = {
            "time": float(now),
            "source": str(source),
            "model": self._model,
            "used_llm": bool(used_llm),
            "category": category,
            "confidence": confidence_v,
            "summary": str(data.get("summary") or "").strip(),
            "recommendations": list(data.get("recommendations") or []),
            "actions": list(data.get("actions") or []),
            "input": dict(input_payload),
        }
        if raw:
            diagnosis["raw"] = raw
        return diagnosis

    def _publish(self, diagnosis: Mapping[str, Any]) -> None:
        try:
            self._bus.publish("diagnostics.self_diagnosis", dict(diagnosis))
        except Exception:
            pass

        self._persist(diagnosis)
        self._emit_metrics()
        self._apply_actions(diagnosis)

    def _persist(self, diagnosis: Mapping[str, Any]) -> None:
        router = self._memory_router
        if router is None or not hasattr(router, "add_observation"):
            return
        try:
            summary = (
                f"self_diagnosis category={diagnosis.get('category')} "
                f"confidence={diagnosis.get('confidence')} summary={diagnosis.get('summary')}"
            )
            router.add_observation(summary, source="self_diagnosis", metadata=dict(diagnosis))
        except Exception:
            self._logger.debug("Failed to persist self diagnosis", exc_info=True)

    def _emit_metrics(self) -> None:
        monitor = self._performance_monitor
        if monitor is None or not hasattr(monitor, "log_snapshot"):
            return
        try:
            monitor.log_snapshot({"self_diagnoser_trigger": 1.0})
        except Exception:
            pass

    def _apply_actions(self, diagnosis: Mapping[str, Any]) -> None:
        actions = diagnosis.get("actions")
        if not isinstance(actions, list):
            actions = []

        if self._config.publish_learning_request:
            for act in actions:
                if not isinstance(act, Mapping):
                    continue
                if str(act.get("kind") or "").strip().lower() == "learning.request":
                    reason = str(act.get("reason") or diagnosis.get("category") or "diagnosis")
                    try:
                        self._bus.publish("learning.request", {"time": time.time(), "reason": f"self_diagnosis:{reason}"})
                    except Exception:
                        pass
                    break

        if not self._config.publish_plan:
            return

        goal = (
            f"Self-diagnosis ({diagnosis.get('category')}): "
            f"{(diagnosis.get('summary') or 'Investigate failure and apply remediation')}"
        )
        tasks: list[str] = []
        recs = diagnosis.get("recommendations")
        if isinstance(recs, list):
            for rec in recs[:4]:
                if str(rec).strip():
                    tasks.append(f"Recommendation: {str(rec).strip()}")
        for act in actions[:6]:
            if not isinstance(act, Mapping):
                continue
            kind = str(act.get("kind") or "").strip()
            if kind == "knowledge.update":
                q = str(act.get("query") or "").strip()
                if q:
                    tasks.append(f"Update knowledge / search docs for: {q}")
            elif kind == "planner.replan":
                instr = str(act.get("instruction") or "").strip()
                if instr:
                    tasks.append(f"Adjust plan/reasoning: {instr}")
            elif kind == "human.request":
                q = str(act.get("question") or "").strip()
                if q:
                    tasks.append(f"Request human feedback: {q}")

        if not tasks:
            tasks = [
                "Summarize the failure evidence and identify the most likely root cause.",
                "Choose one remediation action and define a quick verification step.",
            ]

        try:
            self._bus.publish(
                "planner.plan_ready",
                {"goal": goal, "tasks": tasks, "source": "self_diagnoser", "metadata": {"diagnosis": dict(diagnosis)}},
            )
        except Exception:
            pass


__all__ = ["SelfDiagnoser", "SelfDiagnoserConfig"]

