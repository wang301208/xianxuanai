from __future__ import annotations

"""Unknown-problem analysis and research-oriented solving.

This module provides a small `ProblemAnalyzer` component that can:

1) decompose a high-level goal into concrete sub-questions
2) gather evidence for each sub-question (via `modules.knowledge.research_tool`)
3) ask an LLM to synthesize an actionable solution plan

The implementation is intentionally opt-in for network-backed LLM calls:
set `PROBLEM_ANALYZER_ENABLED=1` and provide an API key (e.g. `OPENAI_API_KEY`).
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .research_tool import ResearchTool

logger = logging.getLogger(__name__)

LLMCallable = Callable[[str], str]

_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)?\s*\n(?P<body>[\s\S]*?)\n```", re.MULTILINE)


@dataclass(frozen=True)
class ProblemBreakdown:
    goal: str
    sub_questions: List[str]
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "sub_questions": list(self.sub_questions), "created_at": self.created_at}


def _is_truthy_env(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _truncate(text: Any, *, max_chars: int) -> str:
    s = str(text or "")
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def _extract_fenced(text: str, *, lang: str) -> Optional[str]:
    target = (lang or "").strip().lower()
    last: Optional[str] = None
    for match in _FENCE_RE.finditer(text or ""):
        fence_lang = (match.group("lang") or "").strip().lower()
        if fence_lang == target:
            last = (match.group("body") or "").strip()
    return last


def _normalise_questions(items: Iterable[Any], *, max_items: int) -> List[str]:
    out: List[str] = []
    for item in items:
        if len(out) >= max_items:
            break
        text = str(item or "").strip()
        if not text:
            continue
        out.append(text)
    # de-dup while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for q in out:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    return deduped[:max_items]


def _parse_questions_from_text(text: str, *, max_items: int) -> List[str]:
    # 1) try JSON list in a fenced block
    fenced = _extract_fenced(text, lang="json")
    if fenced:
        try:
            parsed = json.loads(fenced)
            if isinstance(parsed, list):
                return _normalise_questions(parsed, max_items=max_items)
            if isinstance(parsed, dict):
                for key in ("sub_questions", "questions", "subtasks", "items"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        return _normalise_questions(value, max_items=max_items)
        except Exception:
            pass

    # 2) try any JSON list embedded
    blob = (text or "").strip()
    start = blob.find("[")
    end = blob.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(blob[start : end + 1])
            if isinstance(parsed, list):
                return _normalise_questions(parsed, max_items=max_items)
        except Exception:
            pass

    # 3) fall back to bullet parsing
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)]|[a-zA-Z][.)])\s+", "", line)
        line = line.strip()
        if line and len(line) >= 4:
            lines.append(line)
    return _normalise_questions(lines, max_items=max_items)


def _default_breakdown(goal: str, *, max_items: int) -> List[str]:
    goal = str(goal or "").strip()
    base = [
        f"Clarify the goal and success criteria for: {goal}",
        f"What constraints, inputs, and outputs exist for: {goal}?",
        f"What are the most common approaches/tools to solve: {goal}?",
        f"What are likely failure modes or edge cases for: {goal}?",
        f"What minimal prototype can validate the approach for: {goal}?",
    ]
    return base[: max(1, int(max_items))]


class ProblemAnalyzer:
    """Decompose unknown problems and orchestrate research-backed solving."""

    def __init__(
        self,
        *,
        llm: Optional[LLMCallable] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_context_chars: int = 8_000,
    ) -> None:
        self._llm = llm
        self._model = model
        self._temperature = float(temperature)
        self._max_context_chars = int(max_context_chars)

    @classmethod
    def from_env(cls) -> "ProblemAnalyzer | None":
        if not _is_truthy_env("PROBLEM_ANALYZER_ENABLED"):
            return None
        model = os.getenv("PROBLEM_ANALYZER_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
        temperature = float(os.getenv("PROBLEM_ANALYZER_TEMPERATURE") or 0.2)
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("AZURE_OPENAI_API_KEY"):
            logger.debug("PROBLEM_ANALYZER_ENABLED is set but no API key found; disabled.")
            return None
        return cls(llm=_openai_chat_completion(model=model, temperature=temperature), model=model, temperature=temperature)

    # ------------------------------------------------------------------ core API
    def analyze_problem(
        self,
        goal: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        max_subquestions: int = 5,
    ) -> List[str]:
        goal = str(goal or "").strip()
        max_subquestions = max(1, int(max_subquestions))
        if not goal:
            return []

        if self._llm is None:
            return _default_breakdown(goal, max_items=max_subquestions)

        ctx_blob = _truncate(
            json.dumps(context or {}, ensure_ascii=False, sort_keys=True, default=str),
            max_chars=self._max_context_chars,
        )
        prompt = (
            "You help an autonomous agent solve an UNKNOWN problem by decomposing it.\n"
            "Return a JSON list of short, actionable sub-questions.\n"
            "Rules:\n"
            f"- Return 1..{max_subquestions} items.\n"
            "- Output ONLY JSON (optionally inside a ```json``` fenced block). No prose.\n\n"
            f"Goal: {goal}\n"
            f"Context (JSON): {ctx_blob}\n"
        )
        try:
            raw = (self._llm(prompt) or "").strip()
        except Exception:  # pragma: no cover - best effort
            logger.debug("ProblemAnalyzer LLM call failed.", exc_info=True)
            return _default_breakdown(goal, max_items=max_subquestions)

        questions = _parse_questions_from_text(raw, max_items=max_subquestions)
        if not questions:
            return _default_breakdown(goal, max_items=max_subquestions)
        return questions

    def research(
        self,
        sub_questions: Sequence[str],
        *,
        research_tool: ResearchTool,
        search_web: bool = True,
        query_docs: bool = True,
        max_results_per_query: int = 3,
        docs_roots: Sequence[str | os.PathLike[str]] | None = None,
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for q in list(sub_questions or []):
            question = str(q or "").strip()
            if not question:
                continue
            entry: Dict[str, Any] = {"question": question}
            if search_web:
                hits = research_tool.search_web(question, max_results=max_results_per_query)
                entry["web_hits"] = [hit.to_dict() for hit in hits]
                entry["web_summary"] = research_tool.render_web_hits(hits)
            if query_docs:
                hits = research_tool.query_docs(
                    question,
                    roots=docs_roots,
                    max_results=max_results_per_query,
                )
                entry["doc_hits"] = [hit.to_dict() for hit in hits]
                entry["doc_summary"] = research_tool.render_doc_hits(hits)
            evidence.append(entry)
        return evidence

    def synthesize_solution(
        self,
        goal: str,
        *,
        breakdown: ProblemBreakdown,
        evidence: Sequence[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if self._llm is None:
            return None

        goal = str(goal or "").strip()
        ctx_blob = _truncate(
            json.dumps(context or {}, ensure_ascii=False, sort_keys=True, default=str),
            max_chars=self._max_context_chars,
        )
        evidence_blob = _truncate(
            json.dumps(list(evidence or []), ensure_ascii=False, sort_keys=True, default=str),
            max_chars=self._max_context_chars,
        )
        prompt = (
            "You help an autonomous agent solve a problem using gathered evidence.\n"
            "Return a compact, executable plan.\n"
            "Output requirements:\n"
            "- Return ONLY JSON (optionally inside ```json```), no prose.\n"
            "- Schema:\n"
            "  {\n"
            '    \"summary\": str,\n'
            '    \"steps\": [str],\n'
            '    \"risks\": [str],\n'
            '    \"open_questions\": [str]\n'
            "  }\n\n"
            f"Goal: {goal}\n"
            f"Sub-questions: {json.dumps(breakdown.sub_questions, ensure_ascii=False)}\n"
            f"Evidence (JSON): {evidence_blob}\n"
            f"Context (JSON): {ctx_blob}\n"
        )
        try:
            raw = (self._llm(prompt) or "").strip()
        except Exception:  # pragma: no cover - best effort
            logger.debug("ProblemAnalyzer synthesis LLM call failed.", exc_info=True)
            return None

        fenced = _extract_fenced(raw, lang="json") or raw
        return fenced.strip()

    def analyze_and_solve(
        self,
        goal: str,
        *,
        research_tool: Optional[ResearchTool] = None,
        context: Optional[Dict[str, Any]] = None,
        max_subquestions: int = 5,
        search_web: bool = True,
        query_docs: bool = True,
        max_results_per_query: int = 3,
        docs_roots: Sequence[str | os.PathLike[str]] | None = None,
    ) -> Dict[str, Any]:
        tool = research_tool or ResearchTool()
        sub_questions = self.analyze_problem(goal, context=context, max_subquestions=max_subquestions)
        breakdown = ProblemBreakdown(goal=str(goal or "").strip(), sub_questions=sub_questions, created_at=time.time())
        evidence = self.research(
            sub_questions,
            research_tool=tool,
            search_web=search_web,
            query_docs=query_docs,
            max_results_per_query=max_results_per_query,
            docs_roots=docs_roots,
        )
        plan = self.synthesize_solution(goal, breakdown=breakdown, evidence=evidence, context=context)
        return {
            "goal": breakdown.goal,
            "sub_questions": list(sub_questions),
            "evidence": list(evidence),
            "plan": plan,
            "created_at": breakdown.created_at,
        }


def _openai_chat_completion(*, model: str, temperature: float) -> LLMCallable:
    def _call(prompt: str) -> str:
        from openai import OpenAI  # local import: optional dependency / lazy loading

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You output strict JSON only. No markdown besides optional ```json``` fencing.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content or ""

    return _call


__all__ = ["ProblemAnalyzer", "ProblemBreakdown"]

