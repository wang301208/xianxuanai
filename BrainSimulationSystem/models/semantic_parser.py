"""
Fallback semantic parsing utilities.

The :class:`SemanticFallbackParser` optionally consults an external LLM when the
internal semantic analyser produces low confidence results.  When an LLM is not
available the parser falls back to deterministic heuristics so the integration
remains self-contained.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence


class SemanticFallbackParser:
    """LLM-assisted semantic parser with graceful heuristic fallback."""

    def __init__(
        self,
        llm_service: Optional[Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm = llm_service
        self.config = config or {}
        self.prompt = self.config.get(
            "prompt",
            (
                "You are a semantic parser. Given user text and optional context "
                "return JSON with keys summary, key_terms, relations, intent_hint, "
                "and confidence. Relations should be an array of objects with "
                "head/dependent/relation fields."
            ),
        )
        self._logger = logging.getLogger(self.__class__.__name__)
        disallowed = self.config.get("disallow_providers", ["internal"])
        self._blocked_providers = {provider.lower() for provider in disallowed}

    # ------------------------------------------------------------------ #
    def parse(
        self,
        text: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        existing: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return enriched semantic hints for ``text``."""

        text = (text or "").strip()
        if not text:
            return None

        context_payload = context or {}

        if self._llm_allowed():
            parsed = self._query_llm(text, context_payload)
            if parsed:
                parsed.setdefault("source", "llm")
                return parsed

        heuristic = self._heuristic_semantics(text, context_payload, existing)
        heuristic.setdefault("source", "heuristic")
        return heuristic

    # ------------------------------------------------------------------ #
    def _llm_allowed(self) -> bool:
        if self.llm is None:
            return False
        provider = getattr(self.llm, "provider", None)
        if not provider:
            return True
        return provider.lower() not in self._blocked_providers

    def _query_llm(self, text: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "text": text,
                        "context": context,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            response = self.llm.chat(messages, response_format="json_object")
        except Exception as exc:  # pragma: no cover - network/LLM failure
            self._logger.debug("LLM semantic parse failed: %s", exc)
            return None

        if not response or not getattr(response, "text", None):
            return None
        return self._safe_json(response.text)

    @staticmethod
    def _safe_json(payload: str) -> Optional[Dict[str, Any]]:
        payload = (payload or "").strip()
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            start = payload.find("{")
            end = payload.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(payload[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    # ------------------------------------------------------------------ #
    def _heuristic_semantics(
        self,
        text: str,
        context: Dict[str, Any],
        existing: Optional[Any],
    ) -> Dict[str, Any]:
        """Fallback deterministic semantic extraction."""

        tokens = [token.strip(".,!?") for token in text.split()]
        tokens = [token for token in tokens if token]
        key_terms = self._top_terms(tokens, limit=6)

        relations: List[Dict[str, str]] = []
        if existing is not None:
            relations.extend(getattr(existing, "relations", []) or [])

        if not relations and len(tokens) >= 2:
            relations.append(
                {"head": tokens[0].lower(), "dependent": tokens[1].lower(), "relation": "mention"}
            )

        entities = list({token for token in tokens if token[:1].isupper()})
        summary = " ".join(text.split()[:25])
        intent_hint = context.get("dialogue_state", {}).get("last_intent")

        confidence = 0.45
        if existing is not None:
            confidence = float(
                min(
                    0.85,
                    0.4 + getattr(existing, "confidence", 0.2) * 0.6 + 0.05 * len(key_terms),
                )
            )

        return {
            "summary": summary,
            "key_terms": key_terms,
            "relations": relations,
            "entities": entities,
            "intent_hint": intent_hint,
            "confidence": confidence,
        }

    @staticmethod
    def _top_terms(tokens: Sequence[str], limit: int) -> List[str]:
        seen: List[str] = []
        for token in tokens:
            lower = token.lower()
            if len(lower) <= 2:
                continue
            if lower not in seen:
                seen.append(lower)
            if len(seen) >= limit:
                break
        return seen


__all__ = ["SemanticFallbackParser"]

