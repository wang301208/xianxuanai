"""
Dialogue state tracking utilities.

The :class:`DialogueStateTracker` maintains a lightweight representation of the
current conversation including the last few comprehension results, referenced
entities, and inferred topics.  The tracker is intentionally simple so it can
run entirely on local data structures while still providing context for
cross-sentence reasoning, intent refinement, and follow-up action planning.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional


class DialogueStateTracker:
    """Maintain rolling dialogue context for the language hub."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.max_turns = int(cfg.get("max_turns", 6))
        self.max_entities = int(cfg.get("max_entities", 12))
        self.topics_window = int(cfg.get("topics_window", 5))
        self._turns: Deque[Dict[str, Any]] = deque(maxlen=self.max_turns)
        self._entities: Deque[str] = deque(maxlen=self.max_entities)
        self._topics: Deque[str] = deque(maxlen=self.topics_window)
        self._pending_actions: Deque[str] = deque(maxlen=self.max_turns)
        self._last_intent: Optional[str] = None

    # ------------------------------------------------------------------ #
    def update(
        self,
        comprehension: Dict[str, Any],
        *,
        speaker: str = "user",
    ) -> Dict[str, Any]:
        """Update state with the latest structured comprehension result."""

        if not isinstance(comprehension, dict):
            return self.snapshot()

        entry = {
            "speaker": speaker,
            "intent": comprehension.get("intent"),
            "key_terms": list(comprehension.get("key_terms", [])),
            "entities": list(comprehension.get("entities", [])),
            "summary": comprehension.get("summary"),
        }
        self._turns.append(entry)

        for entity in entry["entities"]:
            if entity:
                self._entities.append(entity)

        for topic in entry["key_terms"]:
            if topic:
                self._topics.append(topic)

        self._last_intent = entry["intent"] or self._last_intent

        pending_actions = comprehension.get("pending_actions") or []
        for action in pending_actions:
            if action:
                self._pending_actions.append(action)

        return self.snapshot()

    # ------------------------------------------------------------------ #
    def ingest_actions(self, actions: Iterable[str]) -> None:
        """Record planned actions so future turns can reference them."""

        for action in actions:
            if action:
                self._pending_actions.append(str(action))

    def snapshot(self) -> Dict[str, Any]:
        """Return an immutable snapshot of the current dialogue state."""

        return {
            "turns": [dict(turn) for turn in self._turns],
            "entities": list(self._entities),
            "topics": list(self._topics),
            "pending_actions": list(self._pending_actions),
            "last_intent": self._last_intent,
        }


__all__ = ["DialogueStateTracker"]
