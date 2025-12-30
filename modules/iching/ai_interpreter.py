from __future__ import annotations

"""AI enhanced interpreter for I Ching hexagrams.

This module provides :class:`AIEnhancedInterpreter` which augments the
traditional hexagram text with context aware advice.  In a real system this
class could interface with a large language model or an external knowledge
base.  For the purposes of this repository it uses a small, deterministic
knowledge base so that behaviour is consistent and testable.
"""

from dataclasses import replace
from typing import Dict


class AIEnhancedInterpreter:
    """Provide modern, context aware interpretations for hexagrams."""

    #: Default hints for a couple of common contexts.  These strings intentionally
    #: include the context name so that tests can assert on the resulting text
    #: without depending on model randomness.
    _DEFAULT_KNOWLEDGE_BASE: Dict[str, str] = {
        "business": "For business matters, focus on strategic leadership and clear goals.",
        "relationships": "For relationships, nurture open communication and trust.",
        "health": "For health, balance activity with adequate rest and reflection.",
    }

    def __init__(self, knowledge_base: Dict[str, str] | None = None) -> None:
        self.knowledge_base = knowledge_base or self._DEFAULT_KNOWLEDGE_BASE

    def enhance(self, hexagram, context: str):
        """Return a new :class:`Hexagram` with advice tailored to ``context``.

        Parameters
        ----------
        hexagram:
            The base hexagram interpretation.
        context:
            A description of the questioner's context, e.g. "business" or
            "relationships".
        """

        ctx = context.lower()
        advice = self.knowledge_base.get(
            ctx,
            f"For {ctx}, reflect on how the principles of {hexagram.name} apply.",
        )

        judgement = f"{hexagram.judgement} {advice}"
        lines = tuple(f"{line} ({ctx})" for line in hexagram.lines)
        # ``replace`` creates a new dataclass instance preserving the original
        # hexagram fields except for those we override.
        return replace(hexagram, judgement=judgement, lines=lines)
