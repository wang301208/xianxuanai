from __future__ import annotations

"""User personalization utilities for I Ching interpretations."""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from .hexagram64 import Hexagram, HexagramEngine


@dataclass
class UserProfile:
    """Basic user profile storing preferences and interaction history."""

    user_id: str
    history: List[int] = field(default_factory=list)
    preferences: Dict[str, str] = field(default_factory=dict)
    traits: Dict[str, str] = field(default_factory=dict)

    def record(self, hexagram: Hexagram) -> None:
        """Record a consulted ``hexagram`` in the user's history."""

        self.history.append(hexagram.number)


class PersonalizedHexagramEngine:
    """Engine that generates personalised I Ching readings."""

    def __init__(self, engine: Optional[HexagramEngine] = None) -> None:
        self.engine = engine or HexagramEngine()

    def interpret(
        self,
        upper: str,
        lower: str,
        user: UserProfile,
        context: Optional[str] = None,
    ) -> Hexagram:
        """Return a personalised interpretation for ``user``.

        The underlying :class:`HexagramEngine` provides the base reading while
        this method injects the user's preferences and traits into the result.
        Each call also updates the user's history.
        """

        base = self.engine.interpret(upper, lower, context=context)
        previous = user.history[-1] if user.history else None
        user.record(base)

        focus = user.preferences.get("focus", "balance")
        trait = next(iter(user.traits.values()), None)

        judgement = (
            f"{base.judgement} Personalized for {user.user_id}, emphasizing {focus}."
        )
        if trait:
            judgement += f" Draw on your {trait}."
        if previous is not None:
            judgement += f" Previously you received hexagram {previous}."

        lines = tuple(f"{line} [{focus}]" for line in base.lines)
        return replace(base, judgement=judgement, lines=lines)
