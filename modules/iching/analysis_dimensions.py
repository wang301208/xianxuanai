"""Analysis dimensions and associated rules for hexagram interpretations."""

from dataclasses import dataclass
from typing import Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .hexagram64 import Hexagram


@dataclass(frozen=True)
class DimensionRule:
    """Container for dimension-specific interpretation and advice rules."""

    interpret: Callable[["Hexagram"], str]
    advise: Callable[["Hexagram"], str]


def _basic_judgement(prefix: str) -> Callable[["Hexagram"], str]:
    """Return a function using the hexagram's judgement with a prefix."""

    def _fn(hexagram: "Hexagram") -> str:
        return f"{prefix} {hexagram.judgement}"

    return _fn


ANALYSIS_DIMENSIONS: Dict[str, DimensionRule] = {
    "career": DimensionRule(
        interpret=_basic_judgement("Career outlook:"),
        advise=lambda h: "Pursue long-term goals with persistence.",
    ),
    "relationship": DimensionRule(
        interpret=_basic_judgement("Relationship dynamics:"),
        advise=lambda h: "Cultivate communication and mutual respect.",
    ),
    "health": DimensionRule(
        interpret=_basic_judgement("Health focus:"),
        advise=lambda h: "Maintain balance and attend to wellbeing.",
    ),
}
