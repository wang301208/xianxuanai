"""Definitions of the eight trigrams (Bagua)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


@dataclass(frozen=True)
class Trigram:
    """Represents a single trigram with its metadata."""

    name: str
    chinese: str
    symbol: str
    direction: str


class PreHeavenBagua(Enum):
    """Fu Xi's (Earlier Heaven) arrangement of the eight trigrams."""

    QIAN = Trigram("Qian", "乾", "☰", "South")
    DUI = Trigram("Dui", "兑", "☱", "Southeast")
    LI = Trigram("Li", "离", "☲", "East")
    ZHEN = Trigram("Zhen", "震", "☳", "Northeast")
    XUN = Trigram("Xun", "巽", "☴", "Southwest")
    KAN = Trigram("Kan", "坎", "☵", "West")
    GEN = Trigram("Gen", "艮", "☶", "Northwest")
    KUN = Trigram("Kun", "坤", "☷", "North")


class PostHeavenBagua(Enum):
    """King Wen's (Later Heaven) arrangement of the eight trigrams."""

    QIAN = Trigram("Qian", "乾", "☰", "Northwest")
    DUI = Trigram("Dui", "兑", "☱", "West")
    LI = Trigram("Li", "离", "☲", "South")
    ZHEN = Trigram("Zhen", "震", "☳", "East")
    XUN = Trigram("Xun", "巽", "☴", "Southeast")
    KAN = Trigram("Kan", "坎", "☵", "North")
    GEN = Trigram("Gen", "艮", "☶", "Northeast")
    KUN = Trigram("Kun", "坤", "☷", "Southwest")


def _iterate_bagua(order: str) -> Iterable[Trigram]:
    order = order.lower()
    if order not in {"pre", "post"}:
        raise ValueError("order must be 'pre' or 'post'")
    bagua_cls = PreHeavenBagua if order == "pre" else PostHeavenBagua
    for trigram in bagua_cls:
        yield trigram.value


def get_trigram(order: str, name: str) -> Trigram:
    """Return the trigram by order ('pre' or 'post') and its name.

    Parameters
    ----------
    order:
        Either ``"pre"`` for the Earlier Heaven sequence or ``"post"`` for the Later
        Heaven sequence.
    name:
        English (e.g., ``"Qian"``) or Chinese (e.g., ``"乾"``) name of the trigram.
    """

    normalized = name.lower()
    for trigram in _iterate_bagua(order):
        if normalized == trigram.name.lower() or name == trigram.chinese:
            return trigram
    raise KeyError(f"Trigram '{name}' not found in {order}-heaven bagua")

