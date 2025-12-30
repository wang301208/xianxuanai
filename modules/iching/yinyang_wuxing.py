"""Yin-Yang transformations and Five Elements interactions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class YinYangFiveElements:
    """Utility class providing Yin-Yang conversion and Five Elements relations."""

    # Constants for Yin and Yang
    YIN: str = "yin"
    YANG: str = "yang"

    # Generation (sheng) cycle mapping
    GENERATION = {
        "wood": "fire",
        "fire": "earth",
        "earth": "metal",
        "metal": "water",
        "water": "wood",
    }

    # Control (ke) cycle mapping
    CONTROL = {
        "wood": "earth",
        "earth": "water",
        "water": "fire",
        "fire": "metal",
        "metal": "wood",
    }

    @staticmethod
    def transform_yinyang(value: str | int | bool):
        """Return the opposite Yin/Yang for the provided value.

        Parameters
        ----------
        value: str | int | bool
            ``"yin"``/``"yang"`` (case insensitive), ``0``/``1``, or ``True``/``False``.

        Returns
        -------
        Same type as ``value`` with Yin and Yang swapped.
        """

        if isinstance(value, str):
            normalized = value.lower()
            if normalized == YinYangFiveElements.YIN:
                return YinYangFiveElements.YANG
            if normalized == YinYangFiveElements.YANG:
                return YinYangFiveElements.YIN
        elif isinstance(value, bool):
            return not value
        elif isinstance(value, int):
            if value in (0, 1):
                return 1 - value

        raise ValueError("value must be 'yin', 'yang', 0/1, or boolean")

    @classmethod
    def element_interaction(cls, element_a: str, element_b: str) -> str:
        """Determine the interaction between two Five Elements.

        Parameters
        ----------
        element_a, element_b: str
            Names of the Five Elements (wood, fire, earth, metal, water).

        Returns
        -------
        str
            One of ``"generate"``, ``"generated_by"``, ``"control"``,
            ``"controlled_by"``, or ``"same"``.
        """

        a = element_a.lower()
        b = element_b.lower()
        valid = cls.GENERATION.keys()
        if a not in valid or b not in valid:
            raise ValueError("element must be one of wood, fire, earth, metal, water")
        if a == b:
            return "same"
        if cls.GENERATION[a] == b:
            return "generate"
        if cls.GENERATION[b] == a:
            return "generated_by"
        if cls.CONTROL[a] == b:
            return "control"
        if cls.CONTROL[b] == a:
            return "controlled_by"
        return "none"
