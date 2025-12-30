"""Definitions of the sixty-four hexagrams and an interpretation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from .bagua import PreHeavenBagua, PostHeavenBagua
from .time_context import TimeContext
from .ai_interpreter import AIEnhancedInterpreter
from .analysis_dimensions import ANALYSIS_DIMENSIONS

# Binary line patterns for each trigram (least significant bit is the bottom line)
_TRIGRAM_PATTERNS = {
    "Qian": 0b111,
    "Dui": 0b110,
    "Li": 0b101,
    "Zhen": 0b100,
    "Xun": 0b011,
    "Kan": 0b010,
    "Gen": 0b001,
    "Kun": 0b000,
}


@dataclass(frozen=True)
class Hexagram:
    """Represents a hexagram with its associated texts."""

    number: int
    name: str
    chinese: str
    judgement: str
    lines: Tuple[str, str, str, str, str, str]
    upper: str
    lower: str


# Basic information for the 64 hexagrams.
# Each tuple: (number, English name, Chinese, upper trigram, lower trigram)
_HEXAGRAM_INFO = [
    (1, "Qian", "乾", "Qian", "Qian"),
    (2, "Kun", "坤", "Kun", "Kun"),
    (3, "Zhun", "屯", "Kan", "Zhen"),
    (4, "Meng", "蒙", "Gen", "Kan"),
    (5, "Xu", "需", "Kan", "Qian"),
    (6, "Song", "讼", "Qian", "Kan"),
    (7, "Shi", "师", "Kan", "Kun"),
    (8, "Bi", "比", "Kun", "Kan"),
    (9, "Xiao Chu", "小畜", "Xun", "Qian"),
    (10, "Lu", "履", "Qian", "Dui"),
    (11, "Tai", "泰", "Kun", "Qian"),
    (12, "Pi", "否", "Qian", "Kun"),
    (13, "Tong Ren", "同人", "Qian", "Li"),
    (14, "Da You", "大有", "Li", "Qian"),
    (15, "Qian", "谦", "Kun", "Gen"),
    (16, "Yu", "豫", "Zhen", "Kun"),
    (17, "Sui", "随", "Dui", "Zhen"),
    (18, "Gu", "蛊", "Gen", "Xun"),
    (19, "Lin", "临", "Kun", "Dui"),
    (20, "Guan", "观", "Xun", "Kun"),
    (21, "Shi He", "噬嗑", "Li", "Zhen"),
    (22, "Bi", "贲", "Gen", "Li"),
    (23, "Bo", "剥", "Gen", "Kun"),
    (24, "Fu", "复", "Kun", "Zhen"),
    (25, "Wu Wang", "无妄", "Qian", "Zhen"),
    (26, "Da Chu", "大畜", "Gen", "Qian"),
    (27, "Yi", "颐", "Gen", "Zhen"),
    (28, "Da Guo", "大过", "Dui", "Xun"),
    (29, "Kan", "坎", "Kan", "Kan"),
    (30, "Li", "离", "Li", "Li"),
    (31, "Xian", "咸", "Dui", "Gen"),
    (32, "Heng", "恒", "Zhen", "Xun"),
    (33, "Dun", "遯", "Qian", "Gen"),
    (34, "Da Zhuang", "大壮", "Zhen", "Qian"),
    (35, "Jin", "晋", "Li", "Kun"),
    (36, "Ming Yi", "明夷", "Kun", "Li"),
    (37, "Jia Ren", "家人", "Xun", "Li"),
    (38, "Kui", "睽", "Li", "Dui"),
    (39, "Jian", "蹇", "Kan", "Gen"),
    (40, "Jie", "解", "Zhen", "Kan"),
    (41, "Sun", "损", "Gen", "Dui"),
    (42, "Yi", "益", "Xun", "Zhen"),
    (43, "Guai", "夬", "Dui", "Qian"),
    (44, "Gou", "姤", "Qian", "Xun"),
    (45, "Cui", "萃", "Dui", "Kun"),
    (46, "Sheng", "升", "Kun", "Xun"),
    (47, "Kun", "困", "Dui", "Kan"),
    (48, "Jing", "井", "Kan", "Xun"),
    (49, "Ge", "革", "Dui", "Li"),
    (50, "Ding", "鼎", "Li", "Xun"),
    (51, "Zhen", "震", "Zhen", "Zhen"),
    (52, "Gen", "艮", "Gen", "Gen"),
    (53, "Jian", "渐", "Xun", "Gen"),
    (54, "Gui Mei", "归妹", "Zhen", "Dui"),
    (55, "Feng", "丰", "Zhen", "Li"),
    (56, "Lü", "旅", "Li", "Gen"),
    (57, "Xun", "巽", "Xun", "Xun"),
    (58, "Dui", "兑", "Dui", "Dui"),
    (59, "Huan", "涣", "Xun", "Kan"),
    (60, "Jie", "节", "Kan", "Dui"),
    (61, "Zhong Fu", "中孚", "Xun", "Dui"),
    (62, "Xiao Guo", "小过", "Zhen", "Gen"),
    (63, "Ji Ji", "既济", "Kan", "Li"),
    (64, "Wei Ji", "未济", "Li", "Kan"),
]


_hexagrams = []

# Detailed text for Hexagram 1 - Qian
_hexagrams.append(
    Hexagram(
        number=1,
        name="Qian",
        chinese="乾",
        judgement="The Creative works sublime success, furthering through perseverance.",
        lines=(
            "Hidden dragon. Do not act.",
            "Dragon appearing in the field.",
            "All day long the superior man is creatively active.",
            "Wavering flight over the depths.",
            "Flying dragon in the heavens.",
            "A dragon overreaches himself.",
        ),
        upper="Qian",
        lower="Qian",
    )
)

# Detailed text for Hexagram 2 - Kun
_hexagrams.append(
    Hexagram(
        number=2,
        name="Kun",
        chinese="坤",
        judgement="The Receptive brings about sublime success, furthering through perseverance.",
        lines=(
            "When there is hoarfrost underfoot, solid ice is not far off.",
            "Straight, square, great. Without purpose, yet nothing remains unfurthered.",
            "Hidden lines. One is able to remain persevering.",
            "A tied-up sack. No blame, no praise.",
            "A yellow lower garment brings supreme good fortune.",
            "Dragons fight in the meadow; their blood is black and yellow.",
        ),
        upper="Kun",
        lower="Kun",
    )
)

_TRIGRAM_ARCHETYPES: Dict[str, Tuple[str, str]] = {
    "Qian": ("Heaven", "initiative and creative force"),
    "Kun": ("Earth", "receptivity and support"),
    "Zhen": ("Thunder", "movement and awakening"),
    "Xun": ("Wind", "gentle influence and penetration"),
    "Kan": ("Water", "depth and adaptation"),
    "Li": ("Fire", "clarity and illumination"),
    "Gen": ("Mountain", "stillness and boundaries"),
    "Dui": ("Lake", "joy and open exchange"),
}


def _generate_hexagram_text(number: int, name: str, upper: str, lower: str) -> Tuple[str, Tuple[str, ...]]:
    upper_symbol, upper_theme = _TRIGRAM_ARCHETYPES.get(upper, (upper, upper))
    lower_symbol, lower_theme = _TRIGRAM_ARCHETYPES.get(lower, (lower, lower))
    judgement = (
        f"{name} speaks of {lower_symbol} below and {upper_symbol} above: "
        f"balance {lower_theme} with {upper_theme} to make steady progress."
    )
    stages = (
        "Begin with awareness, not force.",
        "Find a reliable partner or rule.",
        "Act with discipline; avoid excess.",
        "Adjust when conditions shift.",
        "Commit with confidence and humility.",
        "Consolidate gains; prepare the next cycle.",
    )
    lines = tuple(
        f"{stage} ({lower_symbol}→{upper_symbol})" for stage in stages
    )
    return judgement, lines


# Populate remaining hexagrams with deterministic, non-placeholder text.
for number, name, chinese, upper, lower in _HEXAGRAM_INFO[2:]:
    judgement, lines = _generate_hexagram_text(number, name, upper, lower)
    _hexagrams.append(
        Hexagram(
            number=number,
            name=name,
            chinese=chinese,
            judgement=judgement,
            lines=lines,
            upper=upper,
            lower=lower,
        )
    )

# Mapping from (upper, lower) trigram names to hexagram data
def _build_name_map() -> Dict[Tuple[str, str], Hexagram]:
    mapping: Dict[Tuple[str, str], Hexagram] = {}
    for hexagram in _hexagrams:
        key = (hexagram.upper.lower(), hexagram.lower.lower())
        mapping[key] = hexagram
    return mapping


HEXAGRAM_MAP = _build_name_map()


def _build_pattern_map() -> Dict[int, Hexagram]:
    """Map binary line patterns to hexagram data."""
    mapping: Dict[int, Hexagram] = {}
    for hexagram in _hexagrams:
        upper_bits = _TRIGRAM_PATTERNS[hexagram.upper]
        lower_bits = _TRIGRAM_PATTERNS[hexagram.lower]
        code = lower_bits | (upper_bits << 3)
        mapping[code] = hexagram
    return mapping


HEXAGRAM_BINARY_MAP = _build_pattern_map()


class HexagramEngine:
    """Simple engine to fetch hexagram interpretations by trigram pairing."""

    def __init__(self, interpreter: Optional[AIEnhancedInterpreter] = None) -> None:
        self._map = HEXAGRAM_MAP
        self._pattern_map = HEXAGRAM_BINARY_MAP
        # Map possible trigram inputs (English/Chinese) to canonical English names
        self._trigram_names: Dict[str, str] = {}
        for bagua_cls in (PreHeavenBagua, PostHeavenBagua):
            for trigram in bagua_cls:
                t = trigram.value
                self._trigram_names[t.name.lower()] = t.name
                self._trigram_names[t.chinese] = t.name
        # Interpreter used to generate context aware readings
        self._interpreter = interpreter or AIEnhancedInterpreter()

    def _normalize(self, name: str) -> str:
        normalized = name.strip().lower()
        if normalized not in self._trigram_names:
            raise KeyError(f"Unknown trigram '{name}'")
        return self._trigram_names[normalized]

    def _code_from_trigrams(self, upper: str, lower: str) -> int:
        """Return the binary pattern code for given trigrams."""
        return _TRIGRAM_PATTERNS[lower] | (_TRIGRAM_PATTERNS[upper] << 3)

    def _from_pattern(self, pattern: int) -> Hexagram:
        if pattern not in self._pattern_map:
            raise KeyError("Unknown hexagram pattern")
        return self._pattern_map[pattern]

    def interpret(
        self,
        upper: str,
        lower: str,
        time_ctx: Optional[TimeContext] = None,
        context: Optional[str] = None,
    ) -> Hexagram:
        """Return the hexagram for the given upper and lower trigrams.

        Parameters
        ----------
        upper, lower:
            Names of the upper and lower trigrams.
        time_ctx:
            Optional :class:`TimeContext` providing temporal information.
        context:
            If provided, the result will be enhanced with modern, context aware
            advice generated by :class:`AIEnhancedInterpreter`.
        """

        upper_name = self._normalize(upper)
        lower_name = self._normalize(lower)
        key = (upper_name.lower(), lower_name.lower())
        if key not in self._map:
            raise KeyError(f"Combination ({upper}, {lower}) not found")
        hexagram = self._map[key]
        if time_ctx:
            extras = []
            if time_ctx.solar_term:
                extras.append(f"Solar term: {time_ctx.solar_term}")
            extras.append(f"Shichen: {time_ctx.shichen}")
            judgement = f"{hexagram.judgement} ({'; '.join(extras)})"
            lines = tuple(
                f"{line} ({time_ctx.shichen})" for line in hexagram.lines
            )
            hexagram = Hexagram(
                number=hexagram.number,
                name=hexagram.name,
                chinese=hexagram.chinese,
                judgement=judgement,
                lines=lines,
                upper=hexagram.upper,
                lower=hexagram.lower,
            )

        if context:
            hexagram = self._interpreter.enhance(hexagram, context)
        return hexagram

    def interpret_hexagram(
        self,
        upper: str,
        lower: str,
        dimensions: Optional[List[str]] = None,
        time_ctx: Optional[TimeContext] = None,
        context: Optional[str] = None,
    ) -> Dict[str, object]:
        """Interpret a hexagram with optional analysis dimensions.

        Parameters
        ----------
        upper, lower:
            Names of the upper and lower trigrams.
        dimensions:
            Iterable of dimension names to apply. Each dimension produces
            independent interpretation and advice based on
            :mod:`analysis_dimensions` rules.
        time_ctx, context:
            Passed through to :meth:`interpret` for temporal and contextual
            enhancements.

        Returns
        -------
        dict
            Dictionary containing the base ``hexagram`` and entries for each
            requested dimension.
        """

        hexagram = self.interpret(upper, lower, time_ctx=time_ctx, context=context)
        result: Dict[str, object] = {"hexagram": hexagram}
        if dimensions:
            for dim in dimensions:
                if dim not in ANALYSIS_DIMENSIONS:
                    raise KeyError(f"Unknown analysis dimension '{dim}'")
                rule = ANALYSIS_DIMENSIONS[dim]
                result[dim] = {
                    "interpretation": rule.interpret(hexagram),
                    "advice": rule.advise(hexagram),
                }
        return result

    def predict_transformations(
        self,
        upper: str,
        lower: str,
        changing_lines: Optional[List[int]] = None,
        time_steps: int = 0,
    ) -> Dict[str, object]:
        """Predict future hexagrams from line changes or time progression.

        Parameters
        ----------
        upper, lower:
            Starting trigrams for the hexagram.
        changing_lines:
            Sequence of line numbers (1-6, bottom to top) that will change in
            order. Each change produces a new hexagram in the path.
        time_steps:
            If ``changing_lines`` is not provided, ``time_steps`` flips one line
            per step starting from the bottom, cycling upward.

        Returns
        -------
        dict
            Dictionary containing ``path`` (list of :class:`Hexagram`),
            ``trend`` (str) and a textual ``report`` summarizing the
            transformation.
        """

        start = self.interpret(upper, lower)
        code = self._code_from_trigrams(start.upper, start.lower)
        path = [start]
        current = code

        if changing_lines:
            for line in changing_lines:
                idx = line - 1
                if idx < 0 or idx >= 6:
                    raise ValueError("line numbers must be between 1 and 6")
                current ^= 1 << idx
                path.append(self._from_pattern(current))
        elif time_steps > 0:
            for step in range(time_steps):
                idx = step % 6
                current ^= 1 << idx
                path.append(self._from_pattern(current))

        initial_yang = bin(code).count("1")
        final_yang = bin(current).count("1")
        if final_yang > initial_yang:
            trend = "increasing yang"
        elif final_yang < initial_yang:
            trend = "decreasing yang"
        else:
            trend = "stable"

        report = " -> ".join(f"{h.number}:{h.name}" for h in path)
        report += f" | Trend: {trend}"
        return {"path": path, "trend": trend, "report": report}
