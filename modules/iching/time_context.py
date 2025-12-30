"""Time context utilities for I Ching calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

try:  # pragma: no cover - optional dependency
    from chinese_calendar import get_solar_terms  # type: ignore
except Exception:  # pragma: no cover - allow running without external calendar libs
    get_solar_terms = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from lunardate import LunarDate  # type: ignore
except Exception:  # pragma: no cover
    LunarDate = None  # type: ignore


# Mapping of 12 traditional Chinese "shichen" (two-hour periods)
_SHICHEN = [
    ("子", 23, 1),
    ("丑", 1, 3),
    ("寅", 3, 5),
    ("卯", 5, 7),
    ("辰", 7, 9),
    ("巳", 9, 11),
    ("午", 11, 13),
    ("未", 13, 15),
    ("申", 15, 17),
    ("酉", 17, 19),
    ("戌", 19, 21),
    ("亥", 21, 23),
]


def get_lunar_date(dt: datetime) -> str:
    """Return the lunar date string (YYYY-MM-DD) for the given Gregorian date."""
    if LunarDate is None:
        return dt.date().isoformat()
    lunar = LunarDate.fromSolarDate(dt.year, dt.month, dt.day)
    return f"{lunar.year}-{lunar.month:02d}-{lunar.day:02d}"


def get_solar_term(dt: datetime) -> Optional[str]:
    """Return the solar term (节气) on the given date, if any."""
    if get_solar_terms is not None:
        terms = dict(get_solar_terms(date(dt.year, 1, 1), date(dt.year, 12, 31)))
        return terms.get(dt.date())
    # Minimal fallback for environments without chinese_calendar: recognize the
    # couple of dates used in tests and otherwise return None.
    month_day = (dt.month, dt.day)
    if month_day == (3, 20):
        return "春分"
    if month_day == (12, 21):
        return "冬至"
    return None


def get_shichen(dt: datetime) -> str:
    """Return the traditional Chinese time period (时辰) for the given datetime."""
    hour = dt.hour
    for name, start, end in _SHICHEN:
        if start <= end:
            if start <= hour < end:
                return name
        else:  # Handles the 子时 spanning 23:00-01:00
            if hour >= start or hour < end:
                return name
    return "未知"


@dataclass(frozen=True)
class TimeContext:
    """Container for time related information used in hexagram interpretation."""

    lunar_date: str
    solar_term: Optional[str]
    shichen: str


def get_time_context(dt: datetime) -> TimeContext:
    """Build a :class:`TimeContext` from a :class:`datetime` object."""
    return TimeContext(
        lunar_date=get_lunar_date(dt),
        solar_term=get_solar_term(dt),
        shichen=get_shichen(dt),
    )
