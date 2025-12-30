import pathlib
import sys
from datetime import datetime

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from modules.iching.hexagram64 import HexagramEngine
from modules.iching.time_context import get_time_context


def test_solar_term_changes_judgement():
    engine = HexagramEngine()
    ctx_spring = get_time_context(datetime(2024, 3, 20, 9))
    ctx_winter = get_time_context(datetime(2024, 12, 21, 9))
    h_spring = engine.interpret("Qian", "Qian", time_ctx=ctx_spring)
    h_winter = engine.interpret("Qian", "Qian", time_ctx=ctx_winter)
    assert h_spring.judgement != h_winter.judgement


def test_shichen_changes_lines():
    engine = HexagramEngine()
    morning = get_time_context(datetime(2024, 6, 21, 6))
    evening = get_time_context(datetime(2024, 6, 21, 20))
    h_morning = engine.interpret("Qian", "Qian", time_ctx=morning)
    h_evening = engine.interpret("Qian", "Qian", time_ctx=evening)
    assert h_morning.lines[0] != h_evening.lines[0]
