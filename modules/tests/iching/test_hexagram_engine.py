import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from modules.iching.hexagram64 import HexagramEngine


def test_qian_hexagram():
    engine = HexagramEngine()
    h = engine.interpret("Qian", "Qian")
    assert h.number == 1
    assert h.name == "Qian"
    assert "creative" in h.judgement.lower()
    assert h.lines[0].startswith("Hidden dragon")


def test_kun_hexagram_chinese_names():
    engine = HexagramEngine()
    h = engine.interpret("坤", "坤")
    assert h.number == 2
    assert h.name == "Kun"
    assert "receptive" in h.judgement.lower()


def test_all_combinations_covered():
    engine = HexagramEngine()
    trigrams = [
        "Qian",
        "Dui",
        "Li",
        "Zhen",
        "Xun",
        "Kan",
        "Gen",
        "Kun",
    ]
    numbers = set()
    for upper in trigrams:
        for lower in trigrams:
            numbers.add(engine.interpret(upper, lower).number)
    assert len(numbers) == 64


def test_predict_transformations_line_change():
    engine = HexagramEngine()
    result = engine.predict_transformations("Qian", "Qian", changing_lines=[6])
    numbers = [h.number for h in result["path"]]
    assert numbers == [1, 9]
    assert result["trend"] == "decreasing yang"


def test_predict_transformations_time_steps():
    engine = HexagramEngine()
    result = engine.predict_transformations("Kun", "Kun", time_steps=2)
    numbers = [h.number for h in result["path"]]
    assert numbers == [2, 15, 46]
    assert result["trend"] == "increasing yang"
