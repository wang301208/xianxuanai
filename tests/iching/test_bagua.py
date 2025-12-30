import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytest

from modules.iching.bagua import get_trigram


def test_pre_heaven_bagua_mapping():
    expected = {
        "Qian": ("乾", "☰", "South"),
        "Dui": ("兑", "☱", "Southeast"),
        "Li": ("离", "☲", "East"),
        "Zhen": ("震", "☳", "Northeast"),
        "Xun": ("巽", "☴", "Southwest"),
        "Kan": ("坎", "☵", "West"),
        "Gen": ("艮", "☶", "Northwest"),
        "Kun": ("坤", "☷", "North"),
    }
    for name, (chinese, symbol, direction) in expected.items():
        trigram = get_trigram("pre", name)
        assert (trigram.chinese, trigram.symbol, trigram.direction) == (
            chinese,
            symbol,
            direction,
        )
        assert get_trigram("pre", chinese) == trigram


def test_post_heaven_bagua_mapping():
    expected = {
        "Qian": ("乾", "☰", "Northwest"),
        "Dui": ("兑", "☱", "West"),
        "Li": ("离", "☲", "South"),
        "Zhen": ("震", "☳", "East"),
        "Xun": ("巽", "☴", "Southeast"),
        "Kan": ("坎", "☵", "North"),
        "Gen": ("艮", "☶", "Northeast"),
        "Kun": ("坤", "☷", "Southwest"),
    }
    for name, (chinese, symbol, direction) in expected.items():
        trigram = get_trigram("post", name)
        assert (trigram.chinese, trigram.symbol, trigram.direction) == (
            chinese,
            symbol,
            direction,
        )
        assert get_trigram("post", chinese) == trigram
