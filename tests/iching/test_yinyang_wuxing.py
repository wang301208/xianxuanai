import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytest

from modules.iching.yinyang_wuxing import YinYangFiveElements


def test_transform_yinyang_strings():
    assert YinYangFiveElements.transform_yinyang("yin") == "yang"
    assert YinYangFiveElements.transform_yinyang("YANG") == "yin"


def test_transform_yinyang_numbers_and_bool():
    assert YinYangFiveElements.transform_yinyang(0) == 1
    assert YinYangFiveElements.transform_yinyang(1) == 0
    assert YinYangFiveElements.transform_yinyang(True) is False


def test_transform_yinyang_invalid():
    with pytest.raises(ValueError):
        YinYangFiveElements.transform_yinyang("neutral")


def test_element_generation_and_control():
    y = YinYangFiveElements
    assert y.element_interaction("wood", "fire") == "generate"
    assert y.element_interaction("fire", "wood") == "generated_by"
    assert y.element_interaction("wood", "earth") == "control"
    assert y.element_interaction("earth", "wood") == "controlled_by"


def test_element_same():
    assert YinYangFiveElements.element_interaction("water", "water") == "same"
