import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.iching.hexagram64 import HexagramEngine


def test_interpret_hexagram_multi_dimensions_independent():
    engine = HexagramEngine()
    dims = ["career", "health"]
    combined = engine.interpret_hexagram("Qian", "Kun", dimensions=dims)

    assert set(combined.keys()) == {"hexagram", "career", "health"}

    single_career = engine.interpret_hexagram("Qian", "Kun", dimensions=["career"])
    single_health = engine.interpret_hexagram("Qian", "Kun", dimensions=["health"])

    assert combined["career"] == single_career["career"]
    assert combined["health"] == single_health["health"]


def test_interpret_hexagram_unknown_dimension():
    engine = HexagramEngine()
    with pytest.raises(KeyError):
        engine.interpret_hexagram("Qian", "Kun", dimensions=["unknown"])
