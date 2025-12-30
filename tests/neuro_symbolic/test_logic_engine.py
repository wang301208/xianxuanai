import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from algorithms.neuro_symbolic.logic_engine import LogicEngine


def test_logic_engine_evaluates_rules():
    nn_output = {"a": 1, "b": 0}
    rules = {"c": "a and not b", "d": "a or b"}
    engine = LogicEngine(rules)
    result = engine.evaluate(nn_output)
    assert result == {"c": True, "d": True}


def test_unknown_symbol_raises_error():
    engine = LogicEngine({"c": "missing"})
    with pytest.raises(ValueError):
        engine.evaluate({"a": True})
