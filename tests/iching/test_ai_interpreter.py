import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.iching.hexagram64 import HexagramEngine


def test_contextual_interpretation_differs():
    engine = HexagramEngine()
    business_reading = engine.interpret("Qian", "Qian", context="business")
    relation_reading = engine.interpret("Qian", "Qian", context="relationships")

    # The AI enhanced interpreter should tailor the judgement and line texts
    # based on the provided context resulting in different outputs.
    assert business_reading.judgement != relation_reading.judgement
    assert business_reading.lines != relation_reading.lines
    assert "business" in business_reading.lines[0]
    assert "relationships" in relation_reading.lines[0]
