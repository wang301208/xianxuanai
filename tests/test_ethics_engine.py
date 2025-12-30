import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain import EthicalReasoningEngine, EthicalRule, MotorCortex


def test_ethics_blocks_harmful_action():
    rules = [
        EthicalRule(
            condition=lambda action: "harm" in action.lower(),
            value="nonmaleficence",
            suggestion="Avoid causing harm; choose a benign alternative.",
        )
    ]
    weights = {"nonmaleficence": 10}
    engine = EthicalReasoningEngine(rules=rules, value_weights=weights)
    cortex = MotorCortex(ethics=engine)

    report = cortex.execute_action("harm humans")

    assert not report["compliant"]
    assert report["score"] == 10
    assert "Avoid causing harm" in report["suggestions"][0]


def test_ethics_allows_safe_action():
    engine = EthicalReasoningEngine(
        rules=[
            EthicalRule(
                condition=lambda action: "steal" in action.lower(),
                value="justice",
                suggestion="Respect property rights.",
            )
        ]
    )
    cortex = MotorCortex(ethics=engine)

    result = cortex.execute_action("wave")

    assert isinstance(result, str)
    assert "executed" in result
