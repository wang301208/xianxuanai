import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.consciousness import ConsciousnessModel


def test_conscious_access_handles_salience() -> None:
    model = ConsciousnessModel()

    salient = {"data": "important", "is_salient": True}
    non_salient = {"data": "trivial", "is_salient": False}

    assert model.conscious_access(salient) is True
    assert model.workspace.broadcasts == [salient]

    assert model.conscious_access(non_salient) is False
    # Broadcast log should remain unchanged after non-salient information
    assert model.workspace.broadcasts == [salient]
