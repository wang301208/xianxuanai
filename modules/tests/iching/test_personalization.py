import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from modules.iching.personalization import (
    UserProfile,
    PersonalizedHexagramEngine,
)


def test_personalized_output_differs():
    engine = PersonalizedHexagramEngine()
    user1 = UserProfile("u1", preferences={"focus": "career"})
    user2 = UserProfile("u2", preferences={"focus": "relationships"})

    h1 = engine.interpret("Qian", "Qian", user1)
    h2 = engine.interpret("Qian", "Qian", user2)

    assert "career" in h1.judgement.lower()
    assert "relationships" in h2.judgement.lower()
    assert h1.judgement != h2.judgement
    assert user1.history == [1]
    assert user2.history == [1]
