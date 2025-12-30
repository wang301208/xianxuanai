from modules.brain.reasoning import AnalogicalReasoner


def test_transfer_monarchy_to_corporate():
    reasoner = AnalogicalReasoner()
    reasoner.add_knowledge(
        "monarchy",
        {"leader": "king", "followers": "subjects"},
        "king rules subjects",
    )
    mapping = reasoner.transfer_knowledge(
        "monarchy",
        "CEO leads employees",
        {"leader": "CEO", "followers": "employees"},
    )
    assert mapping == {"king": "CEO", "subjects": "employees"}


def test_transfer_solar_to_atomic():
    reasoner = AnalogicalReasoner()
    reasoner.add_knowledge(
        "solar",
        {"central": "sun", "orbiting": "planet"},
        "sun attracts planet with gravity",
    )
    mapping = reasoner.transfer_knowledge(
        "solar",
        "nucleus attracts electron electromagnetically",
        {"central": "nucleus", "orbiting": "electron"},
    )
    assert mapping == {"sun": "nucleus", "planet": "electron"}

