from modules.brain.reasoning import CommonSenseReasoner


def _mock_fetch(self, concept):
    data = {
        "dog": [("dog IsA animal", 1.0)],
        "car": [("car IsA vehicle", 0.9)],
    }
    return data.get(concept, [])


def _accuracy(reasoner: CommonSenseReasoner) -> float:
    dataset = [("dog", "animal"), ("car", "vehicle")]
    correct = 0
    for query, expected in dataset:
        results = reasoner.infer(query)
        if any(expected in concl for concl, _ in results):
            correct += 1
    return correct / len(dataset)


def test_accuracy_improves_with_commonsense(monkeypatch):
    monkeypatch.setattr(CommonSenseReasoner, "_fetch_conceptnet_edges", _mock_fetch)
    disabled = CommonSenseReasoner(enabled=False)
    enabled = CommonSenseReasoner()
    assert _accuracy(disabled) < _accuracy(enabled)
