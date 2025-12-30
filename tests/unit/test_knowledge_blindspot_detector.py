from __future__ import annotations

from backend.execution.knowledge_blindspot import (
    KnowledgeBlindspotConfig,
    KnowledgeBlindspotDetector,
)


class _StubRouter:
    def __init__(self, results):
        self._results = list(results)

    def query(self, text: str, *, top_k: int = 5):
        _ = text, top_k
        return list(self._results)


def test_disabled_detector_returns_disabled_assessment() -> None:
    detector = KnowledgeBlindspotDetector(
        memory_router=_StubRouter([]),
        config=KnowledgeBlindspotConfig(enabled=False),
    )
    assessment = detector.assess(query_text="any", keywords=["python"])
    assert assessment.enabled is False
    assert assessment.level == "disabled"
    assert assessment.blindspot is False


def test_memory_support_above_threshold_is_not_blindspot() -> None:
    router = _StubRouter([{"similarity": 0.9}, {"similarity": 0.6}, {"similarity": 0.1}])
    detector = KnowledgeBlindspotDetector(
        memory_router=router,
        config=KnowledgeBlindspotConfig(
            enabled=True,
            docs_enabled=False,
            min_similarity=0.5,
            min_support=2,
            top_k=5,
        ),
    )
    assessment = detector.assess(query_text="topic", keywords=["topic"])
    assert assessment.enabled is True
    assert assessment.memory_hits == 2
    assert assessment.support == 2
    assert assessment.blindspot is False
    assert assessment.level == "ok"


def test_empty_support_is_blindspot_and_emits_declaration() -> None:
    router = _StubRouter([{"similarity": 0.1}])
    detector = KnowledgeBlindspotDetector(
        memory_router=router,
        config=KnowledgeBlindspotConfig(
            enabled=True,
            docs_enabled=False,
            min_similarity=0.5,
            min_support=2,
            top_k=5,
        ),
    )
    assessment = detector.assess(query_text="unknown topic", keywords=["unknown"])
    assert assessment.blindspot is True
    assert assessment.level in {"empty", "sparse"}
    assert assessment.declaration_zh
    assert assessment.declaration_en

