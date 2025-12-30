import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.memory.long_term import LongTermMemory
from backend.knowledge.guard import KnowledgeGuard, ValidationResult
from backend.knowledge.consolidation import KnowledgeConsolidator


class _LowConfidenceValidator:
    def __init__(self, confidence: float, status: str) -> None:
        self.confidence = confidence
        self.status = status
        self.calls = []

    def __call__(self, entry):
        self.calls.append(entry)
        return ValidationResult(self.confidence, self.status, {"checked": True}, reason="unit-test")


def test_guard_updates_long_term_memory(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    try:
        validator = _LowConfidenceValidator(0.2, "needs_review")
        guard = KnowledgeGuard(
            memory=mem,
            validators=[validator],
            base_status="pending",
            auto_promote_threshold=0.9,
            demote_threshold=0.3,
        )
        entry_id = mem.add("facts", "earth_orbits_sun", tags=["astronomy"])
        result = guard.evaluate(
            "Earth orbits the Sun",
            source="sensor:astro",
            entry_id=entry_id,
            metadata={"concept_id": "demo-node"},
        )
        assert result.status == "needs_review"
        records = list(
            mem.get(
                include_metadata=True,
                status=["needs_review"],
            )
        )
        assert records, "expected guard to update long-term memory status"
        assert records[0]["metadata"].get("checked") is True
        assert records[0]["confidence"] == result.confidence
    finally:
        mem.close()


class _DummyMemory:
    def __init__(self) -> None:
        self.updates = []

    def update_entry(self, entry_id, **fields):
        self.updates.append((entry_id, fields))


def test_consolidator_uses_guard_to_demote_hot_concepts():
    validator = _LowConfidenceValidator(0.2, "needs_review")
    guard = KnowledgeGuard(
        memory=_DummyMemory(),  # type: ignore[arg-type]
        validators=[validator],
        auto_promote_threshold=0.9,
        demote_threshold=0.3,
    )
    consolidator = KnowledgeConsolidator(hot_limit=1, guard=guard)
    try:
        consolidator.record_statement(
            "Self-updating systems must validate new rules.",
            source="unit:test",
        )
        consolidator.wait_idle()
        assert validator.calls, "expected validator to be invoked"
        assert not consolidator.hot_concepts, "low-confidence concept should be demoted from hot tier"
    finally:
        consolidator.stop()
