import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.reasoning.multi_hop import MultiHopAssociator
from backend.knowledge import UnifiedKnowledgeBase
from backend.knowledge.unified import KnowledgeSource
from backend.memory import LongTermMemory
from backend.reflection import ReflectionModule, ReflectionResult
from modules.knowledge import KnowledgeFact


def test_unified_knowledge_base():
    kb = UnifiedKnowledgeBase()
    kb.add_source(KnowledgeSource(name="science", data={"atom": "basic unit"}))
    kb.add_source(KnowledgeSource(name="art", data={"atom": "indivisible style"}))
    result = kb.query("atom")
    assert result["science"] == "basic unit"
    assert result["art"] == "indivisible style"
    assert "graph" not in result or not result["graph"]


def test_unified_knowledge_base_fact_ingestion():
    kb = UnifiedKnowledgeBase()
    fact = KnowledgeFact(
        subject="sky",
        predicate="color",
        obj="blue",
        metadata={"source": "test"},
        context="observation",
    )
    kb.ingest_facts([fact])
    semantic = kb.query("sky", semantic=True, top_k=3)
    assert "facts" in semantic
    assert any(entry.get("subject") == "sky" for entry in semantic["facts"])


def test_long_term_memory(tmp_path):
    db_path = tmp_path / "memory.db"
    memory = LongTermMemory(db_path)
    memory.add("dialogue", "hello")
    memory.add("task", "write code")
    assert list(memory.get("dialogue")) == ["hello"]
    assert sorted(memory.get()) == ["hello", "write code"]
    memory.close()


def test_multi_hop_associator():
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": [],
    }
    assoc = MultiHopAssociator(graph)
    assert assoc.find_path("A", "C") == ["A", "B", "C"]


def test_reflection_module():
    module = ReflectionModule(max_passes=2, quality_threshold=2.0)
    evaluation, revised = module.reflect("test response")
    # Ensure structured evaluation is produced and history logs both passes
    assert isinstance(evaluation, ReflectionResult)
    assert isinstance(revised, str) and revised
    assert len(module.history) == 2
