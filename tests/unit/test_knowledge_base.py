from __future__ import annotations

from pathlib import Path

import pytest

from modules.knowledge.knowledge_base import KnowledgeBase


def test_knowledge_base_save_and_query_with_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "kb.db"
    kb = KnowledgeBase(db_path=db_path, enabled=True, embedding_enabled=True)

    entry_id = kb.save_memory("experience", "hello world", tags=["greeting"], metadata={"source": "test"})
    assert isinstance(entry_id, int)

    results = kb.query_memory("hello world", top_k=5)
    assert results
    assert any(item.id == entry_id for item in results)


def test_knowledge_base_fallback_search(tmp_path: Path) -> None:
    db_path = tmp_path / "kb.db"
    kb = KnowledgeBase(db_path=db_path, enabled=True, embedding_enabled=False)

    entry_id = kb.save_memory("experience", "keyword-match", tags=["k"], metadata={"source": "test"})
    assert isinstance(entry_id, int)

    results = kb.query_memory("keyword", top_k=5, allow_fallback_search=True)
    assert results
    assert results[0].content == "keyword-match"


def test_knowledge_base_disabled(tmp_path: Path) -> None:
    kb = KnowledgeBase(db_path=tmp_path / "kb.db", enabled=False, embedding_enabled=False)
    assert kb.save_memory("experience", "x") is None
    assert kb.query_memory("x") == []
    assert kb.recent() == []


@pytest.mark.parametrize("category", [None, "experience"])
def test_knowledge_base_recent(tmp_path: Path, category: str | None) -> None:
    kb = KnowledgeBase(db_path=tmp_path / "kb.db", enabled=True, embedding_enabled=False)
    kb.save_memory(category or "experience", "recent-item", tags=["t"])
    items = kb.recent(limit=10, category=category)
    assert items
    assert items[0].content == "recent-item"

