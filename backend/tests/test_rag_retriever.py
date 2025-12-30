import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.rag_retriever import RAGRetriever


class _DummyLibrarian:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, embedding, n_results=5, vector_type="text", return_content=True):
        self.calls += 1
        return [f"doc{self.calls}"] * n_results


def test_rag_retriever_prefers_hot_cache():
    librarian = _DummyLibrarian()
    retriever = RAGRetriever(librarian)

    first_output = retriever.generate(
        "Question?",
        [0.5, 0.5],
        lambda prompt: prompt,
        n_results=1,
    )
    assert "doc1" in first_output
    assert librarian.calls == 1

    second_output = retriever.generate(
        "Question?",
        [0.5, 0.5],
        lambda prompt: prompt,
        n_results=1,
    )
    assert "doc1" in second_output
    assert librarian.calls == 1, "Expected hot cache to satisfy second lookup"
