import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from backend.knowledge.ingest import DocumentIngestor


class StubLibrarian:
    def __init__(self) -> None:
        self.fragments = []

    def add_document_fragment(self, fragment_id, text, metadata, embedding, *, vector_type="text"):
        self.fragments.append(
            {
                "id": fragment_id,
                "text": text,
                "metadata": metadata,
                "embedding": embedding,
                "vector_type": vector_type,
            }
        )


def test_ingest_text_splits_into_chunks():
    librarian = StubLibrarian()
    ingestor = DocumentIngestor(librarian, chunk_size=50, chunk_overlap=10)

    text = (
        "Autonomous agents require reliable knowledge access. "
        "Chunking documents helps with retrieval.\n\n"
        "Second paragraph provides additional context about execution environments."
    )

    stored = ingestor.ingest_text("doc001", text, metadata={"source": "whitepaper"})

    assert stored
    assert librarian.fragments
    for fragment in librarian.fragments:
        assert fragment["metadata"]["document_id"] == "doc001"
        assert fragment["metadata"]["source"] == "whitepaper"
        assert fragment["vector_type"] == "text"
        assert fragment["text"]
        assert len(fragment["embedding"]) > 0


def test_ingest_file_reads_from_disk(tmp_path):
    librarian = StubLibrarian()
    ingestor = DocumentIngestor(librarian, chunk_size=40, chunk_overlap=5)

    sample_file = tmp_path / "manual.txt"
    sample_file.write_text("Line one describing system.\nLine two elaborates on architecture.", encoding="utf-8")

    results = ingestor.ingest_file(sample_file, metadata={"category": "manual"})

    assert results
    assert librarian.fragments
    first_fragment = librarian.fragments[0]
    assert first_fragment["metadata"]["source_path"] == str(sample_file)
    assert first_fragment["metadata"]["category"] == "manual"
