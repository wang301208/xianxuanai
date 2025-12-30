import os
import sys
from pathlib import Path

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from backend.knowledge.ingest import ImageIngestor


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


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (4, 4), color=color)
    img.save(path, format="PNG")


def test_ingest_single_image(tmp_path):
    librarian = StubLibrarian()
    ingestor = ImageIngestor(librarian)

    image_path = tmp_path / "sample.png"
    _create_image(image_path, (255, 0, 0))

    result = ingestor.ingest_image(image_path, description="Red square", metadata={"category": "icon"})

    assert result["id"] == "sample"
    assert librarian.fragments
    stored = librarian.fragments[0]
    assert stored["vector_type"] == "image"
    assert stored["metadata"]["description"] == "Red square"
    assert stored["metadata"]["category"] == "icon"
    assert len(stored["embedding"]) > 0


def test_ingest_image_directory(tmp_path):
    librarian = StubLibrarian()
    ingestor = ImageIngestor(librarian)

    _create_image(tmp_path / "logo1.png", (0, 255, 0))
    _create_image(tmp_path / "logo2.png", (0, 0, 255))

    results = ingestor.ingest_directory(tmp_path, pattern="logo*.png", metadata={"collection": "logos"})

    assert len(results) == 2
    assert len(librarian.fragments) == 2
    for fragment in librarian.fragments:
        assert fragment["metadata"]["collection"] == "logos"
        assert fragment["vector_type"] == "image"
