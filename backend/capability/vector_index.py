from __future__ import annotations

from typing import Dict, List, Any

try:
    import chromadb
    from chromadb.config import Settings
except Exception as exc:  # pragma: no cover - chromadb is optional in tests
    chromadb = None
    Settings = None


class VectorIndex:
    """Wrapper around ChromaDB for skill embeddings."""

    def __init__(self, persist_directory: str | None = None) -> None:
        if chromadb is None:
            raise ImportError("chromadb is required for VectorIndex")
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collections: Dict[str, Any] = {}

    def _get_collection(self, vector_type: str) -> Any:
        """Get or create a collection for the given vector type."""
        if vector_type not in self.collections:
            self.collections[vector_type] = self.client.get_or_create_collection(vector_type)
        return self.collections[vector_type]

    def add(
        self,
        doc_id: str,
        embedding: List[float],
        metadata: Dict[str, Any] | None = None,
        vector_type: str = "text",
    ) -> None:
        collection = self._get_collection(vector_type)
        collection.add(ids=[doc_id], embeddings=[embedding], metadatas=[metadata or {}])

    def add_image_embedding(
        self, doc_id: str, embedding: List[float], metadata: Dict[str, Any] | None = None
    ) -> None:
        """Convenience method to store an image embedding."""
        self.add(doc_id, embedding, metadata, vector_type="image")

    def query(
        self, embedding: List[float], n_results: int = 1, vector_type: str = "text"
    ) -> Dict[str, Any]:
        collection = self._get_collection(vector_type)
        return collection.query(query_embeddings=[embedding], n_results=n_results)
