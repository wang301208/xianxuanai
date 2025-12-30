"""Utilities for ingesting external documents into the semantic memory store."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from capability.librarian import Librarian

try:  # Optional high-quality embedding backend
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - dependency optional
    SentenceTransformer = None  # type: ignore


def _hash_embedding(text: str, dimensions: int = 12) -> List[float]:
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    chunk = max(1, len(digest) // dimensions)
    embedding: List[float] = []
    for index in range(0, len(digest), chunk):
        piece = digest[index : index + chunk]
        if not piece:
            continue
        integer = int.from_bytes(piece, byteorder="big", signed=False)
        embedding.append(integer / float(256 ** len(piece)))
        if len(embedding) == dimensions:
            break
    while len(embedding) < dimensions:
        embedding.append(0.0)
    return embedding[:dimensions]


@dataclass
class DocumentChunk:
    """Container describing a document fragment ready for storage."""

    id: str
    text: str
    metadata: Dict[str, object]
    embedding: List[float]
    vector_type: str = "text"


class DocumentIngestor:
    """Convert documents into vectorised chunks stored through :class:`Librarian`."""

    def __init__(
        self,
        librarian: Librarian,
        *,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        model_name: str = "all-MiniLM-L6-v2",
        vector_type: str = "text",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        self._librarian = librarian
        self._chunk_size = chunk_size
        self._chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self._vector_type = vector_type
        self._sentence_model: Optional[SentenceTransformer] = None
        self._model_name = model_name
        if SentenceTransformer is not None:
            try:
                self._sentence_model = SentenceTransformer(model_name)
            except Exception:  # pragma: no cover - dependency optional
                self._sentence_model = None

    # ------------------------------------------------------------------#
    def ingest_text(
        self,
        document_id: str,
        text: str,
        *,
        metadata: Optional[Dict[str, object]] = None,
    ) -> List[DocumentChunk]:
        """Slice ``text`` into chunks and persist them via the librarian."""

        base_meta = metadata.copy() if metadata else {}
        chunks = self._chunk_text(text)
        stored: List[DocumentChunk] = []
        for index, chunk_text in enumerate(chunks, start=1):
            chunk_id = f"{document_id}-{index:04d}"
            chunk_meta = {
                "document_id": document_id,
                "chunk_index": index,
                **base_meta,
            }
            embedding = self._encode(chunk_text)
            self._librarian.add_document_fragment(
                fragment_id=chunk_id,
                text=chunk_text,
                metadata=chunk_meta,
                embedding=embedding,
                vector_type=self._vector_type,
            )
            stored.append(
                DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=chunk_meta,
                    embedding=embedding,
                    vector_type=self._vector_type,
                )
            )
        return stored

    # ------------------------------------------------------------------#
    def ingest_file(
        self,
        path: str | Path,
        *,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> List[DocumentChunk]:
        """Read text from ``path`` and ingest it."""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        doc_id = document_id or path.stem
        meta = {"source_path": str(path), **(metadata or {})}
        return self.ingest_text(doc_id, text, metadata=meta)

    # ------------------------------------------------------------------#
    def ingest_directory(
        self,
        directory: str | Path,
        *,
        pattern: str = "*.txt",
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, List[DocumentChunk]]:
        """Ingest all files matching ``pattern`` within ``directory``."""

        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        results: Dict[str, List[DocumentChunk]] = {}
        for path in directory.rglob(pattern):
            doc_id = path.stem
            doc_meta = {"source_path": str(path), **(metadata or {})}
            results[doc_id] = self.ingest_file(path, document_id=doc_id, metadata=doc_meta)
        return results

    # ------------------------------------------------------------------#
    def _chunk_text(self, text: str) -> List[str]:
        normalized = text.replace("\r\n", "\n")
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", normalized) if p.strip()]
        if not paragraphs:
            paragraphs = [normalized.strip()]
        chunks: List[str] = []
        for paragraph in paragraphs:
            start = 0
            length = len(paragraph)
            step = max(1, self._chunk_size - self._chunk_overlap)
            while start < length:
                end = min(length, start + self._chunk_size)
                segment = paragraph[start:end]
                segment = self._trim_to_sentence(segment, paragraph[start:])
                cleaned = segment.strip()
                if cleaned:
                    chunks.append(cleaned)
                start += step
        return chunks

    def _trim_to_sentence(self, segment: str, remainder: str) -> str:
        if remainder and not remainder[0].isspace():
            split = max(segment.rfind(c) for c in (".", "!", "?"))
            if split > 0:
                return segment[: split + 1]
        return segment

    def _encode(self, text: str) -> List[float]:
        if self._sentence_model is not None:
            try:
                vector = self._sentence_model.encode(text, convert_to_numpy=True)
                return vector.astype(float).tolist()
            except Exception:  # pragma: no cover - degrade gracefully
                pass
        return _hash_embedding(text)


class ImageIngestor:
    """Ingest images and store their embeddings through :class:`Librarian`."""

    def __init__(
        self,
        librarian: Librarian,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        vector_type: str = "image",
    ) -> None:
        self._librarian = librarian
        self._vector_type = vector_type
        self._clip: Optional[CLIPFeatureExtractor] = None
        if CLIPFeatureExtractor is not None:
            try:
                self._clip = CLIPFeatureExtractor(model_name=model_name, pretrained=pretrained)
            except Exception:  # pragma: no cover - optional dependency might fail
                self._clip = None

    def ingest_image(
        self,
        path: str | Path,
        *,
        image_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Encode a single image and persist it via the librarian."""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        image_id = image_id or path.stem
        meta = {"source_path": str(path), **(metadata or {})}
        if description:
            meta.setdefault("description", description)

        embedding = self._encode_image(path)
        self._librarian.add_document_fragment(
            fragment_id=image_id,
            text=description or "",
            metadata=meta,
            embedding=embedding,
            vector_type=self._vector_type,
        )
        return {"id": image_id, "metadata": meta, "embedding": embedding}

    def ingest_directory(
        self,
        directory: str | Path,
        *,
        pattern: str = "*.png",
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Ingest all images matching ``pattern`` from ``directory``."""

        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(directory)

        results: Dict[str, Dict[str, object]] = {}
        for path in directory.rglob(pattern):
            results[path.stem] = self.ingest_image(path, metadata=metadata)
        return results

    def _encode_image(self, path: Path) -> List[float]:
        if self._clip is not None:
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    embedding = self._clip.extract_image_features(img)
                return embedding.detach().cpu().numpy().astype(float).tolist()
            except Exception:  # pragma: no cover - fallback if CLIP fails
                pass
        return _hash_embedding(path.read_bytes().hex())
try:  # Optional multimodal encoder
    from backend.ml.feature_extractor import CLIPFeatureExtractor
except Exception:  # pragma: no cover - dependency optional
    CLIPFeatureExtractor = None  # type: ignore

from PIL import Image
