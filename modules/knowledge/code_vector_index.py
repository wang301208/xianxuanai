from __future__ import annotations

"""Lightweight local code vector index.

The goal is to let the agent build a semantic index over local source trees
(including cloned external repos) and later retrieve relevant implementations
for few-shot prompting / skill synthesis.

This module is dependency-light by design:
- Embeddings default to a deterministic hash-based encoder (no GPU/LLM needed).
- Storage uses `backend.knowledge.vector_store.LocalVectorStore` (FAISS optional).
- Persistence is a single JSON file so the index can accumulate across runs.
"""

import ast
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from backend.knowledge.vector_store import LocalVectorStore

logger = logging.getLogger(__name__)

ReadTextFn = Callable[[Path, int], str]

_TOKEN_PATTERN = re.compile(r"\w+")

def _hash_embedding(text: str, dimensions: int = 12) -> List[float]:
    """Deterministic hashing embedder with token overlap signal.

    This is a lightweight, dependency-free approximation of a bag-of-ngrams
    embedding. It is not comparable to transformer embeddings, but unlike a
    raw digest chunking, it preserves some lexical similarity useful for code
    search in minimal environments.
    """

    dims = max(1, int(dimensions))
    tokens = _TOKEN_PATTERN.findall(str(text or "").lower())
    if not tokens:
        return [0.0 for _ in range(dims)]

    vector = [0.0 for _ in range(dims)]
    window_size = 2
    for index in range(len(tokens)):
        window = tokens[index : index + window_size]
        if not window:
            continue
        ngram = " ".join(window).encode("utf-8")
        digest = hashlib.sha256(ngram).digest()
        bucket = int.from_bytes(digest[:4], "little", signed=False) % dims
        vector[bucket] += 1.0

    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _safe_read_text(path: Path, *, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        item = str(value or "")
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    symbol: str
    kind: str
    text: str

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "path": self.path,
            "start_line": int(self.start_line),
            "end_line": int(self.end_line),
            "symbol": self.symbol,
            "kind": self.kind,
            "text": self.text,
        }


class CodeVectorIndex:
    """Persisted semantic index over a local code tree."""

    def __init__(
        self,
        *,
        root: str | os.PathLike[str] = ".",
        persist_path: str | os.PathLike[str] | None = None,
        read_text_fn: ReadTextFn | None = None,
        include_suffixes: Sequence[str] = (".py",),
        exclude_dirs: Sequence[str] = (
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "node_modules",
            ".bss_sandbox",
        ),
        max_files: int = 800,
        max_file_chars: int = 250_000,
        chunk_lines: int = 80,
        chunk_overlap: int = 10,
        max_chunk_chars: int = 8_000,
        embedding_dimensions: int = 12,
        use_faiss: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.persist_path = Path(persist_path).resolve() if persist_path is not None else None
        self.include_suffixes = tuple(str(s).lower() for s in include_suffixes)
        self.exclude_dirs = set(str(d) for d in exclude_dirs)
        self.max_files = max(1, int(max_files))
        self.max_file_chars = max(1, int(max_file_chars))
        self.chunk_lines = max(5, int(chunk_lines))
        self.chunk_overlap = max(0, int(chunk_overlap))
        self.max_chunk_chars = max(200, int(max_chunk_chars))
        self.embedding_dimensions = max(4, int(embedding_dimensions))
        self.use_faiss = bool(use_faiss)
        self._read_text_fn = read_text_fn

        self._store: Optional[LocalVectorStore] = None
        self._entries: List[Dict[str, Any]] = []
        self._built_at: float | None = None

    # ------------------------------------------------------------------ public API
    @property
    def is_loaded(self) -> bool:
        return self._store is not None and bool(self._entries)

    def load(self) -> bool:
        if self.persist_path is None or not self.persist_path.exists():
            return False
        try:
            payload = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Unable to load code index from %s", self.persist_path, exc_info=True)
            return False
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list) or not entries:
            return False

        store = LocalVectorStore(self.embedding_dimensions, use_faiss=self.use_faiss)
        loaded: List[Dict[str, Any]] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            embedding = item.get("embedding")
            meta = item.get("meta")
            if not isinstance(embedding, list) or not isinstance(meta, dict):
                continue
            try:
                store.add([float(v) for v in embedding], dict(meta))
            except Exception:
                continue
            loaded.append({"embedding": embedding, "meta": dict(meta)})

        if not loaded:
            return False
        self._store = store
        self._entries = loaded
        self._built_at = float(payload.get("built_at") or 0.0) or None
        return True

    def build(self, *, save: bool = True) -> Dict[str, Any]:
        started = time.time()
        files = list(self._iter_files())
        store = LocalVectorStore(self.embedding_dimensions, use_faiss=self.use_faiss)
        entries: List[Dict[str, Any]] = []
        chunks_indexed = 0
        file_errors = 0

        for path in files:
            try:
                for chunk in self._extract_chunks(path):
                    embedding = _hash_embedding(self._chunk_to_embedding_text(chunk), self.embedding_dimensions)
                    meta = chunk.to_metadata()
                    store.add(embedding, meta)
                    entries.append({"embedding": embedding, "meta": meta})
                    chunks_indexed += 1
            except Exception:
                file_errors += 1
                logger.debug("Failed to index %s", path, exc_info=True)

        self._store = store if entries else None
        self._entries = entries
        self._built_at = time.time()

        stats = {
            "root": str(self.root),
            "files_scanned": len(files),
            "chunks_indexed": chunks_indexed,
            "file_errors": file_errors,
            "duration_s": round(time.time() - started, 4),
            "persist_path": str(self.persist_path) if self.persist_path is not None else None,
        }

        if save and self.persist_path is not None and entries:
            self._save(entries)

        return stats

    def search(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        query_text = str(query or "").strip()
        if not query_text:
            return []
        if self._store is None:
            if not self.load():
                return []
        store = self._store
        assert store is not None
        embedding = _hash_embedding(query_text, self.embedding_dimensions)
        hits = store.search(embedding, top_k=max(1, int(top_k)))
        return [dict(hit) for hit in hits if isinstance(hit, dict)]

    def hits_as_references(self, hits: Sequence[Dict[str, Any]], *, max_chars: int = 2000) -> List[Dict[str, str]]:
        refs: List[Dict[str, str]] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            path = str(hit.get("path") or "")
            start = int(hit.get("start_line") or 0)
            end = int(hit.get("end_line") or 0)
            symbol = str(hit.get("symbol") or "")
            kind = str(hit.get("kind") or "chunk")
            sim = hit.get("similarity")
            title_parts = [path]
            if symbol:
                title_parts.append(symbol)
            if start:
                title_parts.append(f"L{start}")
            if sim is not None:
                try:
                    title_parts.append(f"sim={float(sim):.3f}")
                except Exception:
                    pass
            title = " | ".join(title_parts) or "code_chunk"

            snippet = str(hit.get("text") or "").strip()
            if max_chars > 0 and len(snippet) > max_chars:
                snippet = snippet[:max_chars].rstrip() + "..."
            if not snippet and path:
                snippet = f"[{kind}] {path}:{start}-{end}"

            refs.append({"title": title, "url": path, "snippet": snippet})
        return refs

    # ------------------------------------------------------------------ internals
    def _save(self, entries: List[Dict[str, Any]]) -> None:
        assert self.persist_path is not None
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "root": str(self.root),
            "embedding_dimensions": int(self.embedding_dimensions),
            "built_at": float(self._built_at or time.time()),
            "entries": entries,
        }
        self.persist_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _iter_files(self) -> Iterable[Path]:
        if not self.root.exists():
            return []
        suffixes = set(self.include_suffixes)
        scanned = 0
        files: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs]
            for fname in filenames:
                if scanned >= self.max_files:
                    break
                path = Path(dirpath) / fname
                if path.suffix.lower() not in suffixes:
                    continue
                files.append(path)
                scanned += 1
            if scanned >= self.max_files:
                break
        return files

    def _extract_chunks(self, path: Path) -> List[CodeChunk]:
        rel_path = str(path.resolve().relative_to(self.root)) if path.exists() else str(path)
        text = self._read_text(path, max_chars=self.max_file_chars)
        if not text:
            return []

        suffix = path.suffix.lower()
        if suffix == ".py":
            chunks = self._extract_python_defs(text, rel_path)
            if chunks:
                return chunks
        return self._chunk_by_lines(text, rel_path, kind="file")

    def _extract_python_defs(self, text: str, rel_path: str) -> List[CodeChunk]:
        try:
            tree = ast.parse(text)
        except Exception:
            return []

        lines = text.splitlines()
        chunks: List[CodeChunk] = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            symbol = getattr(node, "name", "") or "symbol"
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            start = int(getattr(node, "lineno", 1) or 1)
            end = int(getattr(node, "end_lineno", 0) or 0)
            if end <= 0 or end < start:
                end = min(len(lines), start + self.chunk_lines)
            snippet_lines = lines[start - 1 : end]
            snippet = "\n".join(snippet_lines).strip()
            if not snippet:
                continue
            if len(snippet) > self.max_chunk_chars:
                snippet = snippet[: self.max_chunk_chars].rstrip() + "..."
            chunk_id = f"{rel_path}:{start}-{end}:{kind}:{symbol}"
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    path=rel_path,
                    start_line=start,
                    end_line=end,
                    symbol=symbol,
                    kind=kind,
                    text=snippet,
                )
            )

        return chunks

    def _chunk_by_lines(self, text: str, rel_path: str, *, kind: str) -> List[CodeChunk]:
        lines = text.splitlines()
        if not lines:
            return []
        step = max(1, self.chunk_lines - self.chunk_overlap)
        chunks: List[CodeChunk] = []
        start = 0
        while start < len(lines):
            end = min(len(lines), start + self.chunk_lines)
            snippet = "\n".join(lines[start:end]).strip()
            if snippet:
                if len(snippet) > self.max_chunk_chars:
                    snippet = snippet[: self.max_chunk_chars].rstrip() + "..."
                start_line = start + 1
                end_line = end
                chunk_id = f"{rel_path}:{start_line}-{end_line}:{kind}"
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        path=rel_path,
                        start_line=start_line,
                        end_line=end_line,
                        symbol="",
                        kind=kind,
                        text=snippet,
                    )
                )
            start += step
        return chunks

    def _chunk_to_embedding_text(self, chunk: CodeChunk) -> str:
        header = " ".join(_dedupe_preserve_order([chunk.kind, chunk.symbol, chunk.path])).strip()
        raw = f"{header}\n{chunk.text}"
        raw = re.sub(r"\\s+", " ", raw).strip()
        return raw

    def _read_text(self, path: Path, *, max_chars: int) -> str:
        reader = self._read_text_fn
        if reader is None:
            return _safe_read_text(path, max_chars=max_chars)
        try:
            text = reader(path, int(max_chars))
        except Exception:
            logger.debug("read_text_fn failed for %s", path, exc_info=True)
            return ""
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars]
        return text


__all__ = ["CodeVectorIndex", "CodeChunk"]
