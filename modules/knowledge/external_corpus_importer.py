from __future__ import annotations

"""Bulk import a local document/code corpus into the internal knowledge graph.

This module is designed for the "external knowledge indexing" workflow:
- Pre-collect algorithm notes, code snippets, and API docs into a local folder.
- Import them as concept nodes (with a short description + provenance metadata).
- Query them later via knowledge-graph tooling (e.g. knowledge_query ability/tool).

The implementation is dependency-light and falls back to hash-based embeddings
when optional encoders are unavailable.
"""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.knowledge.importer import BulkKnowledgeImporter, RawConcept, RawRelation
from backend.knowledge.registry import get_default_aligner, get_graph_store_instance, set_default_aligner


_DEFAULT_EXCLUDE_DIRS: Tuple[str, ...] = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "node_modules",
    ".bss_sandbox",
)

_DEFAULT_SUFFIXES: Tuple[str, ...] = (
    ".md",
    ".txt",
    ".rst",
    ".py",
)

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
_MD_TITLE_RE = re.compile(r"^\s{0,3}#\s+(.+?)\s*$")


def _slugify(text: str) -> str:
    slug = _SLUG_PATTERN.sub("_", str(text or "").strip().lower()).strip("_")
    return slug or "doc"


def _stable_id(*parts: str, length: int = 12) -> str:
    payload = "|".join(str(p or "") for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[: max(6, int(length))]


def _safe_read_text(path: Path, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _extract_title(path: Path, text: str) -> str:
    if not text:
        return path.stem
    if path.suffix.lower() in {".md", ".rst"}:
        for line in text.splitlines()[:40]:
            match = _MD_TITLE_RE.match(line)
            if match:
                title = match.group(1).strip()
                if title:
                    return title
    return path.stem


def _iter_files(
    root: Path,
    *,
    include_suffixes: Sequence[str],
    exclude_dirs: Sequence[str],
    max_files: int,
) -> List[Path]:
    suffixes = {str(s).lower() for s in include_suffixes if str(s).strip()}
    excluded = {str(d) for d in exclude_dirs if str(d).strip()}
    hits: List[Path] = []

    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        if root.suffix.lower() in suffixes:
            return [root]
        return []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excluded]
        for name in filenames:
            if len(hits) >= max_files:
                break
            candidate = Path(dirpath) / name
            if candidate.suffix.lower() not in suffixes:
                continue
            hits.append(candidate)
        if len(hits) >= max_files:
            break
    return hits


@dataclass(frozen=True)
class ExternalCorpusImportConfig:
    source_name: str = "external_corpus"
    include_suffixes: Sequence[str] = _DEFAULT_SUFFIXES
    exclude_dirs: Sequence[str] = _DEFAULT_EXCLUDE_DIRS
    max_files: int = 800
    max_chars_per_file: int = 12_000
    description_chars: int = 2400
    create_group_node: bool = True
    relation_label: str = "contains"


def import_external_corpus(
    directory: str | Path,
    *,
    config: Optional[ExternalCorpusImportConfig] = None,
) -> Dict[str, Any]:
    """Import a directory of documents/code into the knowledge graph.

    Returns stats including processed files and the underlying importer result.
    """

    cfg = config or ExternalCorpusImportConfig()
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(root)

    aligner = get_default_aligner()
    if aligner is None:
        from backend.concept_alignment import ConceptAligner
        from capability.librarian import Librarian

        aligner = ConceptAligner(librarian=Librarian(), entities={})
        set_default_aligner(aligner)

    graph_store = get_graph_store_instance()
    importer = BulkKnowledgeImporter(aligner, graph_store=graph_store)

    files = _iter_files(
        root,
        include_suffixes=cfg.include_suffixes,
        exclude_dirs=cfg.exclude_dirs,
        max_files=max(1, int(cfg.max_files)),
    )

    concepts: Dict[str, RawConcept] = {}
    relations: List[RawRelation] = []

    group_id = f"corpus:{_slugify(cfg.source_name)}"
    if cfg.create_group_node:
        concepts[group_id] = RawConcept(
            id=group_id,
            label=str(cfg.source_name or "external_corpus"),
            description=f"External corpus rooted at {str(root)}",
            metadata={"kind": "external_corpus", "root": str(root)},
        )

    processed = 0
    skipped = 0
    for path in files:
        rel = None
        try:
            rel = path.resolve().relative_to(root.resolve())
        except Exception:
            rel = path.name

        rel_str = str(rel).replace("\\", "/")
        text = _safe_read_text(path, max_chars=max(1, int(cfg.max_chars_per_file)))
        if not text.strip():
            skipped += 1
            continue

        title = _extract_title(path, text)
        stable = _stable_id(cfg.source_name, rel_str)
        doc_id = f"doc:{_slugify(title)}:{stable}"
        description_limit = max(0, int(cfg.description_chars))
        description = text if description_limit <= 0 else text[:description_limit]

        concepts[doc_id] = RawConcept(
            id=doc_id,
            label=title or path.stem,
            description=description,
            metadata={
                "kind": "external_doc",
                "source": str(cfg.source_name),
                "path": str(path),
                "relative_path": rel_str,
                "suffix": path.suffix.lower(),
                "chars": len(text),
            },
        )

        if cfg.create_group_node:
            relations.append(
                RawRelation(
                    source=group_id,
                    relation=str(cfg.relation_label or "related_to"),
                    target=doc_id,
                    weight=1.0,
                    metadata={"corpus_source": str(cfg.source_name), "kind": "corpus_membership"},
                )
            )

        processed += 1

    result = importer.ingest_records(concepts.values(), relations)
    return {
        "root": str(root),
        "source_name": str(cfg.source_name),
        "processed_files": int(processed),
        "skipped_files": int(skipped),
        "concepts_built": int(len(concepts)),
        "relations_built": int(len(relations)),
        "import_result": dict(result or {}),
        "group_id": group_id if cfg.create_group_node else None,
    }


__all__ = ["ExternalCorpusImportConfig", "import_external_corpus"]
