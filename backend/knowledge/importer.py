"""Utilities for bulk importing external knowledge graphs.

This module provides :class:`BulkKnowledgeImporter`, a helper that can ingest
knowledge graph descriptions from a variety of common formats (CSV triples,
JSON dictionaries, and OWL/RDF files) and convert them into
:class:`modules.common.concepts.ConceptNode` objects.  Each concept is enriched
with multimodal embeddings when the required feature extractors are available.
Finally the importer distils the new knowledge into an existing
:class:`backend.concept_alignment.ConceptAligner` instance and synchronises the
global knowledge graph store so the imported facts can immediately participate
in downstream reasoning.

The implementation intentionally keeps all optional dependencies lazy and falls
back to deterministic hash-based embeddings when advanced encoders are not
installed.  This allows the importer to be exercised in lightweight test
environments while still supporting richer embeddings in full deployments.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Optional dependency for OWL/RDF ingestion
    import rdflib
except Exception:  # pragma: no cover - dependency may be missing
    rdflib = None  # type: ignore

try:  # Optional dependency for sentence embeddings
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - dependency may be missing
    SentenceTransformer = None  # type: ignore

try:  # Optional dependency for graph embeddings
    import networkx as nx
except Exception:  # pragma: no cover - dependency may be missing
    nx = None  # type: ignore

try:
    from backend.ml.feature_extractor import CLIPFeatureExtractor, GraphFeatureExtractor
except Exception:  # pragma: no cover - optional extras
    CLIPFeatureExtractor = None  # type: ignore
    GraphFeatureExtractor = None  # type: ignore

from backend.concept_alignment import ConceptAligner
from backend.autogpt.autogpt.core.knowledge_graph.graph_store import (
    GraphStore,
    get_graph_store,
)
from backend.autogpt.autogpt.core.knowledge_graph.ontology import EntityType, RelationType
from modules.common.concepts import ConceptNode, ConceptRelation


@dataclass
class RawConcept:
    """Intermediate representation of a concept parsed from external data."""

    id: str
    label: str
    description: str = ""
    image: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RawRelation:
    """Intermediate representation of a relation between two concepts."""

    source: str
    relation: str
    target: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BulkKnowledgeImporter:
    """Bulk import external knowledge bases into a concept graph."""

    def __init__(
        self,
        aligner: ConceptAligner,
        *,
        graph_store: GraphStore | None = None,
        similarity_threshold: float = 0.82,
        vector_type: str = "text",
    ) -> None:
        self.aligner = aligner
        self.graph_store = graph_store or get_graph_store()
        self.similarity_threshold = similarity_threshold
        self.vector_type = vector_type

        self._sentence_model: SentenceTransformer | None = None
        self._clip_extractor: CLIPFeatureExtractor | None = None
        self._graph_extractor: GraphFeatureExtractor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_directory(
        self,
        path: str | Path,
        *,
        nodes_file: str = "nodes.csv",
        relations_file: str = "relations.csv",
    ) -> Dict[str, Any]:
        """Ingest knowledge graph files stored in ``path``.

        ``nodes_file`` should define concept metadata while ``relations_file``
        contains triples mapping concept relationships.  If ``nodes_file`` is
        missing, the importer will synthesise concept stubs from the relations.
        """

        base_path = Path(path)
        if not base_path.exists():
            raise FileNotFoundError(f"{base_path} does not exist")

        node_path = base_path / nodes_file
        rel_path = base_path / relations_file
        nodes, relations = self._load_from_csv(
            node_path if node_path.exists() else None,
            rel_path if rel_path.exists() else None,
        )
        return self._ingest(nodes, relations, base_path=base_path)

    def ingest_file(
        self,
        path: str | Path,
        *,
        fmt: str | None = None,
    ) -> Dict[str, Any]:
        """Ingest a standalone file describing a knowledge graph."""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        format_hint = (fmt or path.suffix.lower().lstrip(".")).lower()
        if format_hint in {"csv"}:
            nodes, relations = self._load_from_csv(None, path)
        elif format_hint in {"json"}:
            nodes, relations = self._load_from_json(path)
        elif format_hint in {"owl", "rdf"}:
            nodes, relations = self._load_from_owl(path)
        else:
            raise ValueError(f"Unsupported import format: {format_hint}")
        return self._ingest(nodes, relations, base_path=path.parent)

    def ingest_records(
        self,
        concepts: Iterable[RawConcept],
        relations: Iterable[RawRelation],
    ) -> Dict[str, Any]:
        """Ingest already parsed concepts and relations."""

        node_map = {concept.id: concept for concept in concepts}
        return self._ingest(node_map, list(relations))

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _load_from_csv(
        self,
        nodes_path: Optional[Path],
        relations_path: Optional[Path],
    ) -> Tuple[Dict[str, RawConcept], List[RawRelation]]:
        nodes: Dict[str, RawConcept] = {}
        if nodes_path is not None:
            with nodes_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    cid = (row.get("id") or row.get("ID") or "").strip()
                    if not cid:
                        continue
                    label = (row.get("label") or row.get("name") or cid).strip()
                    description = (row.get("description") or row.get("comment") or "").strip()
                    image = (row.get("image") or row.get("img") or None) or None
                    metadata = {
                        key: value
                        for key, value in row.items()
                        if key
                        and key not in {"id", "ID", "label", "name", "description", "comment", "image", "img"}
                        and value not in {None, ""}
                    }
                    nodes[cid] = RawConcept(
                        id=cid,
                        label=label,
                        description=description,
                        image=image,
                        metadata=metadata,
                    )

        triples_path = relations_path or nodes_path
        relations: List[RawRelation] = []
        if triples_path is not None:
            with triples_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    src = (
                        row.get("source")
                        or row.get("subject")
                        or row.get("from")
                        or row.get("head")
                        or ""
                    ).strip()
                    tgt = (
                        row.get("target")
                        or row.get("object")
                        or row.get("to")
                        or row.get("tail")
                        or ""
                    ).strip()
                    rel = (row.get("relation") or row.get("predicate") or row.get("type") or "").strip()
                    if not src or not tgt or not rel:
                        continue
                    try:
                        weight = float(row.get("weight") or row.get("confidence") or 1.0)
                    except (TypeError, ValueError):
                        weight = 1.0
                    meta = {
                        key: value
                        for key, value in row.items()
                        if key
                        and key
                        not in {
                            "source",
                            "subject",
                            "from",
                            "head",
                            "target",
                            "object",
                            "to",
                            "tail",
                            "relation",
                            "predicate",
                            "type",
                            "weight",
                            "confidence",
                        }
                        and value not in {None, ""}
                    }
                    relations.append(
                        RawRelation(
                            source=src,
                            relation=rel,
                            target=tgt,
                            weight=weight,
                            metadata=meta,
                        )
                    )

        for relation in relations:
            if relation.source not in nodes:
                nodes[relation.source] = RawConcept(id=relation.source, label=relation.source)
            if relation.target not in nodes:
                nodes[relation.target] = RawConcept(id=relation.target, label=relation.target)

        return nodes, relations

    def _load_from_json(self, path: Path) -> Tuple[Dict[str, RawConcept], List[RawRelation]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes: Dict[str, RawConcept] = {}
        relations: List[RawRelation] = []

        for entry in data if isinstance(data, list) else data.get("nodes", []):
            if not isinstance(entry, dict):
                continue
            cid = str(entry.get("id") or "").strip()
            if not cid:
                continue
            nodes[cid] = RawConcept(
                id=cid,
                label=str(entry.get("label") or entry.get("name") or cid),
                description=str(entry.get("description") or entry.get("summary") or ""),
                image=entry.get("image"),
                metadata={
                    key: value
                    for key, value in entry.items()
                    if key not in {"id", "label", "name", "description", "summary", "image"}
                },
            )

        rel_entries: Iterable[Any]
        if isinstance(data, dict):
            rel_entries = data.get("relations", data.get("edges", []))
        else:
            rel_entries = []

        for entry in rel_entries:
            if not isinstance(entry, dict):
                continue
            src = str(entry.get("source") or entry.get("from") or "").strip()
            tgt = str(entry.get("target") or entry.get("to") or "").strip()
            rel = str(entry.get("relation") or entry.get("type") or "").strip()
            if not src or not tgt or not rel:
                continue
            relations.append(
                RawRelation(
                    source=src,
                    relation=rel,
                    target=tgt,
                    weight=float(entry.get("weight", 1.0)),
                    metadata={
                        key: value
                        for key, value in entry.items()
                        if key not in {"source", "from", "target", "to", "relation", "type", "weight"}
                    },
                )
            )

        for relation in relations:
            if relation.source not in nodes:
                nodes[relation.source] = RawConcept(id=relation.source, label=relation.source)
            if relation.target not in nodes:
                nodes[relation.target] = RawConcept(id=relation.target, label=relation.target)

        return nodes, relations

    def _load_from_owl(self, path: Path) -> Tuple[Dict[str, RawConcept], List[RawRelation]]:
        if rdflib is None:
            raise ImportError("rdflib is required to import OWL/RDF knowledge graphs")

        graph = rdflib.Graph()
        graph.parse(str(path))

        nodes: Dict[str, RawConcept] = {}
        relations: List[RawRelation] = []

        for subject, predicate, obj in graph:
            sid = self._normalise_identifier(subject)
            pid = self._normalise_identifier(predicate)
            if isinstance(obj, rdflib.term.Literal):
                concept = nodes.setdefault(sid, RawConcept(id=sid, label=sid))
                concept.metadata.setdefault("attributes", {})[pid] = str(obj)
                if concept.description == "" and pid.endswith(("comment", "description")):
                    concept.description = str(obj)
                continue
            oid = self._normalise_identifier(obj)
            if not sid or not oid or not pid:
                continue
            relations.append(RawRelation(source=sid, relation=pid, target=oid))
            nodes.setdefault(sid, RawConcept(id=sid, label=sid))
            nodes.setdefault(oid, RawConcept(id=oid, label=oid))

        return nodes, relations

    # ------------------------------------------------------------------
    # Core ingestion logic
    # ------------------------------------------------------------------
    def _ingest(
        self,
        concepts: Dict[str, RawConcept],
        relations: List[RawRelation],
        *,
        base_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if not concepts:
            return {"nodes_added": 0, "relations_added": 0, "concept_relations": []}

        before_ids = set(self.aligner.entities.keys())
        external_entities: Dict[str, ConceptNode] = {}
        mapping: Dict[str, str] = {}
        graph = self._build_network(relations)

        for raw in concepts.values():
            modalities: Dict[str, List[float]] = {}
            text_repr = self._compose_description(raw)
            text_embedding = self._encode_text(text_repr)
            if text_embedding is not None:
                modalities[self.vector_type] = text_embedding
            if raw.image:
                emb = self._encode_image(raw.image, base_path=base_path)
                if emb is not None:
                    modalities["image"] = emb
            if graph is not None and raw.id in graph:
                graph_emb = self._encode_graph(graph, raw.id)
                if graph_emb is not None:
                    modalities["graph"] = graph_emb

            metadata = dict(raw.metadata)
            custom_embeddings = self._extract_custom_embeddings(metadata)
            if raw.description and "description" not in metadata:
                metadata["description"] = raw.description
            metadata.setdefault("sources", [])
            metadata["sources"] = list(metadata.get("sources") or [])
            if base_path is not None:
                metadata.setdefault("import_path", str(base_path))

            concept = ConceptNode(
                id=raw.id,
                label=raw.label or raw.id,
                modalities={**modalities, **custom_embeddings},
                metadata=metadata,
            )
            external_entities[raw.id] = concept
            mapping[raw.id] = self._predict_target_id(concept)

        # Distil concepts into the aligner and update caches/indexes
        self.aligner.distill_from_graph(
            external_entities,
            vector_type=self.vector_type,
            similarity_threshold=self.similarity_threshold,
        )

        if hasattr(self.aligner, "_cached_search"):
            self.aligner._cached_search.cache_clear()  # type: ignore[attr-defined]

        self._update_vector_index(external_entities, mapping)

        added_ids = set(self.aligner.entities.keys()) - before_ids

        concept_relations: List[ConceptRelation] = []
        for raw_id, concept in external_entities.items():
            final_id = self._resolve_final_id(raw_id, concept, mapping)
            final_node = self.aligner.entities.get(final_id)
            if final_node is None:
                continue
            properties = {"label": final_node.label, **final_node.metadata}
            self.graph_store.add_node(final_id, EntityType.CONCEPT, **properties)
            mapping[raw_id] = final_id

        for rel in relations:
            src_id = self._resolve_final_id(rel.source, external_entities.get(rel.source), mapping)
            tgt_id = self._resolve_final_id(rel.target, external_entities.get(rel.target), mapping)
            if src_id not in self.aligner.entities or tgt_id not in self.aligner.entities:
                continue
            relation_enum = self._relation_enum(rel.relation)
            edge_metadata = {"label": rel.relation, **(rel.metadata or {}), "weight": rel.weight}
            self.graph_store.add_edge(src_id, tgt_id, relation_enum, **edge_metadata)
            concept_relations.append(
                ConceptRelation(
                    source=src_id,
                    target=tgt_id,
                    relation=rel.relation,
                    weight=rel.weight,
                    metadata=rel.metadata,
                )
            )

        return {
            "nodes_added": len(added_ids),
            "relations_added": len(concept_relations),
            "concept_relations": concept_relations,
        }

    # ------------------------------------------------------------------
    # Encoder helpers
    # ------------------------------------------------------------------
    def _ensure_sentence_model(self) -> SentenceTransformer | None:
        if self._sentence_model is None and SentenceTransformer is not None:
            try:
                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:  # pragma: no cover - dependency optional
                self._sentence_model = None
        return self._sentence_model

    def _encode_text(self, text: str | None) -> Optional[List[float]]:
        if not text:
            return None
        model = self._ensure_sentence_model()
        if model is not None:
            try:
                vector = model.encode(text, convert_to_numpy=True)
                return vector.astype(float).tolist()
            except Exception:  # pragma: no cover - defensive
                pass
        return self._hash_embedding(text)

    def _encode_image(self, image_path: str, *, base_path: Optional[Path]) -> Optional[List[float]]:
        path = Path(image_path)
        if not path.is_absolute() and base_path is not None:
            path = (base_path / path).resolve()
        if not path.exists():
            return None
        if CLIPFeatureExtractor is None:
            return self._hash_embedding(path.read_bytes().hex())
        if self._clip_extractor is None:
            try:
                self._clip_extractor = CLIPFeatureExtractor()
            except Exception:  # pragma: no cover - dependency optional
                self._clip_extractor = None
                return self._hash_embedding(path.read_bytes().hex())
        assert self._clip_extractor is not None
        from PIL import Image

        with Image.open(path) as img:
            embedding = self._clip_extractor.extract_image_features(img)
        return embedding.detach().cpu().numpy().astype(float).tolist()

    def _encode_graph(self, graph: Any, node_id: str) -> Optional[List[float]]:
        if GraphFeatureExtractor is None or nx is None:
            return None
        if node_id not in graph:
            return None
        if self._graph_extractor is None:
            try:
                self._graph_extractor = GraphFeatureExtractor()
            except Exception:  # pragma: no cover - dependency optional
                self._graph_extractor = None
                return None
        assert self._graph_extractor is not None
        neighbors = [node_id] + list(graph.successors(node_id)) + list(graph.predecessors(node_id))
        subgraph = graph.subgraph(neighbors)
        try:
            embedding = self._graph_extractor.transform([subgraph])[0]
            return embedding.astype(float).tolist()
        except Exception:  # pragma: no cover - graceful degradation
            return None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _build_network(self, relations: Sequence[RawRelation]):
        if nx is None:
            return None
        graph = nx.DiGraph()
        for rel in relations:
            graph.add_edge(rel.source, rel.target)
        return graph

    @staticmethod
    def _compose_description(raw: RawConcept) -> str:
        parts: List[str] = []
        if raw.description:
            parts.append(raw.description)
        for key in ("summary", "comment", "notes"):
            value = raw.metadata.get(key)
            if value:
                parts.append(str(value))
        if not parts:
            parts.append(raw.label or raw.id)
        return " ".join(str(p) for p in parts if p)

    @staticmethod
    def _hash_embedding(value: str, dimensions: int = 12) -> List[float]:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        chunk = max(1, len(digest) // dimensions)
        embedding: List[float] = []
        for index in range(0, len(digest), chunk):
            piece = digest[index : index + chunk]
            if not piece:
                continue
            normalized = int.from_bytes(piece, byteorder="big", signed=False) / float(
                256 ** len(piece)
            )
            embedding.append(normalized)
            if len(embedding) == dimensions:
                break
        while len(embedding) < dimensions:
            embedding.append(0.0)
        return embedding[:dimensions]

    def _predict_target_id(self, concept: ConceptNode) -> str:
        embedding = concept.modalities.get(self.vector_type)
        if embedding is None:
            return concept.id
        matches = self.aligner.align(embedding, n_results=1, vector_type=self.vector_type)
        if matches and matches[0].metadata.get("similarity", 0.0) >= self.similarity_threshold:
            return matches[0].id
        return concept.id

    def _resolve_final_id(
        self,
        original_id: str,
        concept: Optional[ConceptNode],
        mapping: Dict[str, str],
    ) -> str:
        if original_id in mapping:
            resolved = mapping[original_id]
        else:
            resolved = original_id
        if resolved in self.aligner.entities:
            return resolved
        if concept is None:
            return resolved
        embedding = concept.modalities.get(self.vector_type)
        if embedding is None:
            return resolved
        matches = self.aligner.align(embedding, n_results=1, vector_type=self.vector_type)
        if matches:
            return matches[0].id
        return resolved

    def _update_vector_index(
        self,
        concepts: Dict[str, ConceptNode],
        mapping: Dict[str, str],
    ) -> None:
        index = getattr(self.aligner.librarian, "index", None)
        if not index or not hasattr(index, "add"):
            return
        for raw_id, concept in concepts.items():
            target_id = mapping.get(raw_id, raw_id)
            metadata = {"label": concept.label}
            metadata.update(concept.metadata)
            for modality, embedding in concept.modalities.items():
                if embedding is None:
                    continue
                try:
                    index.add(target_id, embedding, metadata, vector_type=modality)
                except Exception:  # pragma: no cover - avoid import failure
                    continue

    def _extract_custom_embeddings(self, metadata: Dict[str, Any]) -> Dict[str, List[float]]:
        embeddings: Dict[str, List[float]] = {}
        single = metadata.pop("embedding", None)
        if single is not None:
            vector = self._coerce_embedding(single)
            if vector:
                vector_type = str(metadata.pop("embedding_type", metadata.get("modality") or self.vector_type))
                embeddings[vector_type] = vector
        multiple = metadata.pop("embeddings", None)
        if isinstance(multiple, dict):
            for key, value in multiple.items():
                vector = self._coerce_embedding(value)
                if vector:
                    embeddings[str(key)] = vector
        return embeddings

    @staticmethod
    def _coerce_embedding(value: Any) -> List[float] | None:
        if isinstance(value, (list, tuple)):
            try:
                return [float(v) for v in value]
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _normalise_identifier(term: Any) -> str:
        value = str(term)
        if "#" in value:
            value = value.rsplit("#", 1)[-1]
        elif "/" in value:
            value = value.rstrip("/").rsplit("/", 1)[-1]
        return value

    @staticmethod
    def _relation_enum(relation: str) -> RelationType:
        cleaned = relation.strip().upper().replace(" ", "_")
        return RelationType.__members__.get(cleaned, RelationType.RELATED_TO)
