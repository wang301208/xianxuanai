"""Utilities for aligning text embeddings with knowledge graph entities."""
from __future__ import annotations

from functools import lru_cache
from math import sqrt
from typing import Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

from capability.librarian import Librarian
from modules.common import ConceptNode


class ConceptAligner:
    """Align query embeddings to knowledge graph concept nodes."""

    def __init__(
        self,
        librarian: Librarian,
        entities: Dict[str, ConceptNode],
        encoders: Dict[str, str] | None = None,
    ) -> None:
        self.librarian = librarian
        self.entities = entities
        self.encoders = encoders or {}

    @classmethod
    def from_config(
        cls, librarian: Librarian, entities: Dict[str, ConceptNode], config_path: str
    ) -> "ConceptAligner":
        """Construct an aligner from a YAML configuration file."""
        if yaml is None:
            raise RuntimeError("PyYAML is required to load concept aligner config")
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        encoders = data.get("encoders", {})
        return cls(librarian=librarian, entities=entities, encoders=encoders)

    @lru_cache(maxsize=128)
    def _cached_search(
        self, embedding_key: Tuple[float, ...], n_results: int, vector_type: str
    ) -> Tuple[str, ...]:
        return tuple(
            self.librarian.search(
                list(embedding_key),
                n_results=n_results,
                vector_type=vector_type,
                return_content=False,
            )
        )

    def align(
        self, query_embedding: List[float], n_results: int = 5, vector_type: str = "text"
    ) -> List[ConceptNode]:
        """Return concept nodes from the knowledge graph most similar to the query."""
        entity_ids = self._cached_search(tuple(query_embedding), n_results, vector_type)
        results: List[ConceptNode] = []
        for entity_id in entity_ids:
            node = self.entities.get(entity_id)
            if not node:
                continue
            embedding = node.modalities.get(vector_type)
            if embedding is None:
                continue
            similarity = self._cosine_similarity(query_embedding, embedding)
            node.metadata["similarity"] = similarity
            if node.causal_links:
                node.metadata["causal_relations"] = node.causal_links
            results.append(node)
        results.sort(key=lambda n: n.metadata.get("similarity", 0.0), reverse=True)
        return results

    def distill_from_graph(
        self,
        external_entities: Dict[str, ConceptNode],
        *,
        vector_type: str = "text",
        similarity_threshold: float = 0.8,
    ) -> None:
        """Distill knowledge from an external graph.

        External concepts are aligned to the current graph. If the best match
        exceeds ``similarity_threshold`` the node is merged, otherwise it is
        added as a new entity.
        """

        for node in external_entities.values():
            embedding = node.modalities.get(vector_type)
            if not embedding:
                continue
            matches = self.align(embedding, n_results=1, vector_type=vector_type)
            if (
                matches
                and matches[0].metadata.get("similarity", 0.0) >= similarity_threshold
            ):
                existing = self.entities[matches[0].id]
                existing.metadata.update(node.metadata)
                existing.causal_links.extend(node.causal_links)
            else:
                self.entities[node.id] = node

    def transfer_knowledge(
        self,
        nodes: List[ConceptNode],
        *,
        n_results: int = 3,
        vector_type: str = "text",
    ) -> List[ConceptNode]:
        """Enrich provided nodes with related concepts from the knowledge graph."""

        enriched: List[ConceptNode] = []
        for node in nodes:
            embedding = node.modalities.get(vector_type)
            if not embedding:
                enriched.append(node)
                continue
            matches = self.align(embedding, n_results=n_results, vector_type=vector_type)
            node.metadata.setdefault("related_concepts", [])
            node.metadata["related_concepts"].extend([m.id for m in matches])
            for match in matches:
                node.causal_links.extend(match.causal_links)
            enriched.append(node)
        return enriched

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sqrt(sum(x * x for x in a))
        norm_b = sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
