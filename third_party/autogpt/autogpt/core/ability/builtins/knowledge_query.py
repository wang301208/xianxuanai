import json
import logging
from typing import ClassVar, Dict, List, Optional, Tuple

try:  # Optional dependency for high-quality sentence embeddings
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - dependency may be missing
    SentenceTransformer = None  # type: ignore

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema

from backend.concept_alignment import ConceptAligner
from backend.knowledge.registry import (
    get_graph_store_instance,
    require_default_aligner,
)
from modules.common.concepts import ConceptNode


class KnowledgeQuery(Ability):
    """Retrieve related concepts from the knowledge graph for a free-form query."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.KnowledgeQuery",
        ),
        performance_hint=0.2,
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
        *,
        aligner: Optional[ConceptAligner] = None,
    ) -> None:
        self._logger = logger
        self._configuration = configuration
        self._sentence_model: Optional[SentenceTransformer] = None
        self._aligner: Optional[ConceptAligner] = aligner
        self._graph = get_graph_store_instance()

    description: ClassVar[str] = (
        "Search the internal knowledge graph for concepts related to the supplied query."
        " Returns the most relevant concepts, their metadata, and optionally nearby relations."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Free-form text describing the knowledge to retrieve.",
        ),
        "top_k": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="Maximum number of concepts to return.",
            default=5,
        ),
        "vector_type": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Embedding modality to use for similarity scoring (default: text).",
            default="text",
        ),
        "include_metadata": JSONSchema(
            type=JSONSchema.Type.BOOLEAN,
            description="Whether to include concept metadata in the response.",
            default=True,
        ),
        "include_relations": JSONSchema(
            type=JSONSchema.Type.BOOLEAN,
            description="Whether to include outbound relations for each concept.",
            default=False,
        ),
    }

    async def __call__(
        self,
        query: str,
        *,
        top_k: int = 5,
        vector_type: str = "text",
        include_metadata: bool = True,
        include_relations: bool = False,
    ) -> AbilityResult:
        query = (query or "").strip()
        if not query:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query},
                success=False,
                message="Query must be a non-empty string.",
            )

        aligner = self._ensure_aligner()
        if aligner is None:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query},
                success=False,
                message=(
                    "Knowledge aligner is not configured. Initialise KnowledgeConsolidator "
                    "or register a ConceptAligner before querying."
                ),
            )

        embedding = self._encode_query(query)
        if embedding is None:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query},
                success=False,
                message="Failed to encode query into embedding space.",
            )

        ranked = self._rank_concepts(aligner, embedding, vector_type, top_k)
        if not ranked:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"query": query, "top_k": top_k},
                success=True,
                message="No relevant concepts found for the query.",
            )

        results_payload: List[Dict[str, object]] = []
        lines: List[str] = []
        for node, score in ranked:
            entry: Dict[str, object] = {
                "id": node.id,
                "label": node.label,
                "similarity": round(score, 4),
            }
            if include_metadata:
                entry["metadata"] = node.metadata
            if include_relations:
                entry["relations"] = self._collect_relations(node.id)
            results_payload.append(entry)
            lines.append(f"- {node.label} (similarity={score:.3f})")

        content = json.dumps(
            {"query": query, "results": results_payload},
            ensure_ascii=False,
            indent=2,
        )

        new_knowledge = Knowledge(
            content=content,
            content_type=ContentType.TEXT,
            content_metadata={"source": "knowledge_query", "query": query},
        )

        summary = "Relevant concepts:\n" + "\n".join(lines)
        return AbilityResult(
            ability_name=self.name(),
            ability_args={
                "query": query,
                "top_k": top_k,
                "vector_type": vector_type,
                "include_metadata": include_metadata,
                "include_relations": include_relations,
            },
            success=True,
            message=summary,
            new_knowledge=new_knowledge,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_aligner(self) -> Optional[ConceptAligner]:
        if self._aligner is not None:
            return self._aligner
        try:
            self._aligner = require_default_aligner()
        except RuntimeError as err:  # pragma: no cover - defensive
            self._logger.debug("No default aligner configured: %s", err)
            self._aligner = None
        return self._aligner

    def _rank_concepts(
        self,
        aligner: ConceptAligner,
        query_embedding: List[float],
        vector_type: str,
        top_k: int,
    ) -> List[Tuple[ConceptNode, float]]:
        scored: List[Tuple[ConceptNode, float]] = []
        for node in aligner.entities.values():
            embedding = node.modalities.get(vector_type)
            if not embedding:
                continue
            score = aligner._cosine_similarity(query_embedding, embedding)  # type: ignore[attr-defined]
            scored.append((node, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, top_k)]

    def _collect_relations(self, node_id: str) -> List[Dict[str, object]]:
        try:
            snapshot = self._graph.query(node_id=node_id)
        except Exception:  # pragma: no cover - defensive
            return []
        relations: List[Dict[str, object]] = []
        for edge in snapshot.get("edges", []):
            relations.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type.value if hasattr(edge.type, "value") else str(edge.type),
                    "properties": edge.properties,
                }
            )
        return relations

    def _encode_query(self, text: str) -> Optional[List[float]]:
        model = self._get_sentence_model()
        if model is not None:
            try:
                vector = model.encode(text, convert_to_numpy=True)
                return vector.astype(float).tolist()
            except Exception:  # pragma: no cover - fall back if encoding fails
                self._logger.debug("SentenceTransformer encoding failed; using hash fallback.", exc_info=True)
        return self._hash_embedding(text)

    def _get_sentence_model(self) -> Optional[SentenceTransformer]:
        if self._sentence_model is None and SentenceTransformer is not None:
            try:
                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:  # pragma: no cover - optional dependency may be missing
                self._sentence_model = None
        return self._sentence_model

    @staticmethod
    def _hash_embedding(value: str, dimensions: int = 12) -> List[float]:
        digest = KnowledgeQuery._sha256(value)
        chunk = max(1, len(digest) // dimensions)
        embedding: List[float] = []
        for index in range(0, len(digest), chunk):
            piece = digest[index : index + chunk]
            if not piece:
                continue
            normalized = KnowledgeQuery._bytes_to_unit(piece)
            embedding.append(normalized)
            if len(embedding) == dimensions:
                break
        while len(embedding) < dimensions:
            embedding.append(0.0)
        return embedding[:dimensions]

    @staticmethod
    def _sha256(value: str) -> bytes:
        import hashlib

        return hashlib.sha256(value.encode("utf-8")).digest()

    @staticmethod
    def _bytes_to_unit(data: bytes) -> float:
        integer = int.from_bytes(data, byteorder="big", signed=False)
        return integer / float(256 ** len(data) or 1.0)
