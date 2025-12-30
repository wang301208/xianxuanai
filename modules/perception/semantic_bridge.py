"""Semantic decoding and knowledge integration for multimodal perception."""

from __future__ import annotations

import asyncio
import logging
import math
import base64
import io
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # optional dependency for image handling
    from PIL import Image
except Exception:  # pragma: no cover - pillow may be absent in minimal installs
    Image = None  # type: ignore[assignment]

try:  # optional dependency for tensor conversion
    import torch
except Exception:  # pragma: no cover - torch may be absent when clip unavailable
    torch = None  # type: ignore[assignment]

try:  # optional vision dependency
    from backend.ml.feature_extractor import CLIPFeatureExtractor
except ImportError:  # pragma: no cover - fallback when vision stack absent
    CLIPFeatureExtractor = None  # type: ignore[assignment]

try:  # optional knowledge stack dependency
    from backend.knowledge.registry import require_default_aligner
except Exception:  # pragma: no cover - fallback for minimal environments
    def require_default_aligner() -> Any:  # type: ignore
        raise RuntimeError("Concept aligner unavailable")

from modules.memory import (
    ExperiencePayload,
    TaskMemoryManager,
    VectorMemoryStore,
)

try:  # optional knowledge stack dependency
    from modules.knowledge import KnowledgeFact, RuntimeKnowledgeImporter
except Exception:  # pragma: no cover - fallback for minimal environments
    @dataclass
    class KnowledgeFact:  # type: ignore[override]
        subject: str
        predicate: str
        obj: str
        subject_id: Optional[str] = None
        object_id: Optional[str] = None
        subject_description: Optional[str] = None
        object_description: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)
        confidence: Optional[float] = None
        source: Optional[str] = None
        context: Optional[str] = None
        timestamp: Optional[float] = None

    class RuntimeKnowledgeImporter:  # type: ignore[override]
        def ingest_facts(self, _facts: Iterable[KnowledgeFact]) -> Dict[str, Any]:
            raise RuntimeError("Knowledge importer unavailable")

try:  # optional multimodal fusion dependency
    from modules.brain.multimodal import MultimodalFusionEngine
except Exception:  # pragma: no cover - fusion stack may be optional
    MultimodalFusionEngine = None  # type: ignore[assignment]

try:  # optional self-supervision dependency
    from modules.perception.self_supervised import SelfSupervisedPerceptionLearner
except Exception:  # pragma: no cover - learner is optional
    SelfSupervisedPerceptionLearner = None  # type: ignore[assignment]

try:  # optional learning stack dependency
    from modules.learning import EpisodeRecord, ExperienceHub
except Exception:  # pragma: no cover - fallback for minimal environments
    @dataclass
    class EpisodeRecord:  # type: ignore[override]
        task_id: str
        policy_version: str
        total_reward: float
        steps: int
        success: bool
        metadata: Dict[str, Any] = field(default_factory=dict)

    class ExperienceHub:  # type: ignore[override]
        def __init__(self, root: Path) -> None:
            self.root = root

        def append(self, _record: EpisodeRecord) -> None:
            return None

LOGGER = logging.getLogger(__name__)


def _normalise(vector: Sequence[float], *, limit: int = 128) -> List[float]:
    arr = np.asarray(list(vector), dtype=np.float32)
    if arr.size == 0:
        return []
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    if limit and arr.size > limit:
        arr = arr[:limit]
    return arr.astype(float).tolist()


def _hash_labels(values: Iterable[str]) -> float:
    score = 0
    for value in values:
        score = (score * 33 + sum(ord(c) for c in value)) % 9973
    return float(score) / 9973.0


ASRInput = Union[str, bytes, Sequence[float], np.ndarray]
ASRTranscriber = Callable[[ASRInput, Dict[str, Any]], str]


@dataclass
class ASRConfig:
    """Configuration for optional automatic speech recognition."""

    enabled: bool = True
    provider: Optional[str] = "autogpt"
    model: Optional[str] = "whisper-1"
    agent: Any | None = None
    transcriber: ASRTranscriber | None = None


@dataclass
class SemanticBridgeOutput:
    semantic_annotations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    knowledge_facts: List[Dict[str, Any]] = field(default_factory=list)
    fused_embedding: List[float] | None = None
    modality_embeddings: Dict[str, List[float]] = field(default_factory=dict)


class SemanticBridge:
    """Decode sensory embeddings into semantic summaries and knowledge facts."""

    def __init__(
        self,
        *,
        task_memory: TaskMemoryManager | None = None,
        knowledge_importer: RuntimeKnowledgeImporter | None = None,
        experience_hub: ExperienceHub | None = None,
        storage_root: Path | None = None,
        asr_config: ASRConfig | None = None,
        clip_config: Dict[str, Any] | None = None,
        self_supervised_config: Dict[str, Any] | None = None,
    ) -> None:
        root = storage_root or Path("data/perception")
        root.mkdir(parents=True, exist_ok=True)

        memory_store = VectorMemoryStore(root / "vector_store")
        self._task_memory = task_memory or TaskMemoryManager(memory_store)

        self._clip_extractor: Any | None = None
        self._clip_available = False
        clip_options = clip_config or {}
        self._clip_prompt_table: List[Tuple[str, str]] = clip_options.get(
            "prompts",
            [
                ("person", "a photo focused on a person"),
                ("group", "a photo of multiple people"),
                ("nature", "a photo of a natural landscape"),
                ("indoor", "an indoor room scene"),
                ("document", "a screenshot or document page"),
                ("diagram", "a diagram or chart"),
                ("food", "a close-up photo of food"),
                ("object", "a detailed photo of an object"),
                ("city", "an outdoor urban city scene"),
                ("night", "a night scene"),
            ],
        )
        self._clip_prompt_embeddings: List[Tuple[str, np.ndarray]] = []
        self._clip_prompt_lookup: Dict[str, str] = {label: prompt for label, prompt in self._clip_prompt_table}
        self._clip_prompt_top_k = int(clip_options.get("top_k", 3))
        if CLIPFeatureExtractor is not None:
            try:
                self._clip_extractor = CLIPFeatureExtractor(**clip_options)
            except Exception:  # pragma: no cover - optional dependency
                LOGGER.warning(
                    "Failed to initialize CLIP feature extractor; vision features disabled.",
                    exc_info=True,
                )
                self._clip_extractor = None
            else:
                self._clip_available = True
                self._prepare_clip_prompt_embeddings()
        else:
            LOGGER.debug("CLIP feature extractor unavailable: backend not installed.")

        try:
            self._importer = knowledge_importer or RuntimeKnowledgeImporter()
        except Exception:  # pragma: no cover - consolidation may be unavailable in tests
            self._importer = None

        hub_path = root / "experience"
        hub_path.mkdir(parents=True, exist_ok=True)
        self._experience_hub = experience_hub or ExperienceHub(hub_path)
        try:
            self._aligner = require_default_aligner()
        except RuntimeError:
            self._aligner = None

        self._asr_config = asr_config or ASRConfig()
        self._asr_transcriber = self._prepare_asr_transcriber()
        self._fusion_engine = MultimodalFusionEngine() if MultimodalFusionEngine is not None else None
        self._self_supervised = None
        if SelfSupervisedPerceptionLearner is not None:
            try:
                options = self_supervised_config or {}
                self._self_supervised = SelfSupervisedPerceptionLearner(
                    fusion_engine=self._fusion_engine,
                    **options,
                )
            except Exception:  # pragma: no cover - optional learner
                LOGGER.debug("Failed to initialize self-supervised perception learner.", exc_info=True)
                self._self_supervised = None

    # ------------------------------------------------------------------
    def process(
        self,
        perception_snapshot: Any,
        *,
        agent_id: str | None = None,
        cycle_index: int | None = None,
        ingest: bool = True,
    ) -> SemanticBridgeOutput:
        annotations: Dict[str, Dict[str, Any]] = {}
        knowledge_facts: List[KnowledgeFact] = []
        memory_entries: List[ExperiencePayload] = []
        modality_vectors: Dict[str, Dict[str, Any]] = {}

        modalities = getattr(perception_snapshot, "modalities", {}) or {}
        for modality, payload in modalities.items():
            if modality in {"vision", "visual", "image"}:
                result = self._process_visual(payload, modality, agent_id, cycle_index)
            elif modality in {"audio", "auditory", "sound"}:
                result = self._process_audio(payload, modality, agent_id, cycle_index)
            elif modality in {"text", "transcript"}:
                result = self._process_text(payload, modality, agent_id, cycle_index)
            else:
                result = None

            if not result:
                continue

            facts = result.get("facts") or []
            knowledge_facts.extend(facts)
            annotation = result.get("annotation")
            if annotation:
                annotations[modality] = annotation
                primary_values = annotation.get("embedding")
                primary_vector = self._coerce_embedding(primary_values)
                if primary_vector is not None:
                    payload_entry: Dict[str, Any] = {
                        "embedding": primary_vector,
                        "primary_embedding": primary_vector,
                        "confidence": self._max_confidence(facts),
                    }

                    legacy_values = annotation.get("heuristic_embedding") or annotation.get("legacy_embedding")
                    legacy_vector = self._coerce_embedding(legacy_values)
                    if legacy_vector is not None and not np.array_equal(legacy_vector, primary_vector):
                        payload_entry["legacy_embedding"] = legacy_vector

                    clip_info = annotation.get("clip")
                    if isinstance(clip_info, dict):
                        clip_values = clip_info.get("embedding")
                        clip_vector = self._coerce_embedding(clip_values)
                        if clip_vector is not None:
                            payload_entry["clip_embedding"] = clip_vector

                    modality_vectors[modality] = payload_entry
            memory_entry = result.get("memory")
            if memory_entry:
                memory_entries.append(memory_entry)

        fused_embedding: List[float] | None = None
        modality_embedding_lists: Dict[str, List[float]] = {}
        if modality_vectors:
            for name, payload in modality_vectors.items():
                modality_embedding_lists[name] = payload["embedding"].astype(float).tolist()
            fused_embedding = self._fuse_modalities(modality_vectors)
            if fused_embedding:
                multimodal_annotation = annotations.setdefault("multimodal", {})
                multimodal_annotation["embedding"] = fused_embedding
                multimodal_annotation["modalities"] = sorted(modality_vectors.keys())
                confidence_hint = max((payload.get("confidence") or 0.0) for payload in modality_vectors.values())
                if confidence_hint:
                    multimodal_annotation["confidence"] = float(confidence_hint)
                knowledge_facts.append(
                    KnowledgeFact(
                        subject=self._build_subject(agent_id, cycle_index, "multimodal"),
                        predicate="hasEmbedding",
                        obj="multimodal",
                        source="perception.multimodal",
                        metadata={
                            "modality": "multimodal",
                            "embedding": list(fused_embedding),
                            "modalities": sorted(modality_vectors.keys()),
                        },
                        confidence=min(1.0, 0.65 + 0.05 * len(modality_vectors)),
                    )
                )

        if self._self_supervised and modality_vectors:
            try:
                ss_result = self._self_supervised.observe(
                    modality_vectors,
                    fused_embedding=fused_embedding,
                    annotations=annotations,
                )
            except Exception:  # pragma: no cover - learner is optional
                LOGGER.debug("Self-supervised perceptual learner step failed.", exc_info=True)
            else:
                if ss_result:
                    self._apply_self_supervision(
                        ss_result,
                        annotations=annotations,
                        knowledge_facts=knowledge_facts,
                        modality_vectors=modality_vectors,
                        agent_id=agent_id,
                        cycle_index=cycle_index,
                    )
                    concept = ss_result.get("concept")
                    if concept and fused_embedding is None:
                        try:
                            fused_embedding = list(concept.embedding) if concept.embedding else None
                        except Exception:
                            fused_embedding = None

        if ingest and knowledge_facts and self._importer:
            try:
                self._importer.ingest_facts(knowledge_facts)
            except Exception:  # pragma: no cover - defensive guard
                self._importer = None

        if ingest:
            for entry in memory_entries:
                try:
                    self._task_memory.store_experience(entry)
                except Exception:  # pragma: no cover - defensive guard
                    pass

        if ingest and annotations and self._experience_hub:
            summary = "; ".join(
                f"{mod}:{', '.join(data.get('labels', []))}" for mod, data in annotations.items()
            )
            try:
                self._experience_hub.append(
                    EpisodeRecord(
                        task_id=f"perception:{agent_id or 'agent'}",
                        policy_version="semantic_bridge",
                        total_reward=0.0,
                        steps=len(annotations),
                        success=True,
                        metadata={"summary": summary, "cycle": cycle_index},
                    )
                )
            except Exception:  # pragma: no cover - defensive guard
                pass

        return SemanticBridgeOutput(
            semantic_annotations=annotations,
            knowledge_facts=[asdict(fact) for fact in knowledge_facts],
            fused_embedding=fused_embedding,
            modality_embeddings=modality_embedding_lists,
        )

    async def process_async(
        self,
        perception_snapshot: Any,
        *,
        agent_id: str | None = None,
        cycle_index: int | None = None,
        ingest: bool = True,
        timeout: float | None = 5.0,
        executor: Executor | None = None,
    ) -> SemanticBridgeOutput:
        """Asynchronously execute :meth:`process` in a background executor.

        The perception pipeline can involve heavy numerical operations. Running
        it in a thread pool prevents the event loop from being blocked during
        concurrent planning.
        """

        loop = asyncio.get_running_loop()
        bound = (
            perception_snapshot,
            agent_id,
            cycle_index,
            ingest,
        )

        def _call() -> SemanticBridgeOutput:
            return self.process(
                bound[0],
                agent_id=bound[1],
                cycle_index=bound[2],
                ingest=bound[3],
            )

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(executor, _call),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return SemanticBridgeOutput()

    # ------------------------------------------------------------------
    def _fuse_modalities(
        self,
        modality_vectors: Dict[str, Dict[str, Any]],
    ) -> List[float] | None:
        if self._fusion_engine is None or not modality_vectors:
            return None
        try:
            fusion_inputs = {}
            for name, payload in modality_vectors.items():
                primary = payload.get("primary_embedding") or payload.get("embedding")
                if primary is None:
                    continue
                fusion_payload: Dict[str, Any] = {"embedding": primary}
                confidence = payload.get("confidence")
                if confidence is not None:
                    fusion_payload["confidence"] = confidence

                metadata: Dict[str, Any] = {}
                for key in ("clip_embedding", "legacy_embedding", "heuristic_embedding"):
                    if key in payload and payload[key] is not None:
                        metadata[key] = payload[key]
                if metadata:
                    fusion_payload["metadata"] = metadata

                fusion_inputs[name] = fusion_payload
        except Exception:  # pragma: no cover - defensive guard
            return None
        if not fusion_inputs:
            return None
        try:
            fused = self._fusion_engine.fuse_sensory_modalities(**fusion_inputs)
        except Exception:  # pragma: no cover - fusion is optional
            LOGGER.debug(
                "Multimodal fusion failed during semantic bridge processing.",
                exc_info=True,
            )
            return None
        vector = np.asarray(fused, dtype=float)
        if vector.ndim == 0:
            vector = vector.reshape(1)
        if vector.size == 0:
            return None
        return _normalise(vector, limit=min(128, vector.size))

    def _apply_self_supervision(
        self,
        result: Dict[str, Any],
        *,
        annotations: Dict[str, Dict[str, Any]],
        knowledge_facts: List[KnowledgeFact],
        modality_vectors: Dict[str, Dict[str, Any]],
        agent_id: Optional[str],
        cycle_index: Optional[int],
    ) -> None:
        predictions = result.get("predictions") or {}
        for modality, pred in predictions.items():
            if pred is None:
                continue
            target = annotations.setdefault(modality, {})
            bucket = target.setdefault("self_supervised", {})
            instant = getattr(pred, "instant_error", None)
            ema = getattr(pred, "ema_error", None)
            if instant is not None:
                bucket["prediction_error"] = float(instant)
            if ema is not None:
                bucket["prediction_error_ema"] = float(ema)

        contrastive = result.get("contrastive")
        if contrastive:
            target = annotations.setdefault("multimodal", {})
            ss_bucket = target.setdefault("self_supervised", {})
            avg_loss = getattr(contrastive, "average_loss", None)
            if avg_loss is not None:
                ss_bucket["contrastive_loss"] = float(avg_loss)
            pair_losses = getattr(contrastive, "pair_losses", None)
            if pair_losses:
                ss_bucket["pair_losses"] = {f"{a}->{b}": float(val) for (a, b), val in pair_losses.items()}
            positives = getattr(contrastive, "positives", None)
            if positives:
                ss_bucket["positive_similarity"] = {f"{a}->{b}": float(val) for (a, b), val in positives.items()}

        concept = result.get("concept")
        if concept:
            concept_key = getattr(concept, "key", None)
            concept_embedding = getattr(concept, "embedding", None)
            concept_alignment = getattr(concept, "alignment", None)
            target = annotations.setdefault("multimodal", {})
            if concept_key:
                target["concept_key"] = concept_key
            if concept_alignment is not None:
                target["concept_alignment"] = float(concept_alignment)
            if concept_embedding:
                try:
                    target["concept_embedding"] = list(concept_embedding)
                except Exception:
                    pass
            if concept_key and concept_embedding is not None:
                try:
                    confidence = min(1.0, 0.55 + 0.05 * len(modality_vectors))
                except Exception:
                    confidence = 0.55
                try:
                    knowledge_facts.append(
                        KnowledgeFact(
                            subject=self._build_subject(agent_id, cycle_index, f"concept:{concept_key}"),
                            predicate="hasConceptEmbedding",
                            obj=concept_key,
                            source="perception.self_supervised",
                            metadata={
                                "embedding": list(concept_embedding),
                                "alignment": float(concept_alignment or 0.0),
                                "modalities": sorted(modality_vectors.keys()),
                            },
                            confidence=confidence,
                        )
                    )
                except Exception:  # pragma: no cover - defensive guard
                    LOGGER.debug("Failed to append concept knowledge fact.", exc_info=True)

    @staticmethod
    def _coerce_embedding(values: Any) -> Optional[np.ndarray]:
        if values is None:
            return None
        try:
            array = np.asarray(values, dtype=float)
        except Exception:  # pragma: no cover - guard against malformed inputs
            return None
        if array.size == 0:
            return None
        if array.ndim > 1:
            array = array.reshape(-1)
        return array.astype(float)

    @staticmethod
    def _max_confidence(facts: Sequence[KnowledgeFact]) -> float:
        best = 0.0
        for fact in facts:
            confidence = getattr(fact, "confidence", None)
            if confidence is None:
                continue
            try:
                best = max(best, float(confidence))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
        return best

    # ------------------------------------------------------------------
    def _process_visual(
        self,
        payload: Dict[str, Any],
        modality: str,
        agent_id: Optional[str],
        cycle_index: Optional[int],
    ) -> Dict[str, Any] | None:
        vector = payload.get("vector") or []
        features = payload.get("features") or {}
        metadata = payload.get("metadata") or {}
        if not vector and not features:
            return None

        clip_embedding: List[float] | None = None
        clip_labels_clip: List[str] = []
        clip_caption: Optional[str] = None
        clip_label_scores: List[Dict[str, float]] = []
        clip_image_source: Optional[str] = None
        if self._clip_available and self._clip_extractor is not None:
            image, clip_image_source = self._resolve_clip_image(payload, metadata)
            if image is not None:
                try:
                    (
                        clip_embedding,
                        clip_labels_clip,
                        clip_caption,
                        clip_label_scores,
                    ) = self._run_clip_analysis(image)
                except Exception:  # pragma: no cover - guard against optional dependency failures
                    LOGGER.warning("CLIP analysis failed; using heuristic vision labels only.", exc_info=True)
                    clip_embedding = None
                    clip_labels_clip = []
                    clip_caption = None
                    clip_label_scores = []
            elif metadata:
                LOGGER.debug("CLIP requested but no image payload available; retaining heuristic labels.")

        edge_energy = float(features.get("edge_energy", 0.0))
        contrast = float(features.get("contrast", 0.0))
        orientation_strength = float(features.get("orientation_strength", 0.0))
        orientation_entropy = float(features.get("orientation_entropy", 0.0))

        labels: List[str] = []
        if edge_energy > 0.4:
            labels.append("high_edge_detail")
        elif edge_energy < 0.15:
            labels.append("smooth_surface")

        if contrast >= 0.35:
            labels.append("high_contrast")
        elif contrast <= 0.15:
            labels.append("low_contrast")

        if orientation_strength > 0.3:
            dominant = metadata.get("dominant_orientation")
            if dominant is None:
                histogram = metadata.get("orientation_histogram") or []
                angles = metadata.get("orientation_angles") or []
                if histogram and angles:
                    index = int(np.argmax(histogram))
                    dominant = angles[index] if index < len(angles) else None
            if dominant is not None:
                labels.append(f"oriented_{int(dominant)}deg")

        if orientation_entropy < 0.6:
            labels.append("structured_scene")
        elif orientation_entropy > 1.2:
            labels.append("chaotic_scene")

        if not labels:
            labels.append("generic_scene")

        if clip_labels_clip:
            for label in clip_labels_clip:
                if label not in labels:
                    labels.append(label)

        subject = self._build_subject(agent_id, cycle_index, "image")

        heuristic_embedding = _normalise(vector, limit=128)
        clip_embedding_normalised: List[float] = []
        if clip_embedding is not None:
            clip_embedding_normalised = _normalise(clip_embedding, limit=512)

        primary_embedding = clip_embedding_normalised or heuristic_embedding
        embedding_source = "clip" if clip_embedding_normalised else "heuristic"

        caption_tokens = []
        if "structured_scene" in labels:
            caption_tokens.append("structured layout")
        if "chaotic_scene" in labels:
            caption_tokens.append("complex texture")
        if "high_contrast" in labels:
            caption_tokens.append("high contrast")
        if "low_contrast" in labels:
            caption_tokens.append("soft lighting")
        if "high_edge_detail" in labels:
            caption_tokens.append("rich detail")

        summary = (
            "Visual scene features "
            + ", ".join(caption_tokens or ["mixed textures"])
            + f" (edge_energy={edge_energy:.2f}, contrast={contrast:.2f})."
        )

        clip_fact_metadata: Dict[str, Any] = {}
        if clip_embedding_normalised:
            clip_fact_metadata["clip_embedding"] = clip_embedding_normalised
        if clip_labels_clip:
            clip_fact_metadata["clip_labels"] = clip_labels_clip
        if clip_caption:
            clip_fact_metadata["clip_caption"] = clip_caption
        if clip_label_scores:
            clip_fact_metadata["clip_label_scores"] = clip_label_scores
        if clip_image_source:
            clip_fact_metadata["clip_image_source"] = clip_image_source

        if clip_caption:
            caption_sentence = clip_caption.rstrip()
            if caption_sentence.endswith(('.', '!', '?')):
                summary += f" CLIP insight: {caption_sentence}"
            else:
                summary += f" CLIP insight: {caption_sentence}."

        facts = [
            KnowledgeFact(
                subject=subject,
                predicate="hasLabel",
                obj=label,
                source="perception.vision",
                metadata={
                    "modality": "vision",
                    "edge_energy": edge_energy,
                    "contrast": contrast,
                    "orientation_entropy": orientation_entropy,
                    "embedding": primary_embedding,
                    "embedding_source": embedding_source,
                },
                confidence=min(1.0, 0.6 + 0.1 * len(labels)),
            )
            for label in labels
        ]
        facts.append(
            KnowledgeFact(
                subject=subject,
                predicate="hasDescription",
                obj=summary,
                source="perception.vision",
                metadata={"modality": "vision"},
                confidence=0.7,
            )
        )

        related = self._find_related_concepts(primary_embedding, "image")
        annotation = {
            "labels": labels,
            "summary": summary,
            "embedding": primary_embedding,
            "metrics": {
                "edge_energy": edge_energy,
                "contrast": contrast,
                "orientation_entropy": orientation_entropy,
            },
            "related_concepts": related,
        }
        annotation["embedding_source"] = embedding_source
        if heuristic_embedding:
            annotation["heuristic_embedding"] = heuristic_embedding

        if clip_fact_metadata:
            clip_annotation: Dict[str, Any] = {
                "labels": clip_labels_clip,
                "scores": clip_label_scores,
            }
            if clip_caption:
                clip_annotation["caption"] = clip_caption
            if clip_embedding_normalised:
                clip_annotation["embedding"] = clip_embedding_normalised
            if clip_image_source:
                clip_annotation["source"] = clip_image_source
            annotation["clip"] = clip_annotation

        if related:
            for concept in related:
                facts.append(
                    KnowledgeFact(
                        subject=subject,
                        predicate="similarTo",
                        obj=concept["id"],
                        source="perception.vision",
                        metadata={
                            "modality": "vision",
                            "related_label": concept["label"],
                        },
                        confidence=concept["similarity"],
                    )
                )

        if clip_fact_metadata:
            for fact in facts:
                fact.metadata.update(
                    {key: value for key, value in clip_fact_metadata.items() if value is not None and value != []}
                )
        for fact in facts:
            fact.metadata.setdefault("embedding_source", embedding_source)
            if heuristic_embedding and embedding_source == "clip":
                fact.metadata.setdefault("legacy_embedding", heuristic_embedding)

        memory_metadata: Dict[str, Any] = {
            "modality": "vision",
            "labels": labels,
            "cycle": cycle_index,
            "subject": subject,
            "embedding_source": embedding_source,
            "embedding": primary_embedding,
        }
        if heuristic_embedding and embedding_source == "clip":
            memory_metadata["legacy_embedding"] = heuristic_embedding
        if clip_fact_metadata:
            for key, value in clip_fact_metadata.items():
                if value is not None and value != []:
                    memory_metadata[key] = value

        memory_entry = ExperiencePayload(
            task_id=self._build_memory_task(agent_id),
            summary=f"Vision {subject}: {summary}",
            messages=[
                {"role": "system", "content": "Semantic visual observation"},
                {"role": "observation", "content": summary},
            ],
            metadata=memory_metadata,
        )

        return {"annotation": annotation, "facts": facts, "memory": memory_entry}

    def _process_text(
        self,
        payload: Dict[str, Any] | str,
        modality: str,
        agent_id: Optional[str],
        cycle_index: Optional[int],
    ) -> Dict[str, Any] | None:
        text_content: str = ""
        metadata: Dict[str, Any] = {}
        confidence_value: Optional[float] = None

        if isinstance(payload, str):
            text_content = payload
        elif isinstance(payload, dict):
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            confidence_value = payload.get("confidence") if payload.get("confidence") is not None else metadata.get("confidence")
            candidates = [
                payload.get("text"),
                payload.get("transcript"),
                payload.get("content"),
                payload.get("value"),
                metadata.get("text") if isinstance(metadata, dict) else None,
                metadata.get("transcript") if isinstance(metadata, dict) else None,
            ]
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip():
                    text_content = candidate
                    break
            if not text_content and isinstance(payload.get("tokens"), Sequence):
                text_content = " ".join(str(token) for token in payload["tokens"] if isinstance(token, str))
        elif isinstance(payload, Sequence):
            text_content = " ".join(str(item) for item in payload if isinstance(item, str))

        normalised_text = " ".join(str(text_content or "").split())
        if not normalised_text:
            return None

        try:
            confidence = float(confidence_value) if confidence_value is not None else 0.7
        except (TypeError, ValueError):
            confidence = 0.7
        if not math.isfinite(confidence):
            confidence = 0.7
        confidence = min(max(confidence, 0.0), 1.0)

        embedding, embedding_source, tokens = self._encode_text_embedding(normalised_text)
        if not embedding:
            return None

        subject = self._build_subject(agent_id, cycle_index, modality)
        summary = metadata.get("summary") if isinstance(metadata, dict) else None
        if not summary:
            if isinstance(payload, dict):
                summary = payload.get("summary")
        summary_text = self._summarise_text(summary or normalised_text)

        related = self._find_related_concepts(embedding, "text")

        annotation: Dict[str, Any] = {
            "text": normalised_text,
            "summary": summary_text,
            "embedding": embedding,
            "embedding_source": embedding_source,
            "confidence": confidence,
        }
        if tokens:
            annotation["tokens"] = tokens
        if related:
            annotation["related_concepts"] = related

        primary_predicate = "hasTranscript" if "transcript" in modality.lower() else "hasText"
        facts = [
            KnowledgeFact(
                subject=subject,
                predicate=primary_predicate,
                obj=normalised_text,
                source=f"perception.{modality}",
                metadata={
                    "modality": modality,
                    "embedding": embedding,
                    "embedding_source": embedding_source,
                    "confidence": confidence,
                },
                confidence=min(1.0, 0.55 + 0.05 * min(len(tokens), 10) if tokens else 0.65),
            ),
            KnowledgeFact(
                subject=subject,
                predicate="hasSummary",
                obj=summary_text,
                source=f"perception.{modality}",
                metadata={
                    "modality": modality,
                    "text": normalised_text,
                    "embedding_source": embedding_source,
                },
                confidence=0.6,
            ),
        ]

        if related:
            for concept in related:
                facts.append(
                    KnowledgeFact(
                        subject=subject,
                        predicate="similarTo",
                        obj=concept["id"],
                        source=f"perception.{modality}",
                        metadata={
                            "modality": modality,
                            "related_label": concept.get("label"),
                        },
                        confidence=concept.get("similarity", 0.0),
                    )
                )

        memory_metadata: Dict[str, Any] = {
            "modality": modality,
            "subject": subject,
            "cycle": cycle_index,
            "confidence": confidence,
            "embedding": embedding,
            "embedding_source": embedding_source,
        }
        if related:
            memory_metadata["related_concepts"] = related

        memory_entry = ExperiencePayload(
            task_id=self._build_memory_task(agent_id),
            summary=f"Text {subject}: {summary_text}",
            messages=[
                {"role": "system", "content": "Semantic text observation"},
                {"role": "observation", "content": summary_text},
                {"role": modality, "content": normalised_text},
            ],
            metadata=memory_metadata,
        )

        annotation["confidence"] = confidence

        return {"annotation": annotation, "facts": facts, "memory": memory_entry}

    def _process_audio(
        self,
        payload: Dict[str, Any],
        modality: str,
        agent_id: Optional[str],
        cycle_index: Optional[int],
    ) -> Dict[str, Any] | None:
        vector = payload.get("vector") or []
        features = payload.get("features") or {}
        if not vector and not features:
            return None

        energy = float(features.get("energy", 0.0))
        spectral_centroid = float(features.get("spectral_centroid", 0.0))
        spectral_flux = float(features.get("spectral_flux", 0.0))
        modulation = float(features.get("temporal_modulation", 0.0))

        labels: List[str] = []
        if energy > 0.4:
            labels.append("loud_audio")
        elif energy < 0.15:
            labels.append("quiet_audio")

        if spectral_flux > 0.25:
            labels.append("dynamic_audio")
        else:
            labels.append("steady_audio")

        if modulation > 0.05:
            labels.append("rhythmic_pattern")

        tonal_description = "mid"
        if spectral_centroid > 4000:
            tonal_description = "bright"
            labels.append("high_timbre")
        elif spectral_centroid < 1500:
            tonal_description = "warm"
            labels.append("low_timbre")

        subject = self._build_subject(agent_id, cycle_index, "audio")
        embedding = _normalise(vector, limit=96)

        transcript, transcript_info = self._resolve_transcript(payload)
        transcription_status = "success" if transcript else transcript_info.get("error") and "error" or transcript_info.get("reason") or ("disabled" if not self._asr_config.enabled else "unavailable")

        summary = (
            "Audio segment is "
            + ", ".join(labels[:2])
            + f" with {tonal_description} timbre "
            + f"(energy={energy:.2f}, flux={spectral_flux:.2f})."
        )
        if transcript:
            summary += f" Transcript: \"{transcript}\"."

        facts = [
            KnowledgeFact(
                subject=subject,
                predicate="hasLabel",
                obj=label,
                source="perception.audio",
                metadata={
                    "modality": "audio",
                    "energy": energy,
                    "spectral_flux": spectral_flux,
                    "temporal_modulation": modulation,
                    "embedding": embedding,
                },
                confidence=min(1.0, 0.55 + 0.1 * len(labels)),
            )
            for label in labels
        ]
        facts.append(
            KnowledgeFact(
                subject=subject,
                predicate="hasDescription",
                obj=summary,
                source="perception.audio",
                metadata={"modality": "audio"},
                confidence=0.65,
            )
        )

        if transcript:
            facts.append(
                KnowledgeFact(
                    subject=subject,
                    predicate="hasTranscript",
                    obj=transcript,
                    source="perception.audio",
                    metadata={
                        "modality": "audio",
                        "transcript_source": transcript_info.get("source", "asr"),
                        "energy": energy,
                        "spectral_flux": spectral_flux,
                        "temporal_modulation": modulation,
                    },
                    confidence=0.8,
                )
            )

        emotion_weight = _hash_labels(labels)
        facts.append(
            KnowledgeFact(
                subject=subject,
                predicate="suggestsAffect",
                obj="energetic" if emotion_weight > 0.5 else "calm",
                source="perception.audio",
                metadata={
                    "modality": "audio",
                    "affect_score": emotion_weight,
                },
                confidence=0.5 + 0.4 * emotion_weight,
            )
        )

        related = self._find_related_concepts(embedding, "audio")
        annotation = {
            "labels": labels,
            "summary": summary,
            "embedding": embedding,
            "metrics": {
                "energy": energy,
                "spectral_flux": spectral_flux,
                "temporal_modulation": modulation,
            },
            "related_concepts": related,
        }
        annotation["transcription_status"] = transcription_status
        if transcript:
            annotation["transcript"] = transcript
            annotation["transcript_source"] = transcript_info.get("source", "asr")
        elif transcript_info.get("error"):
            annotation["transcript_error"] = transcript_info["error"]

        if related:
            for concept in related:
                facts.append(
                    KnowledgeFact(
                        subject=subject,
                        predicate="similarTo",
                        obj=concept["id"],
                        source="perception.audio",
                        metadata={
                            "modality": "audio",
                            "related_label": concept["label"],
                        },
                        confidence=concept["similarity"],
                    )
                )

        memory_entry = ExperiencePayload(
            task_id=self._build_memory_task(agent_id),
            summary=f"Audio {subject}: {summary}",
            messages=[
                {"role": "system", "content": "Semantic audio observation"},
                {"role": "observation", "content": summary},
            ],
            metadata={
                "modality": "audio",
                "labels": labels,
                "cycle": cycle_index,
                "subject": subject,
                "transcription_status": transcription_status,
            },
        )
        if transcript:
            memory_entry.metadata["transcript"] = transcript
            memory_entry.metadata["transcript_source"] = transcript_info.get("source", "asr")
            memory_entry.messages.append({"role": "transcript", "content": transcript})
        elif transcript_info.get("error"):
            memory_entry.metadata["transcript_error"] = transcript_info["error"]

        return {"annotation": annotation, "facts": facts, "memory": memory_entry}

    def _encode_text_embedding(self, text: str) -> Tuple[List[float], str, List[str]]:
        tokens = [token for token in text.lower().split() if token]

        def _to_numpy(value: Any) -> np.ndarray:
            if torch is not None and isinstance(value, torch.Tensor):
                return value.detach().cpu().float().numpy()
            return np.asarray(value, dtype=np.float32)

        if self._clip_available and self._clip_extractor is not None:
            try:
                tensor = self._clip_extractor.extract_text_features(text)
            except Exception:
                LOGGER.debug("CLIP text encoding failed; attempting tokenizer fallback.", exc_info=True)
            else:
                try:
                    vector = _to_numpy(tensor)
                except Exception:
                    LOGGER.debug("Failed to convert CLIP text embedding to numpy.", exc_info=True)
                else:
                    if vector.ndim > 1:
                        vector = vector.reshape(-1)
                    norm = float(np.linalg.norm(vector))
                    if norm > 0:
                        vector = vector / norm
                    return vector.astype(float).tolist(), "clip", tokens

            tokenizer = getattr(self._clip_extractor, "tokenizer", None)
            if tokenizer is not None:
                try:
                    token_values = tokenizer([text])
                    vector = _to_numpy(token_values)
                except Exception:
                    LOGGER.debug("CLIP tokenizer fallback failed; resorting to hash embedding.", exc_info=True)
                else:
                    if vector.ndim > 1:
                        vector = vector.reshape(-1)
                    norm = float(np.linalg.norm(vector))
                    if norm > 0:
                        vector = vector / norm
                    return vector.astype(float).tolist(), "tokenizer", tokens

        dimension = 96
        hashed = np.zeros(dimension, dtype=np.float32)
        for index, token in enumerate(tokens or text.split()):
            score = sum(ord(char) for char in token)
            hashed_index = (score + 31 * index) % dimension
            hashed[hashed_index] += 1.0
        if not np.any(hashed):
            hashed[0] = 1.0
        norm = float(np.linalg.norm(hashed))
        if norm > 0:
            hashed = hashed / norm
        return hashed.astype(float).tolist(), "token-hash", tokens

    @staticmethod
    def _summarise_text(text: str, *, limit: int = 160) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        truncated = text[: limit - 1]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        return truncated.rstrip(".,;:-") + "…"

    def _prepare_clip_prompt_embeddings(self) -> None:
        if not self._clip_available or not self._clip_extractor:
            return

        prepared: List[Tuple[str, np.ndarray]] = []
        for label, prompt in self._clip_prompt_table:
            if not prompt:
                continue
            try:
                tensor = self._clip_extractor.extract_text_features(prompt)
            except Exception:  # pragma: no cover - optional dependency guard
                LOGGER.debug("Failed to encode CLIP prompt '%s'.", label, exc_info=True)
                continue

            try:
                if torch is not None and isinstance(tensor, torch.Tensor):  # type: ignore[arg-type]
                    vector = tensor.detach().cpu().float().numpy()
                else:
                    vector = np.asarray(tensor, dtype=np.float32)
            except Exception:  # pragma: no cover - tensor conversion failure
                LOGGER.debug("Unable to convert CLIP prompt '%s' embedding to numpy.", label, exc_info=True)
                continue

            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            prepared.append((label, vector.astype(np.float32)))

        self._clip_prompt_embeddings = prepared

    def _coerce_image_value(self, value: Any) -> Any:
        if Image is None or value is None:
            return None

        try:
            if isinstance(value, Image.Image):  # type: ignore[arg-type]
                return value.convert("RGB")

            if isinstance(value, (bytes, bytearray)):
                with Image.open(io.BytesIO(value)) as img:  # type: ignore[arg-type]
                    return img.convert("RGB")

            if isinstance(value, np.ndarray):
                array = np.asarray(value)
                if array.ndim == 2:
                    return Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
                if array.ndim == 3:
                    return Image.fromarray(array.astype(np.uint8)).convert("RGB")

            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                if text.startswith("data:image") and "," in text:
                    _, b64_data = text.split(",", 1)
                    data = base64.b64decode(b64_data)
                    with Image.open(io.BytesIO(data)) as img:
                        return img.convert("RGB")
                path = Path(text)
                if path.exists():
                    with Image.open(path) as img:
                        return img.convert("RGB")
                try:
                    data = base64.b64decode(text, validate=True)
                except Exception:
                    return None
                with Image.open(io.BytesIO(data)) as img:
                    return img.convert("RGB")
        except Exception:  # pragma: no cover - guard against malformed image data
            LOGGER.debug("Failed to coerce value into image for CLIP analysis.", exc_info=True)
            return None

        return None

    def _resolve_clip_image(
        self, payload: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[Any, Optional[str]]:
        if Image is None:
            return None, None

        candidates: List[Tuple[str, Any]] = []
        for key in ("image", "image_path", "image_bytes", "image_base64"):
            if key in payload:
                candidates.append((f"payload.{key}", payload[key]))
        for key in ("image", "image_path", "image_bytes", "image_base64", "uri", "url", "source", "path"):
            if key in metadata:
                candidates.append((f"metadata.{key}", metadata[key]))

        for origin, value in candidates:
            image = self._coerce_image_value(value)
            if image is not None:
                return image, origin

        return None, None

    def _run_clip_analysis(
        self,
        image: Any,
    ) -> Tuple[List[float], List[str], Optional[str], List[Dict[str, float]]]:
        if not self._clip_extractor:
            return [], [], None, []

        tensor = self._clip_extractor.extract_image_features(image)

        if torch is not None and isinstance(tensor, torch.Tensor):  # type: ignore[arg-type]
            vector = tensor.detach().cpu().float().numpy()
        else:
            vector = np.asarray(tensor, dtype=np.float32)

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        clip_embedding = vector.astype(np.float32).tolist()

        clip_labels: List[str] = []
        scores: List[Dict[str, float]] = []
        clip_caption: Optional[str] = None

        if self._clip_prompt_embeddings:
            similarities: List[Tuple[str, float]] = []
            for label, prompt_vector in self._clip_prompt_embeddings:
                score = float(np.dot(vector, prompt_vector))
                similarities.append((label, score))

            similarities.sort(key=lambda item: item[1], reverse=True)
            limit = min(len(self._clip_prompt_embeddings), max(self._clip_prompt_top_k, 5))
            scores = [
                {"label": label, "score": float(round(score, 4))}
                for label, score in similarities[:limit]
            ]

            top_labels = [label for label, score in similarities[: self._clip_prompt_top_k] if score > 0]
            clip_labels = top_labels
            if similarities:
                best_label, best_score = similarities[0]
                prompt_text = self._clip_prompt_lookup.get(best_label, best_label)
                clip_caption = f"{prompt_text} ({best_score:.2f})"

        return clip_embedding, clip_labels, clip_caption, scores

    # ------------------------------------------------------------------
    @staticmethod
    def _build_subject(agent_id: Optional[str], cycle_index: Optional[int], suffix: str) -> str:
        agent_part = agent_id or "agent"
        cycle_part = cycle_index if cycle_index is not None else 0
        return f"{agent_part}:{suffix}:{cycle_part}"

    @staticmethod
    def _build_memory_task(agent_id: Optional[str]) -> str:
        return f"{agent_id or 'agent'}:perception"

    def _prepare_asr_transcriber(self) -> ASRTranscriber | None:
        if not self._asr_config.enabled:
            return None
        if self._asr_config.transcriber is not None:
            return self._asr_config.transcriber

        provider = (self._asr_config.provider or "").lower()
        if provider == "autogpt":
            agent = self._asr_config.agent
            if agent is None:
                return None
            try:
                from backend.autogpt.autogpt.commands.audio_gen import speech_to_text
            except Exception:
                return None

            model = self._asr_config.model or "whisper-1"

            def _call(audio_input: ASRInput, metadata: Dict[str, Any]) -> str:
                if isinstance(audio_input, (str, Path)):
                    return speech_to_text(str(audio_input), agent, model=model)
                raise ValueError("AutoGPT speech_to_text requires a file path input.")

            return _call

        return None

    def _resolve_transcript(self, payload: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any]]:
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        existing = payload.get("transcript") or metadata.get("transcript")
        if isinstance(existing, str):
            text = existing.strip()
            if text:
                source = metadata.get("transcript_source") or "payload"
                return text, {"source": source}

        if not self._asr_config.enabled:
            return None, {"reason": "disabled"}

        transcriber = self._asr_transcriber
        if transcriber is None:
            return None, {"reason": "unavailable"}

        audio_input = self._extract_audio_input(payload)
        if audio_input is None:
            return None, {"reason": "no_input"}

        try:
            transcript = transcriber(audio_input, metadata)
        except Exception as exc:
            LOGGER.debug("ASR transcription failed: %s", exc, exc_info=True)
            return None, {"error": str(exc)}

        if transcript is None:
            return None, {"reason": "empty"}

        text = str(transcript).strip()
        if not text:
            return None, {"reason": "empty"}

        return text, {"source": self._asr_config.provider or "asr"}

    @staticmethod
    def _extract_audio_input(payload: Dict[str, Any]) -> ASRInput | None:
        candidates = [payload]
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            candidates.append(metadata)

        for container in candidates:
            for key in ("audio_path", "file_path", "path"):
                value = container.get(key)
                if isinstance(value, (str, Path)):
                    return str(value)

        for container in candidates:
            for key in ("raw_waveform", "waveform", "raw_signal", "signal", "audio_samples"):
                if key in container:
                    return container[key]

        for container in candidates:
            value = container.get("audio_bytes")
            if isinstance(value, (bytes, bytearray)):
                return bytes(value)

        return None

    def _find_related_concepts(
        self,
        embedding: List[float],
        vector_type: str,
        *,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        if not self._aligner or not embedding:
            return []
        try:
            matches = self._aligner.align(embedding, n_results=max(1, limit), vector_type=vector_type)
        except Exception:
            return []
        related: List[Dict[str, Any]] = []
        for node in matches:
            similarity = float(node.metadata.get("similarity", 0.0))
            related.append(
                {
                    "id": node.id,
                    "label": node.label,
                    "similarity": similarity,
                }
            )
        return related

