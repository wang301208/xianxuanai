"""Cross-modal creative synthesis engine.

This module provides :class:`CrossModalCreativeEngine` which links text,
image and audio modalities through a shared concept graph. The engine uses
:class:`backend.concept_alignment.ConceptAligner` to retrieve concept nodes
related to a given prompt and can delegate generation to optional external
models for each modality.

Example
-------
>>> aligner = ConceptAligner(...)
>>> encoders = {"text": text_encoder}
>>> generators = {"image": image_model.generate}
>>> engine = CrossModalCreativeEngine(aligner, encoders, generators)
>>> result = engine.generate("a cat playing piano", ["text", "image"])
>>> result["image"]["output"]  # Data produced by image_model

The ``generators`` mapping allows integration with third-party models for
rendering images, audio, or other modalities. When a generator is not
provided for a requested modality, the engine returns the retrieved concept
nodes so callers can implement custom handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Tuple, Union

try:  # optional dependency used for convenient array handling
    import numpy as _np
except Exception:  # pragma: no cover - numpy might be absent in lightweight envs
    _np = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    import numpy as np
else:
    class np:  # type: ignore[too-many-ancestors]
        ndarray = Any

EncoderInput = Any
EncoderReturn = Union[
    Iterable[float],
    "np.ndarray",
    MutableMapping[str, Any],
    Tuple[Iterable[float], MutableMapping[str, Any]],
]


def _normalise_embedding(output: EncoderReturn) -> Tuple[List[float], Dict[str, Any]]:
    """Convert encoder output into a numeric embedding and metadata."""

    metadata: Dict[str, Any] = {}
    embedding: Optional[Iterable[float]]

    if isinstance(output, tuple) and len(output) == 2:
        embedding, extra = output
        metadata = dict(extra)
    elif isinstance(output, MutableMapping):
        embedding = output.get("embedding")  # type: ignore[assignment]
        metadata = {k: v for k, v in output.items() if k != "embedding"}
    else:
        embedding = output

    if _np is not None and isinstance(embedding, _np.ndarray):
        vector = embedding.astype(float).tolist()
    elif embedding is None:
        raise ValueError("Encoder did not return an embedding vector")
    else:
        vector = list(float(v) for v in embedding)

    return vector, metadata

from modules.common.concepts import ConceptNode
from backend.concept_alignment import ConceptAligner
from modules.metrics.creative_evaluator import CreativeEvaluator, CreativeScore
from modules.optimization.meta_learner import MetaLearner


@dataclass
class CrossModalCreativeEngine:
    """Compose multimodal outputs by aligning prompts to concept graphs.

    Parameters
    ----------
    aligner:
        Instance of :class:`ConceptAligner` used to retrieve related concepts.
    encoders:
        Mapping from modality name to a callable that converts a text prompt
        to an embedding vector in that modality's space.
    generators:
        Optional mapping from modality name to a callable that consumes a
        prompt and list of related :class:`ConceptNode` objects and returns
        generated content for that modality.
    """

    aligner: ConceptAligner
    encoders: Dict[str, Callable[[EncoderInput], EncoderReturn]]
    generators: Dict[str, Callable[[str, List[ConceptNode]], Any]] = None
    evaluator: Optional[CreativeEvaluator] = None
    meta_learner: Optional[MetaLearner] = None

    def __post_init__(self) -> None:
        if self.generators is None:
            self.generators = {}

    def generate(
        self,
        prompt: str,
        modalities: List[str],
        *,
        inputs: Optional[Dict[str, EncoderInput]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate outputs for the requested modalities.

        For each modality an encoder converts the provided ``inputs`` entry (or
        the textual ``prompt`` when no explicit input is supplied) into an
        embedding compatible with the concept graph. The embedding is aligned
        to related concepts via :class:`ConceptAligner`. If a generator is
        registered for the modality its output is returned under ``"output"``;
        otherwise the caller receives the retrieved concept nodes and encoder
        metadata so additional processing can be performed externally.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for modality in modalities:
            encoder = self.encoders.get(modality)
            if encoder is None:
                raise ValueError(f"No encoder available for modality '{modality}'")
            payload = prompt if inputs is None else inputs.get(modality, prompt)
            raw_embedding = encoder(payload)
            embedding, encoder_metadata = _normalise_embedding(raw_embedding)
            concepts = self.aligner.align(embedding, vector_type=modality)
            generator = self.generators.get(modality)
            output = None
            if generator:
                try:
                    output = generator(
                        prompt,
                        concepts,
                        modality=modality,
                        input_data=payload,
                        embedding=embedding,
                        encoder_metadata=encoder_metadata,
                    )
                except TypeError:
                    output = generator(prompt, concepts)

            results[modality] = {
                "concepts": concepts,
                "embedding": embedding,
                "encoder_metadata": encoder_metadata or None,
                "output": output,
            }

        if self.evaluator is not None:
            scores = self.evaluator.evaluate(results)
            self.evaluator.feedback(self.meta_learner, scores)
            for modality, score in scores.items():
                results[modality]["creative_score"] = score

        return results


__all__ = ["CrossModalCreativeEngine", "EncoderInput", "EncoderReturn"]
