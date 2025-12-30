"""Unit tests for :mod:`backend.creative_engine`."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List

yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda _data: {}
sys.modules.setdefault("yaml", yaml_stub)

from backend.creative_engine import CrossModalCreativeEngine
from modules.common.concepts import ConceptNode


@dataclass
class RecordingAligner:
    """Capture align requests for assertions."""

    calls: List[Dict[str, Any]]

    def align(self, embedding: List[float], vector_type: str) -> List[ConceptNode]:
        self.calls.append({"embedding": embedding, "vector_type": vector_type})
        node = ConceptNode(
            id=f"{vector_type}-node",
            label=f"{vector_type} concept",
            modalities={vector_type: embedding},
        )
        return [node]


def flexible_generator(prompt: str, concepts: List[ConceptNode], **kwargs: Any) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "concept_ids": [c.id for c in concepts],
        "kwargs": kwargs,
    }


def simple_encoder(data: Any):
    if isinstance(data, str) and data.startswith("image:"):
        vector = [1.0, 0.5, -0.5]
        return {"embedding": vector, "source": data}
    return {"embedding": [float(len(str(data)))], "source": data}


def legacy_generator(prompt: str, concepts: List[ConceptNode]) -> Dict[str, Any]:
    return {"prompt": prompt, "concept_ids": [c.id for c in concepts]}


def test_generate_supports_multimodal_inputs() -> None:
    aligner = RecordingAligner(calls=[])
    engine = CrossModalCreativeEngine(
        aligner=aligner,
        encoders={"text": simple_encoder, "image": simple_encoder},
        generators={"text": flexible_generator, "image": flexible_generator},
    )

    inputs = {"image": "image:/tmp/mock.png"}
    result = engine.generate("describe object", ["text", "image"], inputs=inputs)

    assert {call["vector_type"] for call in aligner.calls} == {"text", "image"}
    image_call = next(call for call in aligner.calls if call["vector_type"] == "image")
    assert image_call["embedding"] == [1.0, 0.5, -0.5]

    image_result = result["image"]
    assert image_result["encoder_metadata"]["source"] == inputs["image"]
    assert image_result["concepts"][0].id == "image-node"

    generator_kwargs = image_result["output"]["kwargs"]
    assert generator_kwargs["modality"] == "image"
    assert generator_kwargs["input_data"] == inputs["image"]
    assert generator_kwargs["embedding"] == [1.0, 0.5, -0.5]


def test_generate_backward_compatible_generator() -> None:
    aligner = RecordingAligner(calls=[])
    engine = CrossModalCreativeEngine(
        aligner=aligner,
        encoders={"text": simple_encoder},
        generators={"text": legacy_generator},
    )

    result = engine.generate("hello", ["text"])

    assert result["text"]["output"]["concept_ids"] == ["text-node"]
