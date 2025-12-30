from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union


@dataclass
class MultimodalInput:
    """Container for text with optional image and metadata.

    Attributes:
        text: The textual portion of the input.
        image: Optional raw image data or a string pointing to an image resource.
        metadata: Arbitrary metadata associated with the input.
    """

    text: str
    image: Optional[Union[bytes, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def merge_embeddings(
    text_embedding: Iterable[float], image_embedding: Optional[Iterable[float]]
) -> List[float]:
    """Merge text and image embeddings by element-wise addition.

    If one of the embeddings is shorter, it will be padded with zeros before the
    element-wise addition.
    """

    text_vec = list(text_embedding)
    if not image_embedding:
        return text_vec

    img_vec = list(image_embedding)
    length = max(len(text_vec), len(img_vec))
    text_vec.extend([0.0] * (length - len(text_vec)))
    img_vec.extend([0.0] * (length - len(img_vec)))
    return [t + i for t, i in zip(text_vec, img_vec)]


def embed_multimodal_input(
    multimodal_input: MultimodalInput,
    embed_fn: Callable[[Union[str, bytes]], Iterable[float]],
) -> List[float]:
    """Create a merged embedding for a ``MultimodalInput``.

    ``embed_fn`` is a callable used to obtain embeddings for the text and image
    components. The resulting embeddings are merged using ``merge_embeddings``.
    """

    text_embedding = embed_fn(multimodal_input.text)
    image_embedding = (
        embed_fn(multimodal_input.image) if multimodal_input.image is not None else None
    )
    return merge_embeddings(text_embedding, image_embedding)
