import pytest

from autogpt.core.multimodal import (
    MultimodalInput,
    embed_multimodal_input,
    merge_embeddings,
)


def test_merge_embeddings():
    text_emb = [1.0, 2.0]
    img_emb = [0.5]
    assert merge_embeddings(text_emb, img_emb) == [1.5, 2.0]


def test_embed_multimodal_input():
    mm_input = MultimodalInput(text="ab", image=b"\x01")

    def _embed(data: str | bytes):
        if isinstance(data, (bytes, bytearray)):
            return [float(b) for b in data]
        return [float(ord(c)) for c in data]

    assert embed_multimodal_input(mm_input, _embed) == [98.0, 98.0]
