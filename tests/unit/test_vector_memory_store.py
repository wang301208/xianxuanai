import importlib

import pytest

from modules.memory.vector_store import VectorMemoryStore

np = pytest.importorskip("numpy")

embedders_module = importlib.import_module("modules.memory.embedders")


class _DummySentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def encode(self, texts, normalize_embeddings: bool = True, batch_size: int | None = None):
        if isinstance(texts, str):
            return np.arange(4, dtype=np.float32) + 1
        embeddings = []
        for idx, text in enumerate(texts):
            base = np.arange(4, dtype=np.float32) + 1
            embeddings.append(base + idx)
        return np.stack(embeddings)

    def get_sentence_embedding_dimension(self) -> int:
        return 4


def test_vector_store_falls_back_to_hash(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_EMBEDDER", "transformer")
    monkeypatch.setattr(embedders_module, "SentenceTransformer", None)

    store = VectorMemoryStore(tmp_path)

    assert isinstance(store.embedder, embedders_module.HashingEmbedder)


def test_vector_store_uses_transformer_embedder(monkeypatch, tmp_path):
    monkeypatch.setattr(embedders_module, "SentenceTransformer", _DummySentenceTransformer)

    store = VectorMemoryStore(
        tmp_path,
        embedder="transformer",
        embedder_options={"model_name": "dummy-transformer"},
    )

    assert isinstance(store.embedder, embedders_module.TransformerEmbedder)
    record_id = store.add_text("hello world", {"task_id": "demo"})
    results = store.query("hello world", top_k=1)
    assert results and results[0].id == record_id


@pytest.mark.skipif(
    embedders_module.SentenceTransformer is None,
    reason="sentence-transformers not installed",
)
def test_transformer_embedder_real_dependency():
    try:
        embedder = embedders_module.TransformerEmbedder()
    except Exception as exc:  # pragma: no cover - requires optional dependency
        pytest.skip(f"sentence-transformers unavailable in environment: {exc}")
    vector = embedder("hello world")
    assert vector.ndim == 1
    assert vector.shape[0] == embedder.dimension
