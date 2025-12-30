import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'autogpts' / 'autogpt'))


class SimpleConfig:
    fast_llm = "gpt-3.5-turbo"
    embedding_model = "text-embedding-3-small"
    plugins = []

import types
sys.modules['autogpt.config'] = types.SimpleNamespace(Config=SimpleConfig)
sys.modules['autogpt.config.config'] = types.SimpleNamespace(Config=SimpleConfig)
sys.modules['spacy'] = types.SimpleNamespace(load=lambda *args, **kwargs: None)
from third_party.autogpt.autogpt.memory.vector.memory_item import MemoryItemFactory, MemoryItemRelevance
from third_party.autogpt.autogpt.memory.vector.utils import get_embedding
from third_party.autogpt.autogpt.processing.code import chunk_code_by_structure
from third_party.autogpt.autogpt.core.resource.model_providers.schema import (
    EmbeddingModelProvider,
    EmbeddingModelInfo,
    EmbeddingModelResponse,
    ChatModelProvider,
    ChatModelInfo,
    ChatModelResponse,
    ModelProviderName,
    ModelTokenizer,
)


class SimpleTokenizer(ModelTokenizer):
    def encode(self, text: str) -> list:
        return text.split()

    def decode(self, tokens: list) -> str:
        return " ".join(tokens)


class FakeEmbeddingProvider(EmbeddingModelProvider):
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        self.calls: list[str] = []

    def count_tokens(self, text: str, model_name: str) -> int:
        return len(text.split())

    def get_tokenizer(self, model_name: str) -> ModelTokenizer:
        return self.tokenizer

    def get_token_limit(self, model_name: str) -> int:
        return 1000

    async def create_embedding(
        self, text: str, model_name: str, embedding_parser, **kwargs
    ) -> EmbeddingModelResponse:
        self.calls.append(text)
        vocab = ["cat", "dog", "bird"]
        vec = [text.lower().count(w) for w in vocab]
        info = EmbeddingModelInfo(
            name=model_name,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0,
            max_tokens=1000,
            embedding_dimensions=len(vocab),
        )
        return EmbeddingModelResponse(
            embedding=embedding_parser(vec),
            model_info=info,
            prompt_tokens_used=0,
            completion_tokens_used=0,
        )


class FakeChatProvider(ChatModelProvider):
    def __init__(self):
        self.tokenizer = SimpleTokenizer()

    async def get_available_models(self) -> list[ChatModelInfo]:
        return []

    def count_message_tokens(self, messages, model_name: str) -> int:
        return 0

    async def create_chat_completion(self, *args, **kwargs) -> ChatModelResponse:
        raise NotImplementedError

    def count_tokens(self, text: str, model_name: str) -> int:
        return len(text.split())

    def get_tokenizer(self, model_name: str) -> ModelTokenizer:
        return self.tokenizer

    def get_token_limit(self, model_name: str) -> int:
        return 1000


@pytest.mark.asyncio
async def test_embedding_generation(monkeypatch):
    config = SimpleConfig()
    embed_provider = FakeEmbeddingProvider()
    chat_provider = FakeChatProvider()
    factory = MemoryItemFactory(chat_provider, embed_provider)

    import third_party.autogpt.autogpt.memory.vector.memory_item as mi

    tokenizer = embed_provider.get_tokenizer(config.fast_llm)

    def split_text_stub(text: str, **kwargs):
        return [(text, len(tokenizer.encode(text)))]

    async def summarize_text_stub(text: str, **kwargs):
        return text, None

    monkeypatch.setattr(mi, "split_text", split_text_stub)
    monkeypatch.setattr(mi, "summarize_text", summarize_text_stub)

    item = await factory.from_text("Cats purr", "text_file", config)

    assert embed_provider.calls[0] == "Cats purr"
    assert embed_provider.calls[-1] == "Cats purr"
    assert item.e_summary is not None
    assert len(item.e_chunks) == 1


def test_chunk_code_by_structure():
    code = """
import os

def foo():
    pass

class Bar:
    def baz(self):
        pass
"""
    tokenizer = SimpleTokenizer()
    chunks = list(chunk_code_by_structure(code, 50, tokenizer))
    joined = "\n".join(c for c, _ in chunks)
    assert "def foo" in joined
    assert "class Bar" in joined


@pytest.mark.asyncio
async def test_search_retrieval(monkeypatch):
    config = SimpleConfig()
    embed_provider = FakeEmbeddingProvider()
    chat_provider = FakeChatProvider()
    factory = MemoryItemFactory(chat_provider, embed_provider)

    import third_party.autogpt.autogpt.memory.vector.memory_item as mi

    tokenizer = embed_provider.get_tokenizer(config.fast_llm)

    def split_text_stub(text: str, **kwargs):
        return [(text, len(tokenizer.encode(text)))]

    async def summarize_text_stub(text: str, **kwargs):
        return text, None

    monkeypatch.setattr(mi, "split_text", split_text_stub)
    monkeypatch.setattr(mi, "summarize_text", summarize_text_stub)

    item_cat = await factory.from_text("Cats purr", "text_file", config)
    item_dog = await factory.from_text("Dogs bark", "text_file", config)

    q_embed = await get_embedding("cat", config, embed_provider)
    score_cat, _, _ = MemoryItemRelevance.calculate_scores(item_cat, q_embed)
    score_dog, _, _ = MemoryItemRelevance.calculate_scores(item_dog, q_embed)
    assert score_cat > score_dog


@pytest.mark.asyncio
async def test_code_memory(monkeypatch):
    config = SimpleConfig()
    embed_provider = FakeEmbeddingProvider()
    chat_provider = FakeChatProvider()
    factory = MemoryItemFactory(chat_provider, embed_provider)

    import third_party.autogpt.autogpt.memory.vector.memory_item as mi

    async def summarize_text_stub(text: str, **kwargs):
        return text, None

    monkeypatch.setattr(mi, "summarize_text", summarize_text_stub)

    code = "def foo():\n    return 1\n"
    item = await factory.from_code_file(code, "foo.py", config)

    assert item.metadata["language"] == "python"
    assert "foo" in item.metadata["symbols"]
    assert item.summary != ""
    assert item.e_summary is not None
    assert len(item.e_chunks) == 1
