import asyncio
from dataclasses import dataclass
from typing import Iterator

import numpy as np

import types
import sys

# Provide minimal config to satisfy imports
class SimpleConfig:
    fast_llm = "gpt-3.5-turbo"
    embedding_model = "text-embedding-3-small"
    plugins = []

sys.modules['autogpt.config'] = types.SimpleNamespace(Config=SimpleConfig)
sys.modules['autogpt.config.config'] = types.SimpleNamespace(Config=SimpleConfig)

from third_party.autogpt.autogpt.memory.vector.memory_item import MemoryItemFactory, MemoryItemRelevance
from third_party.autogpt.autogpt.memory.vector.utils import get_embedding
from third_party.autogpt.autogpt.core.resource.model_providers.schema import (
    EmbeddingModelProvider,
    EmbeddingModelInfo,
    EmbeddingModelResponse,
    ChatModelProvider,
    ChatModelInfo,
    ChatModelResponse,
    AssistantChatMessage,
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

    def count_tokens(self, text: str, model_name: str) -> int:
        return len(text.split())

    def get_tokenizer(self, model_name: str) -> ModelTokenizer:
        return self.tokenizer

    def get_token_limit(self, model_name: str) -> int:
        return 1000

    async def create_embedding(
        self, text: str, model_name: str, embedding_parser, **kwargs
    ) -> EmbeddingModelResponse:
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


async def main():
    config = SimpleConfig()
    embedding_provider = FakeEmbeddingProvider()
    chat_provider = FakeChatProvider()
    factory = MemoryItemFactory(chat_provider, embedding_provider)

    import third_party.autogpt.autogpt.memory.vector.memory_item as mi

    tokenizer = embedding_provider.get_tokenizer(config.fast_llm)

    def split_text_stub(text: str, **kwargs) -> Iterator[tuple[str, int]]:
        return [(text, len(tokenizer.encode(text)))]

    async def summarize_text_stub(text: str, **kwargs):
        return text, None

    mi.split_text = split_text_stub  # type: ignore
    mi.summarize_text = summarize_text_stub  # type: ignore

    texts = [
        "Cats purr and scratch",
        "Dogs bark loudly",
        "Birds can fly",
    ]
    queries = ["cat", "dog", "bird"]

    items = [
        await factory.from_text(t, "text_file", config) for t in texts
    ]

    summary_correct = 0
    average_correct = 0

    for i, q in enumerate(queries):
        q_embed = await get_embedding(q, config, embedding_provider)
        summary_scores = [
            MemoryItemRelevance.calculate_scores(item, q_embed)[0] for item in items
        ]
        avg_embeds = [
            np.average(item.e_chunks, axis=0, weights=[len(c) for c in item.chunks])
            for item in items
        ]
        avg_scores = [float(np.dot(e, q_embed)) for e in avg_embeds]
        if int(np.argmax(summary_scores)) == i:
            summary_correct += 1
        if int(np.argmax(avg_scores)) == i:
            average_correct += 1

    total = len(queries)
    print(
        f"Summary-based accuracy: {summary_correct}/{total}\n"
        f"Weighted-average accuracy: {average_correct}/{total}"
    )
    if summary_correct >= average_correct:
        print("Chosen approach: summary-based")
    else:
        print("Chosen approach: weighted-average")


if __name__ == "__main__":
    asyncio.run(main())
