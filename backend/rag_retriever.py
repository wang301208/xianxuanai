"""Simple Retrieval Augmented Generation helper."""
from __future__ import annotations

import math
from collections import deque
from functools import lru_cache
from typing import Callable, Deque, Iterable, List, Tuple

from capability.librarian import Librarian


class RAGRetriever:
    """Perform retrieval augmented generation using tiered caches."""

    def __init__(self, librarian: Librarian, *, hot_capacity: int = 64) -> None:
        self.librarian = librarian
        # Reusable buffer for retrieved documents to reduce temporary object creation.
        self._docs_buffer: List[str] = []
        self._hot_cache: Deque[Tuple[Tuple[float, ...], str]] = deque(maxlen=max(1, hot_capacity))

    @staticmethod
    def _cosine_similarity(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _search_hot(self, embedding: Tuple[float, ...], top_k: int) -> List[str]:
        if not self._hot_cache or top_k <= 0:
            return []
        scored = [
            (self._cosine_similarity(stored_embedding, embedding), document)
            for stored_embedding, document in self._hot_cache
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        seen: set[str] = set()
        results: List[str] = []
        for score, document in scored:
            if document in seen:
                continue
            if score <= 0.0:
                continue
            results.append(document)
            seen.add(document)
            if len(results) == top_k:
                break
        return results

    def _update_hot_cache(self, embedding: Tuple[float, ...], documents: Iterable[str]) -> None:
        if not documents:
            return
        for document in documents:
            if not document:
                continue
            if self._hot_cache:
                retained = [(vec, doc) for vec, doc in self._hot_cache if doc != document]
                self._hot_cache.clear()
                for vec, doc in retained:
                    self._hot_cache.append((vec, doc))
            self._hot_cache.appendleft((embedding, document))

    @lru_cache(maxsize=128)
    def _cached_search(
        self, embedding_key: Tuple[float, ...], n_results: int, vector_type: str
    ) -> Tuple[str, ...]:
        """Cache search results for repeated queries."""
        return tuple(
            self.librarian.search(
                list(embedding_key),
                n_results=n_results,
                vector_type=vector_type,
                return_content=True,
            )
        )

    def generate(
        self,
        prompt: str,
        query_embedding: List[float],
        llm_callable: Callable[[str], str],
        n_results: int = 3,
        vector_type: str = "text",
    ) -> str:
        """Generate LLM output with retrieved context.

        Parameters
        ----------
        prompt: str
            The user prompt/question.
        query_embedding: List[float]
            Embedding of the query used for similarity search.
        llm_callable: Callable[[str], str]
            Function that takes the final prompt and returns generated text.
        n_results: int
            Number of documents to retrieve.
        vector_type: str
            Vector space to query (e.g. ``"text"`` or ``"image"``).
        """
        embedding_key = tuple(query_embedding)
        docs: List[str] = []
        hot_hits = self._search_hot(embedding_key, n_results)
        docs.extend(hot_hits)

        if len(docs) < n_results:
            cold_hits = self._cached_search(embedding_key, n_results, vector_type)
            for document in cold_hits:
                if document in docs:
                    continue
                docs.append(document)
                if len(docs) == n_results:
                    break

        self._update_hot_cache(embedding_key, docs)
        self._docs_buffer.clear()
        self._docs_buffer.extend(docs)
        context = "\n".join(self._docs_buffer)
        final_prompt = f"{context}\n\n{prompt}" if context else prompt
        return llm_callable(final_prompt)

