from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import ftfy
import numpy as np
from pydantic import BaseModel

from autogpt.config import Config
from autogpt.core.resource.model_providers import (
    ChatMessage,
    ChatModelProvider,
    EmbeddingModelProvider,
)
try:  # pragma: no cover - optional heavy dependencies
    from autogpt.processing.text import chunk_content, split_text, summarize_text
    from autogpt.processing.code import chunk_code_by_structure
except Exception:  # pragma: no cover
    def chunk_content(*args, **kwargs):
        return []

    def split_text(*args, **kwargs):
        return []

    async def summarize_text(*args, **kwargs):
        return "", None

    def chunk_code_by_structure(*args, **kwargs):
        return []

from .utils import Embedding, get_embedding

try:  # pragma: no cover - optional dependency
    from tree_sitter_languages import get_parser
except Exception:  # pragma: no cover
    get_parser = None

logger = logging.getLogger(__name__)

MemoryDocType = Literal["webpage", "text_file", "code_file", "agent_history"]


class MemoryItem(BaseModel, arbitrary_types_allowed=True):
    """Memory object containing raw content as well as embeddings"""

    raw_content: str
    summary: str
    chunks: list[str]
    chunk_summaries: list[str]
    e_summary: Embedding
    e_weighted: Embedding
    e_chunks: list[Embedding]
    metadata: dict

    def relevance_for(
        self,
        query: str,
        e_query: Embedding | None = None,
        strategy: Literal["summary", "weighted"] = "summary",
    ):
        return MemoryItemRelevance.of(self, query, e_query, strategy)

    def dump(self, calculate_length=False) -> str:
        n_chunks = len(self.e_chunks)
        return f"""
=============== MemoryItem ===============
Size: {n_chunks} chunks
Metadata: {json.dumps(self.metadata, indent=2)}
---------------- SUMMARY -----------------
{self.summary}
------------------ RAW -------------------
{self.raw_content}
==========================================
"""

    def __eq__(self, other: MemoryItem):
        return (
            self.raw_content == other.raw_content
            and self.chunks == other.chunks
            and self.chunk_summaries == other.chunk_summaries
            # Embeddings can either be list[float] or np.ndarray[float32],
            # and for comparison they must be of the same type
            and np.array_equal(
                self.e_summary
                if isinstance(self.e_summary, np.ndarray)
                else np.array(self.e_summary, dtype=np.float32),
                other.e_summary
                if isinstance(other.e_summary, np.ndarray)
                else np.array(other.e_summary, dtype=np.float32),
            )
            and np.array_equal(
                self.e_weighted
                if isinstance(self.e_weighted, np.ndarray)
                else np.array(self.e_weighted, dtype=np.float32),
                other.e_weighted
                if isinstance(other.e_weighted, np.ndarray)
                else np.array(other.e_weighted, dtype=np.float32),
            )
            and np.array_equal(
                self.e_chunks
                if isinstance(self.e_chunks[0], np.ndarray)
                else [np.array(c, dtype=np.float32) for c in self.e_chunks],
                other.e_chunks
                if isinstance(other.e_chunks[0], np.ndarray)
                else [np.array(c, dtype=np.float32) for c in other.e_chunks],
            )
        )


class MemoryItemFactory:
    def __init__(
        self,
        llm_provider: ChatModelProvider,
        embedding_provider: EmbeddingModelProvider,
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider

    async def from_text(
        self,
        text: str,
        source_type: MemoryDocType,
        config: Config,
        metadata: dict = {},
        how_to_summarize: str | None = None,
        question_for_summary: str | None = None,
    ):
        logger.debug(f"Memorizing text:\n{'-'*32}\n{text}\n{'-'*32}\n")

        # Fix encoding, e.g. removing unicode surrogates (see issue #778)
        text = ftfy.fix_text(text)

        tokenizer = self.llm_provider.get_tokenizer(config.fast_llm)
        chunks = [
            chunk
            for chunk, _ in (
                split_text(
                    text=text,
                    config=config,
                    max_chunk_length=1000,  # arbitrary, but shorter ~= better
                    tokenizer=tokenizer,
                )
                if source_type != "code_file"
                else chunk_code_by_structure(
                    code=text,
                    max_chunk_length=1000,
                    tokenizer=tokenizer,
                )
            )
        ]
        logger.debug("Chunks: " + str(chunks))

        chunk_summaries = [
            summary
            for summary, _ in [
                await summarize_text(
                    text=text_chunk,
                    instruction=how_to_summarize,
                    question=question_for_summary,
                    llm_provider=self.llm_provider,
                    config=config,
                )
                for text_chunk in chunks
            ]
        ]
        logger.debug("Chunk summaries: " + str(chunk_summaries))

        e_chunks_list = await get_embedding(chunks, config, self.embedding_provider)
        e_chunks = np.array(e_chunks_list, dtype=np.float32)

        # Weight chunk embeddings by token count to capture their relative size
        chunk_token_lengths = [len(tokenizer.encode(c)) for c in chunks]
        e_weighted = np.average(e_chunks, axis=0, weights=chunk_token_lengths).astype(
            np.float32
        )

        summary = (
            chunk_summaries[0]
            if len(chunks) == 1
            else (
                await summarize_text(
                    text="\n\n".join(chunk_summaries),
                    instruction=how_to_summarize,
                    question=question_for_summary,
                    llm_provider=self.llm_provider,
                    config=config,
                )
            )[0]
        )
        logger.debug("Total summary: " + summary)

        # TODO: investigate search performance of weighted average vs summary
        # e_average = np.average(e_chunks, axis=0, weights=[len(c) for c in chunks])
        e_summary = np.array(
            await get_embedding(summary, config, self.embedding_provider),
            dtype=np.float32,
        )

        metadata["source_type"] = source_type

        return MemoryItem(
            raw_content=text,
            summary=summary,
            chunks=chunks,
            chunk_summaries=chunk_summaries,
            e_summary=e_summary,
            e_weighted=e_weighted,
            e_chunks=e_chunks,
            metadata=metadata,
        )

    def from_text_file(self, content: str, path: str, config: Config):
        return self.from_text(content, "text_file", config, {"location": path})

    async def from_code_file(self, content: str, path: str, config: Config):
        """Create a memory item from a code file."""

        language_map = {
            ".c": "c",
            ".cpp": "cpp",
            ".cs": "c_sharp",
            ".go": "go",
            ".java": "java",
            ".js": "javascript",
            ".jsx": "javascript",
            ".php": "php",
            ".py": "python",
            ".rb": "ruby",
            ".rs": "rust",
            ".sh": "bash",
            ".swift": "swift",
            ".ts": "typescript",
            ".tsx": "typescript",
        }

        language = language_map.get(Path(path).suffix.lower())

        def extract_symbols(code: str) -> list[str]:
            if not language or not get_parser:
                return []
            try:
                parser = get_parser(language)
                tree = parser.parse(code.encode("utf-8"))
                symbols: list[str] = []

                def visit(node):
                    if "function" in node.type or "class" in node.type:
                        name_node = node.child_by_field_name("name")
                        if name_node:
                            symbols.append(
                                code[name_node.start_byte : name_node.end_byte]
                            )
                    for child in node.children:
                        visit(child)

                visit(tree.root_node)
                return symbols
            except Exception:
                return []

        symbols = extract_symbols(content)

        tokenizer = self.llm_provider.get_tokenizer(config.fast_llm)
        max_chunk_length = 1000

        chunks: list[str] = []
        if language and get_parser:
            try:
                parser = get_parser(language)
                tree = parser.parse(content.encode("utf-8"))
                byte_code = content.encode("utf-8")
                last_end = 0
                for node in tree.root_node.children:
                    start, end = node.start_byte, node.end_byte
                    snippet = byte_code[start:end].decode("utf-8")
                    length = len(tokenizer.encode(snippet))
                    if length <= max_chunk_length:
                        chunks.append(snippet)
                    else:
                        chunks.extend(
                            c for c, _ in chunk_content(snippet, max_chunk_length, tokenizer)
                        )
                    last_end = end
                if last_end < len(byte_code):
                    snippet = byte_code[last_end:].decode("utf-8")
                    chunks.append(snippet)
            except Exception:
                chunks = [
                    c for c, _ in chunk_content(content, max_chunk_length, tokenizer)
                ]
        else:
            chunks = [c for c, _ in chunk_content(content, max_chunk_length, tokenizer)]

        how_to_summarize = (
            "Provide a concise summary of the following code."
        )
        chunk_summaries = [
            (
                await summarize_text(
                    text=chunk,
                    instruction=how_to_summarize,
                    llm_provider=self.llm_provider,
                    config=config,
                )
            )[0]
            for chunk in chunks
        ]
        e_chunks = await get_embedding(chunks, config, self.embedding_provider)

        chunk_token_lengths = [len(tokenizer.encode(c)) for c in chunks]
        e_weighted = np.average(e_chunks, axis=0, weights=chunk_token_lengths)

        summary = (
            chunk_summaries[0]
            if len(chunks) == 1
            else (
                await summarize_text(
                    text="\n\n".join(chunk_summaries),
                    instruction=how_to_summarize,
                    llm_provider=self.llm_provider,
                    config=config,
                )
            )[0]
        )
        e_summary = await get_embedding(summary, config, self.embedding_provider)

        metadata = {
            "location": path,
            "language": language,
            "symbols": symbols,
            "source_type": "code_file",
        }

        return MemoryItem(
            raw_content=content,
            summary=summary,
            chunks=chunks,
            chunk_summaries=chunk_summaries,
            e_summary=e_summary,
            e_weighted=e_weighted,
            e_chunks=e_chunks,
            metadata=metadata,
        )

    def from_ai_action(self, ai_message: ChatMessage, result_message: ChatMessage):
        # The result_message contains either user feedback
        # or the result of the command specified in ai_message

        if ai_message.role != "assistant":
            raise ValueError(f"Invalid role on 'ai_message': {ai_message.role}")

        result = (
            result_message.content
            if result_message.content.startswith("Command")
            else "None"
        )
        user_input = (
            result_message.content
            if result_message.content.startswith("Human feedback")
            else "None"
        )
        memory_content = (
            f"Assistant Reply: {ai_message.content}"
            "\n\n"
            f"Result: {result}"
            "\n\n"
            f"Human Feedback: {user_input}"
        )

        return self.from_text(
            text=memory_content,
            source_type="agent_history",
            how_to_summarize=(
                "if possible, also make clear the link between the command in the"
                " assistant's response and the command result. "
                "Do not mention the human feedback if there is none.",
            ),
        )

    def from_webpage(
        self, content: str, url: str, config: Config, question: str | None = None
    ):
        return self.from_text(
            text=content,
            source_type="webpage",
            config=config,
            metadata={"location": url},
            question_for_summary=question,
        )


class MemoryItemRelevance(BaseModel):
    """
    Class that encapsulates memory relevance search functionality and data.
    Instances contain a MemoryItem and its relevance scores for a given query.
    """

    memory_item: MemoryItem
    for_query: str
    summary_relevance_score: float
    chunk_relevance_scores: list[float]

    @staticmethod
    def of(
        memory_item: MemoryItem,
        for_query: str,
        e_query: Embedding | None = None,
        strategy: Literal["summary", "weighted"] = "summary",
    ) -> MemoryItemRelevance:
        e_query = e_query if e_query is not None else get_embedding(for_query)
        _, srs, crs = MemoryItemRelevance.calculate_scores(
            memory_item, e_query, strategy
        )
        return MemoryItemRelevance(
            for_query=for_query,
            memory_item=memory_item,
            summary_relevance_score=srs,
            chunk_relevance_scores=crs,
        )

    @staticmethod
    def calculate_scores(
        memory: MemoryItem,
        compare_to: Embedding,
        strategy: Literal["summary", "weighted"] = "summary",
    ) -> tuple[float, float, list[float]]:
        """
        Calculates similarity between given embedding and all embeddings of the memory

        Returns:
            float: the aggregate (max) relevance score of the memory
            float: the relevance score of the memory summary
            list: the relevance scores of the memory chunks
        """
        base_embedding = memory.e_weighted if strategy == "weighted" else memory.e_summary
        summary_relevance_score = np.dot(base_embedding, compare_to)
        chunk_relevance_scores = np.dot(memory.e_chunks, compare_to).tolist()
        logger.debug(f"Relevance of summary: {summary_relevance_score}")
        logger.debug(f"Relevance of chunks: {chunk_relevance_scores}")

        relevance_scores = [summary_relevance_score, *chunk_relevance_scores]
        logger.debug(f"Relevance scores: {relevance_scores}")
        return max(relevance_scores), summary_relevance_score, chunk_relevance_scores

    @property
    def score(self) -> float:
        """The aggregate relevance score of the memory item for the given query"""
        return max([self.summary_relevance_score, *self.chunk_relevance_scores])

    @property
    def most_relevant_chunk(self) -> tuple[str, float]:
        """The most relevant chunk of the memory item + its score for the given query"""
        i_relmax = np.argmax(self.chunk_relevance_scores)
        return self.memory_item.chunks[i_relmax], self.chunk_relevance_scores[i_relmax]

    def __str__(self):
        return (
            f"{self.memory_item.summary} ({self.summary_relevance_score}) "
            f"{self.chunk_relevance_scores}"
        )
