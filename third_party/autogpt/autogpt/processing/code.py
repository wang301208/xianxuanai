import ast
from typing import Iterator

from autogpt.core.resource.model_providers.schema import ModelTokenizer

from .text import chunk_content

def chunk_code_by_structure(
    code: str, max_chunk_length: int, tokenizer: ModelTokenizer
) -> Iterator[tuple[str, int]]:
    """Split code into chunks following top-level structure.

    Falls back to generic token-based chunking when parsing fails or when a
    structure block exceeds ``max_chunk_length`` tokens.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        yield from chunk_content(code, max_chunk_length, tokenizer)
        return

    lines = code.splitlines()
    spans: list[tuple[int, int]] = []
    last_end = 0
    for node in tree.body:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        if start is None or end is None:
            continue
        if start - 1 > last_end:
            spans.append((last_end, start - 1))
        spans.append((start - 1, end))
        last_end = end
    if last_end < len(lines):
        spans.append((last_end, len(lines)))

    for start, end in spans:
        chunk = "\n".join(lines[start:end])
        length = len(tokenizer.encode(chunk))
        if length <= max_chunk_length:
            yield chunk, length
        else:
            yield from chunk_content(chunk, max_chunk_length, tokenizer)
