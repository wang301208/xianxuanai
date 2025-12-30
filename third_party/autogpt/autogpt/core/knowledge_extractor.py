from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Tuple


def summarize_text(text: str, max_length: int = 100) -> str:
    """Return a very small summary of ``text``.

    The implementation is intentionally lightweight and avoids external
    dependencies.  If the text is longer than ``max_length`` characters it is
    truncated and suffixed with an ellipsis.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def extract_structured_knowledge(text: str) -> List[Dict[str, str]]:
    """Extract simple entity-value pairs from ``text``.

    The extractor looks for statements of the form ``"X is Y"`` or
    ``"X has Y"`` and stores the information together with a timestamp and an
    initial version number.  This is obviously a very small scale extractor but
    is sufficient for unit tests and demonstrations.
    """
    pattern = re.compile(r"(?P<entity>[A-Za-z0-9_ ]+)\s+(?:is|are|has|have)\s+(?P<value>[^.]+)")
    knowledge: List[Dict[str, str]] = []
    for match in pattern.finditer(text):
        entity = match.group("entity").strip()
        value = match.group("value").strip()
        knowledge.append(
            {
                "entity": entity,
                "value": value,
                "timestamp": datetime.utcnow().isoformat(),
                "version": 1,
            }
        )
    return knowledge


def extract(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Convenience wrapper returning both the summary and the structured data."""
    return summarize_text(text), extract_structured_knowledge(text)

__all__ = ["summarize_text", "extract_structured_knowledge", "extract"]
