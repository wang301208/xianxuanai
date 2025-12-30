"""Common-sense reasoning using ConceptNet."""

from __future__ import annotations

from typing import List, Tuple
import json
from urllib.request import urlopen
from urllib.parse import quote
from urllib.error import URLError


Conclusion = Tuple[str, float]


class CommonSenseReasoner:
    """Thin wrapper around the ConceptNet API.

    Parameters
    ----------
    enabled:
        When ``False`` the reasoner short-circuits and returns no conclusions.
    endpoint:
        Base URL of the ConceptNet API. Can be overridden for testing.
    """

    def __init__(self, enabled: bool = True, endpoint: str = "https://api.conceptnet.io") -> None:
        self.enabled = enabled
        self.endpoint = endpoint.rstrip("/")

    # ------------------ internal helpers ------------------
    def _fetch_conceptnet_edges(self, concept: str) -> List[Conclusion]:
        """Retrieve ConceptNet edges for ``concept``.

        Returns a list of ``(text, weight)`` tuples. Failures are silenced and
        result in an empty list so that callers can handle unavailable
        connectivity gracefully.
        """

        url = f"{self.endpoint}/c/en/{quote(concept)}"
        try:
            with urlopen(url, timeout=5) as resp:  # type: ignore[call-arg]
                data = json.load(resp)
        except (URLError, OSError, ValueError):
            return []
        edges: List[Conclusion] = []
        for edge in data.get("edges", []):
            rel = edge.get("rel", {}).get("label")
            end = edge.get("end", {}).get("label")
            weight = float(edge.get("weight", 1.0))
            if rel and end:
                text = f"{concept} {rel} {end}"
                edges.append((text, weight))
        return edges

    # ------------------ public API ------------------
    def infer(self, text: str, limit: int = 5) -> List[Conclusion]:
        """Infer common-sense conclusions for ``text``.

        Parameters
        ----------
        text:
            Natural language concept or short phrase.
        limit:
            Maximum number of conclusions to return.

        Returns
        -------
        conclusions:
            List of ``(conclusion, confidence)`` tuples sorted by API order.
        """

        if not self.enabled:
            return []
        concept = text.lower().strip().replace(" ", "_")
        edges = self._fetch_conceptnet_edges(concept)
        return edges[:limit]


__all__ = ["CommonSenseReasoner"]
