from __future__ import annotations

"""API for exposing brain state metrics such as attention and memory hits.

This module provides a tiny FastAPI application that surfaces internal
information from the :class:`GlobalWorkspace` and offers a simple endpoint to
adjust attention weights at runtime.  A small helper ``record_memory_hit`` can
be used by memory backends to keep track of retrieval operations.
"""

from typing import List

try:  # pragma: no cover - exercised via tests mocking FastAPI absence
    from fastapi import FastAPI
except ModuleNotFoundError:  # pragma: no cover - handled in tests
    FastAPI = None  # type: ignore[assignment]

from .global_workspace import GlobalWorkspace, global_workspace


class BrainMetrics:
    """Container for lightweight brain-related metrics."""

    def __init__(self) -> None:
        self.memory_hits: int = 0

    def hit(self) -> None:
        """Record a memory retrieval."""
        self.memory_hits += 1


_metrics = BrainMetrics()


def record_memory_hit() -> None:
    """Increment the memory hit counter.

    Modules that retrieve information from memory can call this helper to
    expose their activity via the monitoring API.
    """

    _metrics.hit()


def get_memory_hits() -> int:
    """Return the cumulative number of recorded memory hits."""

    return _metrics.memory_hits


def create_app(workspace: GlobalWorkspace | None = None) -> FastAPI:
    """Create a FastAPI app exposing brain state metrics.

    Parameters
    ----------
    workspace:
        Optional custom :class:`GlobalWorkspace` instance.  If omitted the
        module level ``global_workspace`` is used.
    """

    workspace = workspace or global_workspace
    if FastAPI is None:  # pragma: no cover - exercised by dedicated tests
        raise RuntimeError(
            "FastAPI is required to create the brain monitoring API. "
            "Install the 'fastapi' package to use create_brain_app()."
        )

    app = FastAPI()

    @app.get("/brain/state")
    def state() -> dict[str, object]:
        """Return current attention weights and memory hit count."""

        return {
            "attention": dict(workspace._attention),
            "memory_hits": _metrics.memory_hits,
        }

    @app.post("/brain/attention/{module}")
    def set_attention(module: str, weights: List[float]) -> dict[str, object]:
        """Set attention ``weights`` for ``module``."""

        workspace._attention[module] = [float(w) for w in weights]
        return {"module": module, "attention": workspace._attention[module]}

    return app
