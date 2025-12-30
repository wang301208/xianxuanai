"""Compatibility package re-exporting `backend.reasoning` as `reasoning`.

Some third-party integrations import `reasoning.*` directly, while the actual
implementation in this repo lives under `backend/reasoning`. This package
bridges that gap by extending the package search path.
"""

from __future__ import annotations

from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_BACKEND_REASONING = (Path(__file__).resolve().parents[1] / "backend" / "reasoning").resolve()
if _BACKEND_REASONING.exists():
    __path__.append(str(_BACKEND_REASONING))  # type: ignore[name-defined]

