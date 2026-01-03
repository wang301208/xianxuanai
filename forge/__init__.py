"""Compatibility package exposing the bundled Forge SDK as `forge`.

The upstream AutoGPT code imports `forge.*` (e.g. `forge.sdk.model.Task`). In this
repository the Forge sources live under `backend/forge/forge`. This lightweight
shim extends the package search path so those imports work without installing an
external `forge` distribution.
"""

from __future__ import annotations

from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_BUNDLED_FORGE = (
    Path(__file__).resolve().parents[1] / "backend" / "forge" / "forge"
).resolve()
if _BUNDLED_FORGE.exists():
    __path__.append(str(_BUNDLED_FORGE))  # type: ignore[name-defined]

