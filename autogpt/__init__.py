"""Compatibility package exposing `third_party.autogpt.autogpt` as `autogpt`.

The vendored AutoGPT code lives under `third_party/autogpt/autogpt`, but some
modules import it as `autogpt.*`. This package bridges that gap by extending
the package search path.
"""

from __future__ import annotations

from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_VENDORED_AUTOGPT = (Path(__file__).resolve().parents[1] / "third_party" / "autogpt" / "autogpt").resolve()
if _VENDORED_AUTOGPT.exists():
    __path__.append(str(_VENDORED_AUTOGPT))  # type: ignore[name-defined]

