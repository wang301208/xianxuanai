"""Best-effort license detection for external code corpora.

This is a lightweight heuristic intended for *risk signalling* and policy gates
when an agent ingests external repositories. It is not a legal determination.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


_LICENSE_FILENAMES: Sequence[str] = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "COPYING",
    "COPYING.txt",
    "COPYING.md",
    "NOTICE",
    "NOTICE.txt",
    "NOTICE.md",
)


@dataclass(frozen=True)
class LicenseDetection:
    spdx: Optional[str]
    confidence: float
    file: Optional[str] = None
    copyleft: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "spdx": self.spdx,
            "confidence": float(self.confidence),
            "file": self.file,
            "copyleft": bool(self.copyleft),
        }


_RX_FLAGS = re.IGNORECASE | re.MULTILINE


def _detect_spdx(text: str) -> LicenseDetection:
    raw = str(text or "")

    # GNU family.
    if re.search(r"GNU\s+AFFERO\s+GENERAL\s+PUBLIC\s+LICENSE", raw, flags=_RX_FLAGS):
        if re.search(r"version\s+3", raw, flags=_RX_FLAGS):
            return LicenseDetection("AGPL-3.0", 0.95, copyleft=True)
        return LicenseDetection("AGPL", 0.7, copyleft=True)

    if re.search(r"GNU\s+LESSER\s+GENERAL\s+PUBLIC\s+LICENSE", raw, flags=_RX_FLAGS):
        if re.search(r"version\s+3", raw, flags=_RX_FLAGS):
            return LicenseDetection("LGPL-3.0", 0.95, copyleft=True)
        if re.search(r"version\s+2\.1", raw, flags=_RX_FLAGS):
            return LicenseDetection("LGPL-2.1", 0.95, copyleft=True)
        return LicenseDetection("LGPL", 0.7, copyleft=True)

    if re.search(r"GNU\s+GENERAL\s+PUBLIC\s+LICENSE", raw, flags=_RX_FLAGS):
        if re.search(r"version\s+3", raw, flags=_RX_FLAGS):
            return LicenseDetection("GPL-3.0", 0.95, copyleft=True)
        if re.search(r"version\s+2", raw, flags=_RX_FLAGS):
            return LicenseDetection("GPL-2.0", 0.95, copyleft=True)
        return LicenseDetection("GPL", 0.7, copyleft=True)

    # Apache.
    if re.search(r"Apache\s+License", raw, flags=_RX_FLAGS) and re.search(r"Version\s+2\.0", raw, flags=_RX_FLAGS):
        return LicenseDetection("Apache-2.0", 0.9, copyleft=False)

    # MPL.
    if re.search(r"Mozilla\s+Public\s+License", raw, flags=_RX_FLAGS) and re.search(r"Version\s+2\.0", raw, flags=_RX_FLAGS):
        return LicenseDetection("MPL-2.0", 0.9, copyleft=True)

    # MIT.
    if re.search(r"Permission\s+is\s+hereby\s+granted,\s+free\s+of\s+charge", raw, flags=_RX_FLAGS):
        return LicenseDetection("MIT", 0.9, copyleft=False)

    # BSD.
    if re.search(r"Redistribution\s+and\s+use\s+in\s+source\s+and\s+binary\s+forms", raw, flags=_RX_FLAGS):
        if re.search(r"Neither\s+the\s+name\s+of", raw, flags=_RX_FLAGS):
            return LicenseDetection("BSD-3-Clause", 0.85, copyleft=False)
        return LicenseDetection("BSD-2-Clause", 0.75, copyleft=False)

    # ISC.
    if re.search(r"Permission\s+to\s+use,\s+copy,\s+modify,\s+and/or\s+distribute", raw, flags=_RX_FLAGS) and re.search(
        r"with\s+or\s+without\s+fee", raw, flags=_RX_FLAGS
    ):
        return LicenseDetection("ISC", 0.85, copyleft=False)

    # Unlicense / public domain.
    if re.search(r"free\s+and\s+unencumbered\s+software\s+released\s+into\s+the\s+public\s+domain", raw, flags=_RX_FLAGS):
        return LicenseDetection("Unlicense", 0.9, copyleft=False)

    return LicenseDetection(None, 0.0, copyleft=False)


def detect_repo_license(
    repo_root: Path,
    *,
    candidate_filenames: Sequence[str] = _LICENSE_FILENAMES,
    max_chars: int = 200_000,
) -> Dict[str, Any]:
    """Detect a repository license from common license files."""

    repo_root = Path(repo_root).resolve()
    best = LicenseDetection(None, 0.0, file=None, copyleft=False)

    for name in candidate_filenames:
        path = repo_root / name
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        detected = _detect_spdx(text)
        detected = LicenseDetection(
            detected.spdx,
            detected.confidence,
            file=str(path.name),
            copyleft=detected.copyleft,
        )
        if detected.confidence > best.confidence:
            best = detected
        if best.confidence >= 0.95:
            break

    return best.as_dict()


def normalize_spdx(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    return raw


def normalize_spdx_list(values: Any) -> set[str]:
    items: set[str] = set()
    if values is None:
        return items
    if isinstance(values, str):
        parts = [p.strip() for p in values.replace(";", ",").split(",")]
        for p in parts:
            if p:
                items.add(p)
        return items
    if isinstance(values, (list, tuple, set)):
        for item in values:
            spdx = normalize_spdx(item)
            if spdx:
                items.add(spdx)
        return items
    spdx = normalize_spdx(values)
    if spdx:
        items.add(spdx)
    return items


__all__ = ["detect_repo_license", "normalize_spdx", "normalize_spdx_list"]
