"""Minimal `inflection` stub used by the vendored AutoGPT components.

The upstream AutoGPT code depends on the third-party `inflection` package for
simple string case conversions. This repository intentionally keeps runtime
dependencies lightweight; the helper functions implemented here cover the
subset of the API used within the repo.
"""

from __future__ import annotations

import re


def underscore(value: str) -> str:
    """Convert `CamelCase` or spaced text into `snake_case`."""

    text = str(value or "")
    text = re.sub(r"[\s-]+", "_", text.strip())
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"__+", "_", text)
    return text.lower().strip("_")


def camelize(value: str) -> str:
    """Convert `snake_case` into `CamelCase`."""

    text = str(value or "")
    parts = [p for p in re.split(r"[\s_-]+", text) if p]
    return "".join(part[:1].upper() + part[1:] for part in parts)


__all__ = ["underscore", "camelize"]

