"""Minimal Org Charter package used by blueprint tooling.

The upstream project expects an `org_charter` package that provides blueprint
schema and (optionally) filesystem watchers. This repository keeps the JSON
schema under `modules/schemas/agent_blueprint.yaml`, so this package acts as a
small compatibility layer.
"""

from __future__ import annotations

__all__ = ["io"]

