"""Vendored AutoGPT package integration.

This module exposes the upstream AutoGPT source tree that we vendor in
``third_party/autogpt``. Downstream code should import symbols via the
``third_party.autogpt.autogpt`` package so the dependency location is
explicit and future upgrades remain isolated from local modules.
"""
from importlib import import_module as _import_module

# Re-export the vendored AutoGPT package under ``third_party.autogpt`` so that
# importing ``third_party.autogpt.autogpt`` behaves the same as importing the
# upstream package directly.
autogpt = _import_module(".autogpt", __name__)

__all__ = ["autogpt"]
