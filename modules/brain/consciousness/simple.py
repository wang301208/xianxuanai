"""Simplified consciousness model components.

This module implements a tiny collection of classes that mimic
fundamental pieces of a cognitive architecture.  They are intentionally
light‑weight and only provide behaviour required by the unit tests:

``GlobalWorkspace``
    Records broadcasts of salient information.
``AttentionController``
    Evaluates whether a piece of information is salient.
``FeatureBinding``
    Placeholder that returns the information unchanged.
``ConsciousnessModel``
    Integrates the above components and exposes ``conscious_access``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Dict


class GlobalWorkspace:
    """Minimal global workspace storing broadcasted messages."""

    def __init__(self) -> None:
        self.broadcasts: List[Any] = []

    def broadcast(self, information: Any) -> None:
        """Store *information* as having been broadcast."""
        self.broadcasts.append(information)


class AttentionController:
    """Determine whether information is salient."""

    def is_salient(self, information: Dict[str, Any]) -> bool:
        """Return ``True`` if the ``information`` is marked salient."""
        return bool(information.get("is_salient"))


class FeatureBinding:
    """Bind features of the information together.

    For this simplified model the binding operation simply returns the
    information unchanged.  The class exists to show how additional
    processing stages might be chained in a more complete system.
    """

    def bind(self, information: Dict[str, Any]) -> Dict[str, Any]:
        return information


@dataclass
class ConsciousnessModel:
    """Toy model combining workspace, attention and feature binding."""

    workspace: GlobalWorkspace = field(default_factory=GlobalWorkspace)
    attention: AttentionController = field(default_factory=AttentionController)
    binding: FeatureBinding = field(default_factory=FeatureBinding)

    def conscious_access(self, information: Dict[str, Any]) -> bool:
        """Process ``information`` and broadcast if it is salient.

        Parameters
        ----------
        information:
            Dictionary that should contain an ``"is_salient"`` boolean flag
            alongside any payload under other keys.

        Returns
        -------
        bool
            ``True`` if the information was broadcast, otherwise ``False``.
        """

        bound = self.binding.bind(information)
        if self.attention.is_salient(bound):
            self.workspace.broadcast(bound)
            return True
        return False


__all__ = [
    "GlobalWorkspace",
    "AttentionController",
    "FeatureBinding",
    "ConsciousnessModel",
]
