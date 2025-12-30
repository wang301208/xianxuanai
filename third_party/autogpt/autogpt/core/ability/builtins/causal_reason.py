"""Ability leveraging the causal reasoner for counterfactual queries."""

from __future__ import annotations

import json
import logging
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema

from backend.reasoning import (
    KnowledgeGraphCausalReasoner,
    require_causal_reasoner,
)


class CausalReason(Ability):
    """Inspect causal relations and counterfactuals using the causal graph."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.CausalReason",
        ),
        performance_hint=0.15,
    )

    description: ClassVar[str] = (
        "Check whether one event causes another, surface intermediate steps, and "
        "simulate what-if interventions using the causal knowledge graph."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "cause": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Candidate cause to analyse.",
        ),
        "effect": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Effect or outcome of interest.",
        ),
        "depth": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="Depth for exploring downstream effects.",
            default=2,
        ),
        "explain_intervention": JSONSchema(
            type=JSONSchema.Type.BOOLEAN,
            description="Whether to include intervention analysis in the response.",
            default=True,
        ),
    }

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
    ) -> None:
        self._logger = logger
        self._configuration = configuration

    async def __call__(
        self,
        cause: str,
        *,
        effect: str,
        depth: int = 2,
        explain_intervention: bool = True,
    ) -> AbilityResult:
        cause = cause.strip()
        effect = effect.strip()
        if not cause or not effect:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"cause": cause, "effect": effect},
                success=False,
                message="Cause and effect must be non-empty strings.",
            )

        reasoner = self._ensure_reasoner()
        exists, path = reasoner.check_causality(cause, effect)
        predictions = reasoner.predict_effects(cause, depth=depth)
        report = {
            "cause": cause,
            "effect": effect,
            "causal_path": list(path),
            "effects": predictions,
        }
        summary_lines = []
        if exists:
            summary_lines.append(f"Causal chain: {' -> '.join(path)}")
        else:
            summary_lines.append(f"No causal chain from {cause} to {effect}.")

        if explain_intervention:
            counterfactual = reasoner.intervention(cause, effect)
            report["intervention"] = counterfactual
            summary_lines.append(counterfactual)

        knowledge = Knowledge(
            content=json.dumps(report, ensure_ascii=False, indent=2),
            content_type=ContentType.TEXT,
            content_metadata={"source": "causal_reason"},
        )

        return AbilityResult(
            ability_name=self.name(),
            ability_args={
                "cause": cause,
                "effect": effect,
                "depth": depth,
                "explain_intervention": explain_intervention,
            },
            success=True,
            message="\n".join(summary_lines),
            new_knowledge=knowledge,
        )

    def _ensure_reasoner(self) -> KnowledgeGraphCausalReasoner:
        try:
            return require_causal_reasoner()
        except RuntimeError:
            self._logger.debug("Causal reasoner not initialised; using empty graph fallback.")
            return KnowledgeGraphCausalReasoner({})
