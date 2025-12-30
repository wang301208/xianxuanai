"""Ability that validates statements using commonsense knowledge."""

from __future__ import annotations

import json
import logging
from typing import ClassVar, Sequence

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema

from backend.reasoning import (
    CommonsenseKnowledge,
    CommonsenseValidator,
    require_commonsense_validator,
)


class CommonsenseValidate(Ability):
    """Check statements against a commonsense knowledge base."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.CommonsenseValidate",
        ),
        performance_hint=0.1,
    )

    description: ClassVar[str] = (
        "Consult the commonsense knowledge base to verify whether a subject, relation, "
        "object triple is plausible, contradictory, or unknown."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "subject": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Subject/entity of the statement.",
        ),
        "relation": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Relation or predicate between subject and object.",
        ),
        "object": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Object/target of the relation.",
        ),
        "context": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            description="Optional contextual hints that may assist validation.",
            items=JSONSchema(type=JSONSchema.Type.STRING),
            default=[],
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
        subject: str,
        *,
        relation: str,
        object: str,
        context: Sequence[str] | None = None,
    ) -> AbilityResult:
        subject = subject.strip()
        relation = relation.strip()
        object = object.strip()
        if not subject or not relation or not object:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"subject": subject, "relation": relation, "object": object},
                success=False,
                message="Subject, relation, and object must be non-empty strings.",
            )

        validator = self._ensure_validator()
        judgement = validator.validate(subject, relation, object, context=context or [])
        payload = {
            "subject": subject,
            "relation": relation,
            "object": object,
            "status": judgement.status,
            "message": judgement.message,
            "evidence": judgement.evidence,
            "suggestions": judgement.suggestions,
        }

        knowledge = Knowledge(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            content_type=ContentType.TEXT,
            content_metadata={"source": "commonsense_validate"},
        )

        return AbilityResult(
            ability_name=self.name(),
            ability_args={
                "subject": subject,
                "relation": relation,
                "object": object,
                "context": list(context or []),
            },
            success=True,
            message=judgement.message,
            new_knowledge=knowledge,
        )

    def _ensure_validator(self) -> CommonsenseValidator:
        try:
            return require_commonsense_validator()
        except RuntimeError:
            self._logger.debug("Commonsense validator not initialised; creating ephemeral instance.")
            return CommonsenseValidator(CommonsenseKnowledge())
