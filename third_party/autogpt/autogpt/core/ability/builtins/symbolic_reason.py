"""Ability for delegating structured logic proofs to the symbolic reasoner."""

from __future__ import annotations

import json
import logging
from typing import ClassVar, Iterable, List, Sequence

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema

from backend.reasoning import (
    LogicRule,
    SymbolicReasoner,
    require_symbolic_reasoner,
)


class SymbolicReason(Ability):
    """Prove logical goals using the shared symbolic reasoning engine."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.SymbolicReason",
        ),
        performance_hint=0.2,
    )

    description: ClassVar[str] = (
        "Use the symbolic reasoning engine to prove whether the supplied goal follows "
        "from the provided premises and optional rules."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "goal": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Logical goal/fact to prove.",
        ),
        "premises": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            description="List of known facts treated as premises for this query.",
            items=JSONSchema(type=JSONSchema.Type.STRING),
            default=[],
        ),
        "rules": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            description="Optional Horn-clause rules expressed as objects with 'head' and 'body'.",
            items=JSONSchema(
                type=JSONSchema.Type.OBJECT,
                properties={
                    "head": JSONSchema(type=JSONSchema.Type.STRING),
                    "body": JSONSchema(
                        type=JSONSchema.Type.ARRAY,
                        items=JSONSchema(type=JSONSchema.Type.STRING),
                    ),
                    "description": JSONSchema(type=JSONSchema.Type.STRING),
                },
            ),
            default=[],
        ),
        "constraints": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            description="Facts that must hold once the proof succeeds.",
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
        goal: str,
        *,
        premises: Sequence[str] | None = None,
        rules: Sequence[dict] | None = None,
        constraints: Sequence[str] | None = None,
    ) -> AbilityResult:
        goal = goal.strip()
        if not goal:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"goal": goal},
                success=False,
                message="Goal must be a non-empty string.",
            )

        reasoner = self._ensure_reasoner()
        parsed_rules = self._parse_rules(rules or [])
        holds, proof = reasoner.prove(
            goal,
            premises=premises or [],
            rules=parsed_rules,
        )
        constraints = list(constraints or [])
        if not holds:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={
                    "goal": goal,
                    "premises": list(premises or []),
                    "rules": list(rules or []),
                },
                success=False,
                message="Goal could not be proven from the supplied premises.",
            )

        constraint_result = None
        missing: Iterable[str] = []
        if constraints:
            satisfied, missing = reasoner.validate_constraints(
                constraints,
                premises=premises or [],
                rules=parsed_rules,
            )
            constraint_result = satisfied

        payload = {
            "goal": goal,
            "premises": list(premises or []),
            "proof": proof,
            "constraints": constraints,
            "constraints_satisfied": constraint_result,
            "missing_constraints": list(missing),
        }

        knowledge = Knowledge(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            content_type=ContentType.TEXT,
            content_metadata={"source": "symbolic_reason"},
        )

        summary = f"Goal '{goal}' proven." if holds else f"Failed to prove '{goal}'."
        if constraints:
            if constraint_result:
                summary += " All constraints satisfied."
            else:
                summary += " Missing constraints: " + ", ".join(missing)

        return AbilityResult(
            ability_name=self.name(),
            ability_args={
                "goal": goal,
                "premises": list(premises or []),
                "rules": list(rules or []),
                "constraints": constraints,
            },
            success=True,
            message=summary,
            new_knowledge=knowledge,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_rules(self, rules: Sequence[dict]) -> List[LogicRule]:
        parsed: List[LogicRule] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            head = str(rule.get("head", "")).strip()
            body = rule.get("body", [])
            if not head or not isinstance(body, (list, tuple)):
                continue
            atoms = tuple(str(atom).strip() for atom in body if str(atom).strip())
            if not atoms:
                continue
            description = str(rule.get("description", ""))
            parsed.append(LogicRule(head=head, body=atoms, description=description))
        return parsed

    def _ensure_reasoner(self) -> SymbolicReasoner:
        try:
            return require_symbolic_reasoner()
        except RuntimeError:
            self._logger.debug("Symbolic reasoner not initialised; falling back to empty graph.")
            return SymbolicReasoner({})
