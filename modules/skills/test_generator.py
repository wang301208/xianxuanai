from __future__ import annotations

"""LLM-assisted pytest generation for synthesized skills.

This module is intentionally opt-in: when no LLM client/model is configured it
falls back to minimal smoke tests.
"""

import json
import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Mapping, Optional, Sequence

from .registry import SkillSpec

logger = logging.getLogger(__name__)


DEFAULT_SMOKE_TEST_TEMPLATE = dedent(
    '''
    """Smoke tests for {name} skill (auto-generated)."""

    from __future__ import annotations

    import asyncio
    import inspect
    from typing import Any, Dict

    import pytest

    from {module_import} import handle


    def _invoke(payload: Dict[str, Any], context: Dict[str, Any] | None = None) -> Any:
        result = handle(payload, context=context)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result


    def test_handle_smoke():
        result = _invoke({{"sample": "value"}})
        assert isinstance(result, dict)
    '''
).strip()


@dataclass
class SkillTestGenerationConfig:
    model: str | None = None
    request_timeout: float | None = 30.0
    max_prompt_chars: int = 10_000
    max_output_chars: int = 20_000


@dataclass
class SkillTestGenerationResult:
    tests_source: str
    used_llm: bool


class SkillTestGenerator:
    """Generate pytest code for a skill handler using an LLM when configured."""

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        config: SkillTestGenerationConfig | None = None,
        smoke_test_template: str = DEFAULT_SMOKE_TEST_TEMPLATE,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or SkillTestGenerationConfig()
        self.smoke_test_template = smoke_test_template

    def generate(
        self,
        spec: SkillSpec,
        *,
        module_import: str,
        references: Any | None = None,
        few_shot_examples: Any | None = None,
    ) -> SkillTestGenerationResult:
        tests_source: Optional[str] = None
        used_llm = False

        if self._can_use_llm():
            try:
                tests_source = self._invoke_llm(
                    spec,
                    module_import=module_import,
                    references=references,
                    few_shot_examples=few_shot_examples,
                )
                used_llm = True
            except Exception as err:  # pragma: no cover - depends on external services
                logger.warning("LLM test generation failed for %s: %s", spec.name, err)
                tests_source = None

        if not tests_source:
            tests_source = self._render_smoke_tests(spec, module_import=module_import)

        return SkillTestGenerationResult(tests_source=str(tests_source), used_llm=used_llm)

    # ------------------------------------------------------------------
    def _invoke_llm(
        self,
        spec: SkillSpec,
        *,
        module_import: str,
        references: Any | None,
        few_shot_examples: Any | None,
    ) -> str:
        if not self.llm_client:
            raise RuntimeError("LLM client not configured")
        if not self.config.model:
            raise RuntimeError("LLM model not configured")

        prompt = self._build_prompt(
            spec,
            module_import=module_import,
            references=references,
            few_shot_examples=few_shot_examples,
        )

        response = self.llm_client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You generate pytest tests for a Python function. Output code only.",
                },
                {"role": "user", "content": prompt},
            ],
            timeout=self.config.request_timeout,
        )

        choice = getattr(response, "choices", [None])[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if not content:
            raise RuntimeError("LLM response did not contain content")

        text = str(content).strip()
        parts = text.split("```")
        extracted = None
        for block in parts:
            if block.strip().startswith("python"):
                extracted = "\n".join(block.splitlines()[1:])
                break
        tests_source = (extracted or text).strip()

        max_chars = int(self.config.max_output_chars)
        if max_chars > 0 and len(tests_source) > max_chars:
            tests_source = tests_source[:max_chars].rstrip() + "..."
        return tests_source

    def _build_prompt(
        self,
        spec: SkillSpec,
        *,
        module_import: str,
        references: Any | None,
        few_shot_examples: Any | None,
    ) -> str:
        input_schema = json.dumps(spec.input_schema, ensure_ascii=False, indent=2)
        output_schema = json.dumps(spec.output_schema, ensure_ascii=False, indent=2)
        prompt = dedent(
            f"""
            Write a standalone pytest module to test `{module_import}.handle`.

            Requirements:
            - Import: `from {module_import} import handle`
            - The handler may be sync or async; include a helper that supports both (asyncio.run).
            - Use only pytest + standard library. No network, no filesystem writes.
            - Prefer deterministic tests. Avoid Hypothesis.
            - Add at least 3 tests (happy path + edge cases).

            Skill name: {spec.name}
            Description: {spec.description}
            Execution mode: {spec.execution_mode}
            Input schema (JSON): {input_schema}
            Output schema (JSON): {output_schema}
            """
        ).strip()

        rendered_refs = self._render_blob(references, label="Reference material", max_chars=6000)
        if rendered_refs:
            prompt += "\n\n" + rendered_refs

        rendered_examples = self._render_blob(few_shot_examples, label="Few-shot examples / pseudocode", max_chars=6000)
        if rendered_examples:
            prompt += "\n\n" + rendered_examples

        prompt += "\n\nOutput ONLY the pytest code (prefer ```python fenced block)."
        max_chars = int(self.config.max_prompt_chars)
        if max_chars > 0 and len(prompt) > max_chars:
            prompt = prompt[:max_chars].rstrip() + "..."
        return prompt

    def _render_smoke_tests(self, spec: SkillSpec, *, module_import: str) -> str:
        return self.smoke_test_template.format(name=spec.name, module_import=module_import).strip()

    def _render_blob(self, value: Any | None, *, label: str, max_chars: int) -> str:
        if value is None:
            return ""
        text = ""
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, Mapping):
            text = json.dumps(value, ensure_ascii=False, indent=2)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            parts: list[str] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    parts.append(item.strip())
                elif isinstance(item, Mapping):
                    parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            text = "\n".join(p for p in parts if p).strip()
        else:
            text = str(value).strip()

        if not text:
            return ""
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        return f"{label}:\n{text}"

    def _can_use_llm(self) -> bool:
        return self.llm_client is not None and bool(self.config.model)


__all__ = ["SkillTestGenerator", "SkillTestGenerationConfig", "SkillTestGenerationResult"]

