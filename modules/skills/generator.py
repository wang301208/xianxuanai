from __future__ import annotations

"""Utilities to synthesize skill handler code from specs."""

import json
import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Mapping, Optional, Sequence

from .registry import SkillSpec

logger = logging.getLogger(__name__)


DEFAULT_HANDLER_TEMPLATE = dedent(
    '''
    """Auto-generated skill handler for {name}."""

    from __future__ import annotations
    from typing import Any, Dict


    def handle(payload: Dict[str, Any], *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """{description}

        Execution mode: {execution_mode}
        Input schema: {input_schema}
        Output schema: {output_schema}
        """

        # TODO: replace with concrete logic
        return {
            "status": "not_implemented",
            "received": payload,
        }
    '''
).strip()

DEFAULT_RPC_HANDLER_TEMPLATE = dedent(
    '''
    """Auto-generated RPC skill handler for {name}."""

    from __future__ import annotations
    from typing import Any, Dict


    async def handle(payload: Dict[str, Any], *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """{description}

        Execution mode: {execution_mode}
        Input schema: {input_schema}
        Output schema: {output_schema}
        """

        # TODO: integrate with the remote execution environment
        return {
            "status": "not_implemented",
            "received": payload,
        }
    '''
).strip()

DEFAULT_TEST_TEMPLATE = dedent(
    '''
    """Tests for {name} skill (auto-generated)."""

    from __future__ import annotations

    import asyncio
    import inspect
    import pytest

    try:
        from {module_import} import handle
    except Exception:  # pragma: no cover - import scaffolding only
        handle = None


    def test_handle_is_defined():
        assert handle is not None


    def _invoke(payload):
        result = handle(payload)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result


    def test_handle_placeholder_behavior():
        payload = {{"sample": "value"}}
        if handle is None:
            pytest.skip("Handler import failed; placeholder scaffolding only")

        result = _invoke(payload)

        assert isinstance(result, dict)
        assert result.get("status") == "not_implemented"
        assert result.get("received") == payload
    '''
).strip()


@dataclass
class SkillGenerationConfig:
    """Configuration for code generation requests."""

    model: str | None = None
    request_timeout: float | None = 30.0
    enable_tests: bool = True


@dataclass
class SkillGenerationResult:
    """Outcome of generating handler/tests for a skill."""

    handler_source: str
    tests_source: Optional[str]
    used_llm: bool


class SkillCodeGenerator:
    """Generate handler code using an LLM or a template library."""

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        config: SkillGenerationConfig | None = None,
        template_library: Mapping[str, str] | None = None,
        default_test_template: str | None = DEFAULT_TEST_TEMPLATE,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or SkillGenerationConfig()
        self.template_library = dict(
            template_library
            or {
                "rpc": DEFAULT_RPC_HANDLER_TEMPLATE,
                "local": DEFAULT_HANDLER_TEMPLATE,
                "default": DEFAULT_HANDLER_TEMPLATE,
            }
        )
        self.default_test_template = default_test_template

    def generate(
        self,
        spec: SkillSpec,
        *,
        module_import: str | None = None,
        include_tests: bool = True,
        references: Any | None = None,
        few_shot_examples: Any | None = None,
    ) -> SkillGenerationResult:
        """Generate handler code (and optionally tests) for ``spec``.

        Falls back to template-based scaffolding when the LLM request fails or
        is not configured.
        """

        handler_source: Optional[str] = None
        tests_source: Optional[str] = None
        used_llm = False

        if self._can_use_llm():
            try:
                handler_source, tests_source = self._invoke_llm(
                    spec,
                    include_tests=include_tests,
                    references=references,
                    few_shot_examples=few_shot_examples,
                )
                used_llm = True
            except Exception as err:  # pragma: no cover - depends on external services
                logger.warning(
                    "LLM generation failed for skill %s: %s. Falling back to template.",
                    spec.name,
                    err,
                )

        if handler_source is None:
            handler_source = self._render_template(spec)
        if include_tests and tests_source is None and self.default_test_template:
            tests_source = self._render_tests(spec, module_import=module_import)

        return SkillGenerationResult(
            handler_source=handler_source,
            tests_source=tests_source if include_tests else None,
            used_llm=used_llm,
        )

    # ------------------------------------------------------------------
    def _invoke_llm(
        self,
        spec: SkillSpec,
        *,
        include_tests: bool = True,
        references: Any | None = None,
        few_shot_examples: Any | None = None,
    ) -> tuple[str, Optional[str]]:
        if not self.llm_client:
            raise RuntimeError("LLM client not configured")
        if not self.config.model:
            raise RuntimeError("LLM model not configured")

        prompt = self._build_prompt(
            spec,
            include_tests=include_tests,
            references=references,
            few_shot_examples=few_shot_examples,
        )
        logger.info("Requesting handler generation for '%s' with model %s", spec.name, self.config.model)

        response = self.llm_client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a code generator that produces complete Python skill handlers.",
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

        parts = content.split("```")
        extracted = None
        for block in parts:
            if block.strip().startswith("python"):
                extracted = "\n".join(block.splitlines()[1:])
                break
        handler_source = extracted.strip() if extracted else content.strip()

        tests_source = None
        if include_tests and "# tests" in content.lower():
            tests_source = content

        return handler_source, tests_source

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        spec: SkillSpec,
        *,
        include_tests: bool,
        references: Any | None = None,
        few_shot_examples: Any | None = None,
    ) -> str:
        input_schema = json.dumps(spec.input_schema, ensure_ascii=False, indent=2)
        output_schema = json.dumps(spec.output_schema, ensure_ascii=False, indent=2)
        prompt = dedent(
            f"""
            Write a Python skill handler named '{spec.name}'.
            Description: {spec.description}
            Execution mode: {spec.execution_mode}
            Input schema: {input_schema}
            Output schema: {output_schema}
            The handler should expose a `handle(payload, *, context=None)` function.
            Return a JSON-serializable dict payload. Keep the implementation minimal and safe.
            """
        ).strip()
        rendered_refs = self._render_references(references)
        if rendered_refs:
            prompt += (
                "\n\nReference material (may be partial, do not copy verbatim):\n"
                + rendered_refs
                + "\n\nUse the references as guidance only. Prefer an original implementation; "
                "if you incorporate any copyrighted text/code, ensure it is permitted and add attribution."
            )
        rendered_examples = self._render_few_shot_examples(few_shot_examples)
        if rendered_examples:
            prompt += (
                "\n\nFew-shot examples / pseudocode (use these to match behavior):\n"
                + rendered_examples
                + "\n\nFollow the examples but keep the implementation original."
            )
        if include_tests and self.config.enable_tests:
            prompt += "\n\nAlso propose lightweight pytest scaffolding validating the placeholder behavior."
        return prompt

    def _render_few_shot_examples(self, examples: Any | None, *, max_chars: int = 6000) -> str:
        """Render few-shot examples in a prompt-friendly way."""

        if examples is None:
            return ""

        if isinstance(examples, Mapping):
            signatures = examples.get("signatures") or examples.get("function_signatures")
            snippets = examples.get("pseudocode") or examples.get("snippets") or examples.get("code_blocks")
            io_examples = examples.get("examples") or examples.get("io_examples")

            chunks: list[str] = []
            if isinstance(signatures, Sequence) and not isinstance(signatures, (bytes, bytearray, str)):
                lines = [str(s).strip() for s in signatures if str(s).strip()]
                if lines:
                    chunks.append("Signatures:\n" + "\n".join(f"- {line}" for line in lines[:10]))
            if isinstance(snippets, Sequence) and not isinstance(snippets, (bytes, bytearray, str)):
                blocks = [str(s).strip() for s in snippets if str(s).strip()]
                if blocks:
                    rendered = "\n\n".join(f"[snippet {i}]\n{block}" for i, block in enumerate(blocks[:5], start=1))
                    chunks.append("Pseudocode / code blocks:\n" + rendered)
            if io_examples is not None:
                chunks.append("I/O examples:\n" + self._render_references(io_examples, max_chars=max_chars))

            text = "\n\n".join(chunk for chunk in chunks if chunk).strip()
            if text:
                if max_chars > 0 and len(text) > max_chars:
                    return text[:max_chars].rstrip() + "..."
                return text

        return self._render_references(examples, max_chars=max_chars)

    def _render_references(self, references: Any | None, *, max_chars: int = 6000) -> str:
        if references is None:
            return ""
        if isinstance(references, str):
            text = references.strip()
        elif isinstance(references, Mapping):
            text = json.dumps(references, ensure_ascii=False, indent=2)
        elif isinstance(references, Sequence) and not isinstance(references, (bytes, bytearray, str)):
            chunks: list[str] = []
            for idx, item in enumerate(references, start=1):
                if item is None:
                    continue
                if isinstance(item, str):
                    chunks.append(f"[{idx}] {item.strip()}")
                    continue
                if isinstance(item, Mapping):
                    title = str(item.get("title") or item.get("name") or "").strip()
                    url = str(item.get("url") or item.get("source") or "").strip()
                    excerpt = str(item.get("excerpt") or item.get("snippet") or item.get("text") or "").strip()
                    header = " / ".join(part for part in (title, url) if part) or f"ref_{idx}"
                    body = excerpt or json.dumps(item, ensure_ascii=False)
                    chunks.append(f"[{idx}] {header}\n{body}".strip())
                    continue
                chunks.append(f"[{idx}] {str(item).strip()}")
            text = "\n\n".join(chunk for chunk in chunks if chunk).strip()
        else:
            text = str(references).strip()

        if not text:
            return ""
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text

    # ------------------------------------------------------------------
    def _render_template(self, spec: SkillSpec) -> str:
        input_schema = json.dumps(spec.input_schema, ensure_ascii=False)
        output_schema = json.dumps(spec.output_schema, ensure_ascii=False)
        template = self.template_library.get(spec.execution_mode) or self.template_library.get("default")
        if not template:
            logger.error("No template available for execution mode '%s'", spec.execution_mode)
            template = DEFAULT_HANDLER_TEMPLATE

        return template.format(
            name=spec.name,
            description=spec.description,
            execution_mode=spec.execution_mode,
            input_schema=input_schema,
            output_schema=output_schema,
        ).strip()

    def _render_tests(self, spec: SkillSpec, *, module_import: Optional[str]) -> str:
        if not self.default_test_template:
            return ""
        module_path = module_import or "<module_path_placeholder>"
        return self.default_test_template.format(name=spec.name, module_import=module_path)

    def _can_use_llm(self) -> bool:
        return self.llm_client is not None and bool(self.config.model)
