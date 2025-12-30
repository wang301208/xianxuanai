from __future__ import annotations

"""Program synthesis + pytest validation loop for skills.

This module is intentionally lightweight and opt-in. It provides a small
"generate -> test -> (retry)" loop that can be used by agents to verify new skill
implementations before persisting or registering them.
"""

import logging
import os
import inspect
import pprint
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .generator import SkillCodeGenerator
from .registry import SkillSpec
from .test_generator import SkillTestGenerator

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in (name or "").lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "skill"


def _truncate(text: str, *, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 1] + "…"


@dataclass(frozen=True)
class SkillExample:
    payload: Dict[str, Any]
    expected: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    match: str = "subset"  # "subset" or "exact"


@dataclass(frozen=True)
class SkillTDDConfig:
    max_attempts: int = 2
    pytest_timeout_s: float = 30.0
    keep_artifacts: bool = False
    disable_pytest_plugin_autoload: bool = True
    max_pytest_output_chars: int = 30_000


@dataclass(frozen=True)
class PytestRunResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    cmd: Tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    def combined_output(self, *, max_chars: int) -> str:
        output = (self.stdout or "") + ("\n" if self.stdout and self.stderr else "") + (self.stderr or "")
        return _truncate(output, max_chars=max_chars)


@dataclass(frozen=True)
class SkillTDDResult:
    passed: bool
    attempts: int
    handler_source: str
    tests_source: str
    pytest: PytestRunResult
    duration_s: float
    artifact_dir: Optional[str] = None


class SkillTDDPipeline:
    """Generate a skill handler, run pytest, and optionally retry."""

    def __init__(
        self,
        *,
        code_generator: SkillCodeGenerator | None = None,
        test_generator: SkillTestGenerator | None = None,
        config: SkillTDDConfig | None = None,
        repo_root: Path | str | None = None,
    ) -> None:
        self._generator = code_generator or SkillCodeGenerator(llm_client=None)
        self._test_generator = test_generator
        self._config = config or SkillTDDConfig()
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[2]
        self._repo_root = Path(repo_root).resolve()

    def synthesize_and_test(
        self,
        spec: SkillSpec,
        *,
        examples: Sequence[SkillExample | Mapping[str, Any]] | None = None,
        references: Any | None = None,
        few_shot_examples: Any | None = None,
        extra_pythonpath: Sequence[str] | None = None,
    ) -> SkillTDDResult:
        start = time.time()
        normalized_examples = self._normalize_examples(examples)
        max_attempts = max(1, int(self._config.max_attempts))
        extra_pythonpath = list(extra_pythonpath or [])

        last_failure: Dict[str, Any] | None = None
        last_result: SkillTDDResult | None = None
        package_name = _slugify(spec.name)
        module_import = f"{package_name}.skill"

        tests_source = self._select_tests_source(
            spec,
            module_import=module_import,
            examples=normalized_examples,
            references=references,
            few_shot_examples=few_shot_examples,
        )
        base_references = self._attach_tests_reference(references, tests_source)

        for attempt in range(1, max_attempts + 1):
            attempt_references = self._augment_references(base_references, last_failure)
            handler_source = self._generate_handler_source(
                spec,
                module_import=module_import,
                examples=normalized_examples,
                references=attempt_references,
                few_shot_examples=few_shot_examples,
            )

            artifact_dir, pytest_result = self._run_pytest(
                package_name=package_name,
                handler_source=handler_source,
                tests_source=tests_source,
                module_import=module_import,
                extra_pythonpath=extra_pythonpath,
            )

            duration_s = time.time() - start
            last_result = SkillTDDResult(
                passed=pytest_result.passed,
                attempts=attempt,
                handler_source=handler_source,
                tests_source=tests_source,
                pytest=pytest_result,
                duration_s=duration_s,
                artifact_dir=str(artifact_dir) if artifact_dir else None,
            )

            if pytest_result.passed:
                return last_result

            last_failure = {
                "attempt": attempt,
                "pytest_stdout": pytest_result.stdout,
                "pytest_stderr": pytest_result.stderr,
                "tests_source": tests_source,
                "handler_source": handler_source,
            }
            logger.info("Skill TDD attempt %s/%s failed for %s", attempt, max_attempts, spec.name)

        assert last_result is not None  # for type checkers
        return last_result

    # ------------------------------------------------------------------
    def _normalize_examples(
        self, examples: Sequence[SkillExample | Mapping[str, Any]] | None
    ) -> Tuple[SkillExample, ...]:
        if not examples:
            return tuple()
        normalized: list[SkillExample] = []
        for item in examples:
            if isinstance(item, SkillExample):
                normalized.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            payload = item.get("payload")
            expected = item.get("expected")
            context = item.get("context")
            match = str(item.get("match") or "subset").strip().lower()
            if not isinstance(payload, dict):
                continue
            if expected is not None and not isinstance(expected, dict):
                continue
            if context is not None and not isinstance(context, dict):
                continue
            normalized.append(
                SkillExample(
                    payload=dict(payload),
                    expected=dict(expected) if isinstance(expected, dict) else None,
                    context=dict(context) if isinstance(context, dict) else None,
                    match=match if match in {"subset", "exact"} else "subset",
                )
            )
        return tuple(normalized)

    def _select_tests_source(
        self,
        spec: SkillSpec,
        *,
        module_import: str,
        examples: Tuple[SkillExample, ...],
        references: Any | None,
        few_shot_examples: Any | None,
    ) -> str:
        if examples:
            return self._render_example_tests(module_import=module_import, examples=examples)
        if self._test_generator is not None:
            try:
                generated = self._test_generator.generate(
                    spec,
                    module_import=module_import,
                    references=references,
                    few_shot_examples=few_shot_examples,
                )
                if generated.tests_source.strip():
                    return generated.tests_source
            except Exception:
                pass
        return self._render_smoke_tests(module_import=module_import)

    def _attach_tests_reference(self, references: Any | None, tests_source: str) -> Any | None:
        tests = str(tests_source or "").strip()
        if not tests:
            return references
        blob = {"tdd_target_tests": _truncate(tests, max_chars=6000)}
        if references is None:
            return [blob]
        if isinstance(references, list):
            return [*references, blob]
        if isinstance(references, tuple):
            return [*list(references), blob]
        if isinstance(references, dict):
            return [references, blob]
        return [references, blob]

    def _augment_references(self, references: Any | None, last_failure: Dict[str, Any] | None) -> Any | None:
        if last_failure is None:
            return references

        feedback = {
            "tdd_feedback": {
                "attempt": last_failure.get("attempt"),
                "pytest_stdout": _truncate(str(last_failure.get("pytest_stdout") or ""), max_chars=6000),
                "pytest_stderr": _truncate(str(last_failure.get("pytest_stderr") or ""), max_chars=6000),
            }
        }

        if references is None:
            return feedback
        if isinstance(references, list):
            return [*references, feedback]
        if isinstance(references, tuple):
            return [*list(references), feedback]
        if isinstance(references, dict):
            return [references, feedback]
        return [references, feedback]

    def _generate_handler_source(
        self,
        spec: SkillSpec,
        *,
        module_import: str,
        examples: Tuple[SkillExample, ...],
        references: Any | None,
        few_shot_examples: Any | None,
    ) -> str:
        generation_kwargs: Dict[str, Any] = {
            "module_import": module_import,
            "include_tests": False,
        }
        if references is not None:
            try:
                if "references" in inspect.signature(self._generator.generate).parameters:
                    generation_kwargs["references"] = references
            except Exception:
                pass

        combined_few_shot: Any | None = None
        if examples:
            try:
                example_payloads = [
                    {
                        "payload": dict(ex.payload),
                        "expected": dict(ex.expected) if isinstance(ex.expected, dict) else None,
                        "context": dict(ex.context) if isinstance(ex.context, dict) else None,
                        "match": ex.match,
                    }
                    for ex in examples
                ]
                if few_shot_examples is None:
                    combined_few_shot = {"examples": example_payloads}
                elif isinstance(few_shot_examples, Mapping):
                    merged = dict(few_shot_examples)
                    existing = merged.get("examples")
                    if isinstance(existing, list):
                        merged["examples"] = [*existing, *example_payloads]
                    else:
                        merged["examples"] = list(example_payloads)
                    combined_few_shot = merged
                else:
                    combined_few_shot = {"pseudocode": few_shot_examples, "examples": example_payloads}
            except Exception:
                pass
        else:
            combined_few_shot = few_shot_examples

        if combined_few_shot is not None:
            try:
                if "few_shot_examples" in inspect.signature(self._generator.generate).parameters:
                    generation_kwargs["few_shot_examples"] = combined_few_shot
            except Exception:
                pass

        generation = self._generator.generate(spec, **generation_kwargs)
        return str(generation.handler_source or "")

    def _run_pytest(
        self,
        *,
        package_name: str,
        handler_source: str,
        tests_source: str,
        module_import: str,
        extra_pythonpath: Sequence[str],
    ) -> Tuple[Optional[Path], PytestRunResult]:
        keep = bool(self._config.keep_artifacts)
        artifact_dir: Optional[Path] = None
        if keep:
            artifact_dir = Path(tempfile.mkdtemp(prefix=f"skill_tdd_{package_name}_"))
            root = artifact_dir
            cleanup = None
        else:
            tmp = tempfile.TemporaryDirectory(prefix=f"skill_tdd_{package_name}_")
            root = Path(tmp.name)
            cleanup = tmp.cleanup

        try:
            pkg_dir = root / package_name
            pkg_dir.mkdir(parents=True, exist_ok=True)
            (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
            (pkg_dir / "skill.py").write_text(handler_source.rstrip() + "\n", encoding="utf-8")

            tests_dir = root / "tests"
            tests_dir.mkdir(parents=True, exist_ok=True)
            test_file = tests_dir / f"test_{package_name}.py"
            test_file.write_text(tests_source.rstrip() + "\n", encoding="utf-8")

            cmd = (
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--maxfail=1",
                str(test_file),
            )
            env = dict(os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            if self._config.disable_pytest_plugin_autoload:
                env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

            pythonpath_parts = [str(root), str(self._repo_root)]
            pythonpath_parts.extend(str(p) for p in extra_pythonpath if p)
            if env.get("PYTHONPATH"):
                pythonpath_parts.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

            started = time.time()
            proc = subprocess.run(
                list(cmd),
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
                timeout=float(self._config.pytest_timeout_s),
            )
            duration_s = time.time() - started
            pytest_result = PytestRunResult(
                returncode=int(proc.returncode),
                stdout=str(proc.stdout or ""),
                stderr=str(proc.stderr or ""),
                duration_s=duration_s,
                cmd=cmd,
            )
            output = pytest_result.combined_output(max_chars=self._config.max_pytest_output_chars)
            logger.debug("Skill TDD pytest finished rc=%s output=%s", pytest_result.returncode, output)
            return artifact_dir, pytest_result
        finally:
            if cleanup is not None:
                cleanup()

    def _render_example_tests(self, *, module_import: str, examples: Tuple[SkillExample, ...]) -> str:
        encoded_examples = pprint.pformat(
            [
                {
                    "payload": ex.payload,
                    "expected": ex.expected,
                    "context": ex.context,
                    "match": ex.match,
                }
                for ex in examples
            ],
        )

        return (
            f'"""Example-driven tests for {module_import} (auto-generated)."""\n\n'
            "from __future__ import annotations\n\n"
            "import asyncio\n"
            "import inspect\n"
            "from typing import Any, Dict\n\n"
            "import pytest\n\n"
            f"from {module_import} import handle\n\n\n"
            "def _invoke(payload: Dict[str, Any], context: Dict[str, Any] | None = None) -> Any:\n"
            "    result = handle(payload, context=context)\n"
            "    if inspect.isawaitable(result):\n"
            "        return asyncio.run(result)\n"
            "    return result\n\n\n"
            "def _assert_subset(expected: Dict[str, Any], result: Any) -> None:\n"
            "    assert isinstance(result, dict)\n"
            "    for key, value in expected.items():\n"
            "        assert key in result\n"
            "        assert result[key] == value\n\n\n"
            f"EXAMPLES = {encoded_examples}\n\n\n"
            "def test_handle_examples():\n"
            "    assert callable(handle)\n"
            "    for idx, item in enumerate(EXAMPLES, start=1):\n"
            "        payload = dict(item.get('payload') or {})\n"
            "        expected = item.get('expected')\n"
            "        context = item.get('context')\n"
            "        match = str(item.get('match') or 'subset').lower()\n"
            "        result = _invoke(payload, context=context)\n"
            "        if expected is None:\n"
            "            assert isinstance(result, dict)\n"
            "            continue\n"
            "        assert isinstance(expected, dict)\n"
            "        if match == 'exact':\n"
            "            assert result == expected\n"
            "        else:\n"
            "            _assert_subset(expected, result)\n"
        )

    def _render_smoke_tests(self, *, module_import: str) -> str:
        return (
            f'"""Smoke tests for {module_import} (auto-generated)."""\n\n'
            "from __future__ import annotations\n\n"
            "import asyncio\n"
            "import inspect\n"
            "from typing import Any, Dict\n\n"
            f"from {module_import} import handle\n\n\n"
            "def _invoke(payload: Dict[str, Any], context: Dict[str, Any] | None = None) -> Any:\n"
            "    result = handle(payload, context=context)\n"
            "    if inspect.isawaitable(result):\n"
            "        return asyncio.run(result)\n"
            "    return result\n\n\n"
            "def test_handle_smoke():\n"
            "    assert callable(handle)\n"
            "    result = _invoke({'sample': 'value'})\n"
            "    assert isinstance(result, dict)\n"
        )
