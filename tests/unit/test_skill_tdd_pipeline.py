from __future__ import annotations

from modules.skills.generator import SkillGenerationResult
from modules.skills.registry import SkillSpec
from modules.skills.tdd_pipeline import SkillTDDConfig, SkillTDDPipeline
from modules.skills.test_generator import SkillTestGenerationResult


class StubGenerator:
    def __init__(self, handler_source: str):
        self.handler_source = handler_source
        self.calls: list[dict] = []

    def generate(self, spec: SkillSpec, *, module_import: str | None = None, include_tests: bool = True, references=None):
        self.calls.append(
            {
                "name": spec.name,
                "module_import": module_import,
                "include_tests": include_tests,
                "references_present": references is not None,
            }
        )
        return SkillGenerationResult(
            handler_source=self.handler_source,
            tests_source=None,
            used_llm=False,
        )


class StubTestGenerator:
    def __init__(self, tests_source: str):
        self.tests_source = tests_source
        self.calls: list[dict] = []

    def generate(self, spec: SkillSpec, *, module_import: str, references=None, few_shot_examples=None):  # noqa: ANN001
        self.calls.append(
            {
                "name": spec.name,
                "module_import": module_import,
                "references_present": references is not None,
                "few_shot_present": few_shot_examples is not None,
            }
        )
        return SkillTestGenerationResult(tests_source=self.tests_source.format(module_import=module_import), used_llm=False)


def test_skill_tdd_pipeline_passes_example_driven_tests() -> None:
    spec = SkillSpec(
        name="AdderSkill",
        description="Add two numbers.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    handler_source = """
from __future__ import annotations

from typing import Any, Dict


def handle(payload: Dict[str, Any], *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    a = int((payload or {}).get("a", 0))
    b = int((payload or {}).get("b", 0))
    return {"status": "ok", "result": a + b}
"""
    generator = StubGenerator(handler_source=handler_source)
    pipeline = SkillTDDPipeline(
        code_generator=generator,
        config=SkillTDDConfig(max_attempts=1, pytest_timeout_s=20.0),
    )

    result = pipeline.synthesize_and_test(
        spec,
        examples=[
            {"payload": {"a": 2, "b": 3}, "expected": {"result": 5}, "match": "subset"},
        ],
        references={"note": "dummy"},
    )

    assert result.passed is True
    assert result.attempts == 1
    assert result.pytest.returncode == 0
    assert generator.calls


def test_skill_tdd_pipeline_reports_failure() -> None:
    spec = SkillSpec(
        name="BrokenAdderSkill",
        description="Add two numbers (buggy).",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    handler_source = """
from __future__ import annotations

from typing import Any, Dict


def handle(payload: Dict[str, Any], *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    a = int((payload or {}).get("a", 0))
    b = int((payload or {}).get("b", 0))
    return {"status": "ok", "result": a - b}
"""
    pipeline = SkillTDDPipeline(
        code_generator=StubGenerator(handler_source=handler_source),
        config=SkillTDDConfig(max_attempts=1, pytest_timeout_s=20.0),
    )

    result = pipeline.synthesize_and_test(
        spec,
        examples=[
            {"payload": {"a": 2, "b": 3}, "expected": {"result": 5}, "match": "subset"},
        ],
    )

    assert result.passed is False
    assert result.attempts == 1
    assert result.pytest.returncode != 0
    assert "FAILED" in result.pytest.combined_output(max_chars=50_000)


def test_skill_tdd_pipeline_can_use_generated_tests_without_examples() -> None:
    spec = SkillSpec(
        name="AdderSkillFromTests",
        description="Add two numbers.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    handler_source = """
from __future__ import annotations

from typing import Any, Dict


def handle(payload: Dict[str, Any], *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    a = int((payload or {}).get("a", 0))
    b = int((payload or {}).get("b", 0))
    return {"status": "ok", "result": a + b}
"""
    tests_source = """
from __future__ import annotations

from {module_import} import handle


def test_addition():
    assert handle({{"a": 2, "b": 3}})["result"] == 5
"""

    pipeline = SkillTDDPipeline(
        code_generator=StubGenerator(handler_source=handler_source),
        test_generator=StubTestGenerator(tests_source),
        config=SkillTDDConfig(max_attempts=1, pytest_timeout_s=20.0),
    )

    result = pipeline.synthesize_and_test(spec, references={"doc": "pseudo"})

    assert result.passed is True
    assert result.pytest.returncode == 0


def test_skill_tdd_pipeline_retries_until_tests_pass() -> None:
    spec = SkillSpec(
        name="FlakyAdderSkill",
        description="Add two numbers (first attempt buggy).",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    wrong = """
from __future__ import annotations

from typing import Any, Dict


def handle(payload: Dict[str, Any], *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    a = int((payload or {}).get("a", 0))
    b = int((payload or {}).get("b", 0))
    return {"status": "ok", "result": a - b}
"""
    right = wrong.replace("a - b", "a + b")

    class FlakyGenerator(StubGenerator):
        def __init__(self):
            super().__init__(handler_source=wrong)
            self._count = 0

        def generate(self, spec: SkillSpec, *, module_import: str | None = None, include_tests: bool = True, references=None):
            self._count += 1
            self.handler_source = wrong if self._count == 1 else right
            return super().generate(spec, module_import=module_import, include_tests=include_tests, references=references)

    tests_source = """
from __future__ import annotations

from {module_import} import handle


def test_addition():
    assert handle({{"a": 2, "b": 3}})["result"] == 5
"""

    pipeline = SkillTDDPipeline(
        code_generator=FlakyGenerator(),
        test_generator=StubTestGenerator(tests_source),
        config=SkillTDDConfig(max_attempts=2, pytest_timeout_s=20.0),
    )

    result = pipeline.synthesize_and_test(spec)

    assert result.passed is True
    assert result.attempts == 2
