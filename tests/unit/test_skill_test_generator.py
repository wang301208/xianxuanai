from __future__ import annotations

from types import SimpleNamespace

from modules.skills.registry import SkillSpec
from modules.skills.test_generator import SkillTestGenerationConfig, SkillTestGenerator


class _StubLLMClient:
    def __init__(self, content: str):
        self._content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *_, **__):  # noqa: ANN001, ANN002 - test stub
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def test_skill_test_generator_falls_back_to_smoke_tests() -> None:
    spec = SkillSpec(name="X", description="Y")
    generator = SkillTestGenerator(llm_client=None, config=SkillTestGenerationConfig(model=None))

    result = generator.generate(spec, module_import="pkg.skill")

    assert result.used_llm is False
    assert "test_handle_smoke" in result.tests_source


def test_skill_test_generator_extracts_python_block_from_llm() -> None:
    spec = SkillSpec(name="Demo", description="Return ok.")
    llm = _StubLLMClient(
        """```python
from demo.skill import handle

def test_demo():
    assert callable(handle)
```"""
    )
    generator = SkillTestGenerator(llm_client=llm, config=SkillTestGenerationConfig(model="dummy"))

    result = generator.generate(spec, module_import="demo.skill")

    assert result.used_llm is True
    assert "from demo.skill import handle" in result.tests_source
    assert "def test_demo" in result.tests_source
