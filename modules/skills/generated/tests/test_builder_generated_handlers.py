import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


# Provide lightweight stubs for optional backend dependencies used by the skills registry.
GraphStore = type("GraphStore", (object,), {})

execution_mod = _stub_module("modules.execution")
task_adapter_mod = _stub_module("modules.execution.task_adapter")


class DummyFuture:
    def __init__(self, result=None):
        self._result = result

    def add_done_callback(self, *_):
        return None

    def result(self, timeout=None):  # noqa: ARG002 - match Future API
        return self._result

    def cancel(self):
        return None


class DummyAdapter:
    def submit(self, fn, *args, **kwargs):  # noqa: ANN001 - test stub
        return DummyFuture(fn(*args))

    def shutdown(self):
        return None


def create_task_adapter(*_, **__):  # noqa: ANN001, ANN002 - test stub
    return DummyAdapter()


task_adapter_mod.TaskAdapter = DummyAdapter
task_adapter_mod.create_task_adapter = create_task_adapter
execution_mod.TaskAdapter = DummyAdapter
execution_mod.create_task_adapter = create_task_adapter

backend_mod = _stub_module("backend")
autogpt_mod = _stub_module("backend.autogpt")
backend_mod.autogpt = autogpt_mod
autogpt_inner = _stub_module("backend.autogpt.autogpt")
autogpt_mod.autogpt = autogpt_inner
core_mod = _stub_module("backend.autogpt.autogpt.core")
autogpt_inner.core = core_mod
kg_mod = _stub_module("backend.autogpt.autogpt.core.knowledge_graph")
core_mod.knowledge_graph = kg_mod
graph_store_mod = _stub_module("backend.autogpt.autogpt.core.knowledge_graph.graph_store")
graph_store_mod.GraphStore = GraphStore
kg_mod.graph_store = graph_store_mod
ontology_mod = _stub_module("backend.autogpt.autogpt.core.knowledge_graph.ontology")
ontology_mod.EntityType = type("EntityType", (object,), {})
ontology_mod.RelationType = type("RelationType", (object,), {})
kg_mod.ontology = ontology_mod

knowledge_mod = _stub_module("backend.knowledge")
backend_mod.knowledge = knowledge_mod
registry_mod = _stub_module("backend.knowledge.registry")
knowledge_mod.registry = registry_mod


def get_graph_store_instance():
    return GraphStore()


registry_mod.get_graph_store_instance = get_graph_store_instance

from modules.skills.builder import DEFAULT_HANDLER, SkillAutoBuilder, _slugify
from modules.skills.generator import SkillGenerationResult
from modules.skills.registry import SkillSpec


class StubGenerator:
    def __init__(self, handler_source: str, tests_source: str | None = None):
        self.handler_source = handler_source
        self.tests_source = tests_source
        self.calls: list[tuple[SkillSpec, str | None, bool]] = []

    def generate(self, spec: SkillSpec, *, module_import: str | None = None, include_tests: bool = True):
        self.calls.append((spec, module_import, include_tests))
        return SkillGenerationResult(
            handler_source=self.handler_source,
            tests_source=self.tests_source if include_tests else None,
            used_llm=False,
        )


class FailingGenerator:
    def generate(self, spec: SkillSpec, *, module_import: str | None = None, include_tests: bool = True):
        raise RuntimeError("unable to generate")


class RecordingSandbox:
    def __init__(self, *, should_fail: bool = False):
        self.should_fail = should_fail
        self.calls: list[dict] = []

    def run(self, handler_fn, payload, *, context=None, metadata=None):
        self.calls.append({"payload": payload, "context": context, "metadata": metadata})
        if self.should_fail:
            raise RuntimeError("sandbox failure")
        return handler_fn(payload, context=context)


@pytest.fixture()
def sample_spec():
    return SkillSpec(
        name="Sample Skill",
        description="Echo inputs",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )


def _load_generated_handler(builder: SkillAutoBuilder, root: Path, spec: SkillSpec):
    package = _slugify(spec.name)
    module_name = ".".join((package, "skill"))
    handler_path = root / package / "skill.py"
    return builder._load_handler(module_name, handler_path)


def test_generated_handler_runs_with_context(tmp_path: Path, sample_spec: SkillSpec):
    handler_source = """
from typing import Any, Dict

def handle(payload: Dict[str, Any], *, context: Dict[str, Any] | None = None):
    ctx = dict(context or {})
    return {"status": "ok", "echo": payload, "context": ctx}
    """
    generator = StubGenerator(handler_source=handler_source)
    builder = SkillAutoBuilder(plugin_root=tmp_path, registry=None, code_generator=generator)

    builder.create_skill(sample_spec, auto_register=False)

    handler = _load_generated_handler(builder, tmp_path, sample_spec)
    result = handler({"foo": "bar"}, context={"user": "tester"})

    assert result == {
        "status": "ok",
        "echo": {"foo": "bar"},
        "context": {"user": "tester"},
    }
    assert generator.calls, "Generator should have been invoked"


def test_fallback_template_used_on_generation_failure(tmp_path: Path, sample_spec: SkillSpec):
    builder = SkillAutoBuilder(
        plugin_root=tmp_path,
        registry=None,
        code_generator=FailingGenerator(),
    )

    builder.create_skill(sample_spec, auto_register=False)

    handler = _load_generated_handler(builder, tmp_path, sample_spec)
    result = handler({"foo": 1}, context={"bar": 2})

    assert result["status"] == "ok"
    assert result["received"] == {"foo": 1}
    assert result["context"] == {"bar": 2}
    assert result["summary"]["received_keys"] == ["foo"]
    assert result["summary"]["context_keys"] == ["bar"]


def test_sandbox_failure_marks_review_failed(tmp_path: Path, sample_spec: SkillSpec):
    review_root = tmp_path / "reviews"
    sandbox = RecordingSandbox(should_fail=True)
    builder = SkillAutoBuilder(
        plugin_root=tmp_path,
        review_root=review_root,
        registry=None,
        code_generator=StubGenerator(handler_source=DEFAULT_HANDLER),
        sandbox=sandbox,
    )

    builder.create_skill(sample_spec, auto_register=False)

    manifest_path = tmp_path / _slugify(sample_spec.name) / f"{_slugify(sample_spec.name)}.skill.json"
    payload = json.loads(manifest_path.read_text())

    assert payload["metadata"]["review"]["status"] == "failed"
    assert payload["metadata"]["sandbox"]["status"] == "failed"
    assert sandbox.calls, "Sandbox should have been invoked"


def test_rpc_skills_skip_sandbox_by_default_and_infer_rpc_config(tmp_path: Path) -> None:
    spec = SkillSpec(
        name="Remote Echo",
        description="Delegate to an external service",
        execution_mode="rpc",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    sandbox = RecordingSandbox(should_fail=True)
    builder = SkillAutoBuilder(
        plugin_root=tmp_path,
        registry=None,
        code_generator=StubGenerator(handler_source=DEFAULT_HANDLER),
        sandbox=sandbox,
    )

    builder.create_skill(
        spec,
        auto_register=False,
        metadata={"retrieval_context": "POST http://localhost:8300/invoke"},
    )

    manifest_path = tmp_path / _slugify(spec.name) / f"{_slugify(spec.name)}.skill.json"
    payload = json.loads(manifest_path.read_text())

    assert payload["execution_mode"] == "rpc"
    assert payload["rpc_config"]["endpoint"] == "http://localhost:8300"
    assert payload["rpc_config"]["path"] == "/invoke"
    assert payload["metadata"]["sandbox"]["status"] == "skipped"
    assert sandbox.calls == [], "RPC skills should not execute sandbox payloads by default"
