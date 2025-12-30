import os
import sys
import types
import asyncio
import logging
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "autogpts", "autogpt")))

# Stub external dependency used by the ability base module
inflection_stub = types.ModuleType("inflection")
inflection_stub.underscore = lambda x: x
sys.modules.setdefault("inflection", inflection_stub)
sys.modules.setdefault("sentry_sdk", types.ModuleType("sentry_sdk"))
sys.modules.setdefault("demjson3", types.ModuleType("demjson3"))
jsonschema_stub = types.ModuleType("jsonschema")

class _ValidationError(Exception):
    pass


class _Draft7Validator:
    def __init__(self, *_, **__):
        pass

    def iter_errors(self, *_, **__):
        return []


jsonschema_stub.ValidationError = _ValidationError
jsonschema_stub.Draft7Validator = _Draft7Validator
sys.modules.setdefault("jsonschema", jsonschema_stub)

from third_party.autogpt.autogpt.core.workspace.simple import (  # noqa: E402
    SimpleWorkspace,
    WorkspaceConfiguration,
    WorkspaceSettings,
)
from third_party.autogpt.autogpt.core.ability.builtins.evaluate_metrics import (  # noqa: E402
    EvaluateMetrics,
    _MetricsCollector,
)


@pytest.fixture
def workspace(tmp_path: Path):
    settings = WorkspaceSettings(
        name="workspace",
        description="test",
        configuration=WorkspaceConfiguration(
            root=str(tmp_path), parent=str(tmp_path), restrict_to_workspace=True
        ),
    )
    return SimpleWorkspace(settings, logging.getLogger(__name__))


@pytest.mark.asyncio
async def test_evaluate_complexity(monkeypatch, workspace):
    # Stub radon.cc_visit
    complexity_mod = types.ModuleType('radon.complexity')

    class Block:
        def __init__(self, complexity):
            self.complexity = complexity

    def cc_visit(_):
        return [Block(1)]

    complexity_mod.cc_visit = cc_visit
    radon_mod = types.ModuleType('radon')
    radon_mod.complexity = complexity_mod
    monkeypatch.setitem(sys.modules, 'radon', radon_mod)
    monkeypatch.setitem(sys.modules, 'radon.complexity', complexity_mod)

    ability = EvaluateMetrics(logging.getLogger(__name__), workspace)
    collector = _MetricsCollector()
    await ability._evaluate_complexity(collector, "def foo():\n    return 1\n")
    assert "complexity=" in collector.message()
    assert collector.success


@pytest.mark.asyncio
async def test_measure_runtime(workspace, tmp_path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")
    ability = EvaluateMetrics(logging.getLogger(__name__), workspace)
    collector = _MetricsCollector()
    await ability._measure_runtime(collector, script)
    assert "runtime=" in collector.message()
    assert collector.success


@pytest.mark.asyncio
async def test_collect_coverage(monkeypatch, workspace):
    class DummyProc:
        def __init__(self):
            self.returncode = 0

        async def communicate(self):
            return (b"TOTAL 1 0 100%", b"")

    async def dummy_exec(*args, **kwargs):
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", dummy_exec)

    ability = EvaluateMetrics(logging.getLogger(__name__), workspace)
    collector = _MetricsCollector()
    await ability._collect_coverage(collector)
    assert "coverage=100%" in collector.message()
    assert collector.success


@pytest.mark.asyncio
async def test_run_style_check(monkeypatch, workspace, tmp_path):
    class DummyProc:
        def __init__(self):
            self.returncode = 0

        async def communicate(self):
            return (b"", b"")

    async def dummy_exec(*args, **kwargs):
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", dummy_exec)

    ability = EvaluateMetrics(logging.getLogger(__name__), workspace)
    collector = _MetricsCollector()
    await ability._run_style_check(collector, tmp_path / "dummy.py")
    assert "style_errors=0" in collector.message()
    assert collector.success

