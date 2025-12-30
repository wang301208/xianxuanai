import builtins
import importlib
import types
import sys

import pytest


def test_monitoring_import_without_fastapi(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fastapi":
            raise ModuleNotFoundError("No module named 'fastapi'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Stub optional monitoring dependencies so the package can import cleanly
    psutil_stub = types.ModuleType("psutil")

    def _stub_process(_pid: int):
        return types.SimpleNamespace(
            cpu_percent=lambda interval=None: 0.0, memory_percent=lambda: 0.0
        )

    psutil_stub.Process = _stub_process  # type: ignore[attr-defined]
    psutil_stub.NoSuchProcess = Exception  # type: ignore[attr-defined]
    psutil_stub.AccessDenied = Exception  # type: ignore[attr-defined]
    psutil_stub.cpu_percent = lambda interval=None: 0.0  # type: ignore[attr-defined]
    psutil_stub.virtual_memory = lambda: types.SimpleNamespace(percent=0.0)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)

    pyplot_stub = types.ModuleType("matplotlib.pyplot")

    def _noop(*_args, **_kwargs):  # pragma: no cover - trivial helper
        return None

    for name in [
        "figure",
        "plot",
        "xlabel",
        "ylabel",
        "title",
        "legend",
        "tight_layout",
        "savefig",
        "close",
    ]:
        setattr(pyplot_stub, name, _noop)

    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.pyplot = pyplot_stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_stub)

    # Ensure a clean import state for the monitoring package
    for module in ["backend.monitoring", "backend.monitoring.brain_state"]:
        sys.modules.pop(module, None)

    monitoring = importlib.import_module("backend.monitoring")

    # Accessing the package should work without FastAPI installed
    assert hasattr(monitoring, "record_memory_hit")

    with pytest.raises(RuntimeError, match="FastAPI is required"):
        monitoring.create_brain_app()
