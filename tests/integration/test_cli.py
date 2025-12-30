import io
import os
import shutil
import sys
import types
import subprocess
from pathlib import Path

from click.testing import CliRunner

# Stub external modules required at import time
sys.modules.setdefault("github", types.ModuleType("github"))

# Stub AutoGPT package used by the CLI
sys.modules.setdefault("autogpts", types.ModuleType("autogpts"))
sys.modules.setdefault("third_party.autogpt", types.ModuleType("third_party.autogpt"))
sys.modules.setdefault(
    "third_party.autogpt.autogpt", types.ModuleType("third_party.autogpt.autogpt")
)
sys.modules.setdefault(
    "third_party.autogpt.autogpt.core",
    types.ModuleType("third_party.autogpt.autogpt.core"),
)
errors_mod = types.ModuleType("third_party.autogpt.autogpt.core.errors")
class AutoGPTError(Exception):
    pass
errors_mod.AutoGPTError = AutoGPTError
logging_mod = types.ModuleType("third_party.autogpt.autogpt.core.logging")
logging_mod.handle_exception = lambda e: None
logging_mod.setup_exception_hooks = lambda: None
sys.modules["third_party.autogpt.autogpt.core.errors"] = errors_mod
sys.modules["third_party.autogpt.autogpt.core.logging"] = logging_mod

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.cli import cli  # noqa: E402  (import after stubs)


def test_setup_command(monkeypatch):
    runner = CliRunner()

    def fake_exists(path):
        if path.endswith("setup.sh") or path.endswith("setup.ps1"):
            return True
        # Simulate missing GitHub token so instructions are printed
        if path == ".github_access_token":
            return False
        return False

    monkeypatch.setattr(os.path, "exists", fake_exists)

    def fake_which(name: str):
        # os.path.exists is patched above; stub which() to avoid PATH probing.
        if name in {"pwsh", "powershell", "bash"}:
            return name
        return None

    monkeypatch.setattr(shutil, "which", fake_which)

    # Avoid creating files
    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO())

    # Stub subprocess operations
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)

    def fake_check_output(cmd, **kwargs):
        if cmd[:3] == ["git", "config", "user.name"]:
            return b"Tester"
        if cmd[:3] == ["git", "config", "user.email"]:
            return b"tester@example.com"
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    result = runner.invoke(cli, ["setup"])
    assert result.exit_code == 0
    assert "Setup initiated" in result.output


def test_agent_start(monkeypatch):
    runner = CliRunner()

    def fake_exists(path):
        if "autogpts/test" in path:
            return True
        return False

    def fake_isfile(path):
        if path.endswith("run") or path.endswith("run_benchmark"):
            return True
        return False

    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os.path, "isfile", fake_isfile)
    monkeypatch.setattr(os, "chdir", lambda p: None)

    class DummyProc:
        def wait(self):
            pass

    popen_calls = []

    def fake_popen(cmd, cwd=None):
        popen_calls.append((tuple(cmd), cwd))
        return DummyProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    from scripts import cli as cli_module
    monkeypatch.setattr(cli_module, "wait_until_conn_ready", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "sys", sys, raising=False)

    result = runner.invoke(cli, ["agent", "start", "test", "--no-setup"])
    assert result.exit_code == 0
    assert any("./run" in call[0] for call in popen_calls)


def test_agent_stop_no_process(monkeypatch):
    runner = CliRunner()

    def fake_check_output(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    result = runner.invoke(cli, ["agent", "stop"])
    assert result.exit_code == 0
    assert "No process is running on port 8000" in result.output
    assert "No process is running on port 8080" in result.output
