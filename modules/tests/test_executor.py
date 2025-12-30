import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from pathlib import Path
import subprocess
import hashlib

import logging
import pytest

from capability.skill_library import SkillLibrary
from execution import Executor
from third_party.autogpt.autogpt.core.errors import SkillExecutionError


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)


def test_executor_flow(tmp_path: Path) -> None:
    repo = tmp_path
    init_repo(repo)
    lib = SkillLibrary(repo)

    hello_code = "def hello():\n    return 'hi'\n"
    goodbye_code = "def goodbye():\n    return 'bye'\n"
    lib.add_skill(
        "hello",
        hello_code,
        {"lang": "python", "signature": hashlib.sha256(hello_code.encode()).hexdigest()},
    )
    lib.add_skill(
        "goodbye",
        goodbye_code,
        {"lang": "python", "signature": hashlib.sha256(goodbye_code.encode()).hexdigest()},
    )

    executor = Executor(lib)
    results = executor.execute("hello then goodbye")

    assert list(results.keys()) == ["hello", "goodbye"]
    assert results["hello"] == "hi"
    assert results["goodbye"] == "bye"
    lib.close()


def test_call_skill_logs_exception(tmp_path: Path, caplog) -> None:
    repo = tmp_path
    init_repo(repo)
    lib = SkillLibrary(repo)
    fail_code = "def fail():\n    raise RuntimeError('boom')\n"
    lib.add_skill(
        "fail",
        fail_code,
        {"lang": "python", "signature": hashlib.sha256(fail_code.encode()).hexdigest()},
    )

    executor = Executor(lib)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SkillExecutionError) as exc_info:
            executor._call_skill("local", "fail")

    assert "fail" in str(exc_info.value)
    assert "boom" in str(exc_info.value)
    assert any(
        "fail" in record.message and "boom" in record.message
        for record in caplog.records
    )
    lib.close()
