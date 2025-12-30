import logging

import pytest

from autogpt.core.ability.builtins.lint_code import LintCode
from autogpt.core.workspace.simple import SimpleWorkspace


@pytest.mark.asyncio
async def test_lint_code_pass(tmp_path):
    logger = logging.getLogger("test")
    workspace_settings = SimpleWorkspace.default_settings.copy(deep=True)
    workspace_settings.configuration.root = str(tmp_path)
    workspace = SimpleWorkspace(workspace_settings, logger)

    file_path = tmp_path / "good.py"
    file_path.write_text("x = 1\n")

    lint = LintCode(logger, LintCode.default_configuration, workspace)
    result = await lint(file_path=str(file_path.relative_to(tmp_path)))
    assert result.success


@pytest.mark.asyncio
async def test_lint_code_fail(tmp_path):
    logger = logging.getLogger("test")
    workspace_settings = SimpleWorkspace.default_settings.copy(deep=True)
    workspace_settings.configuration.root = str(tmp_path)
    workspace = SimpleWorkspace(workspace_settings, logger)

    file_path = tmp_path / "bad.py"
    file_path.write_text("import os\n")

    lint = LintCode(logger, LintCode.default_configuration, workspace)
    result = await lint(file_path=str(file_path.relative_to(tmp_path)))
    assert not result.success
