import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.getcwd()))
from modules.diagnostics import record_error


def test_record_error_logs_stack_and_context(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="diagnostics")

    try:
        raise ValueError("boom")
    except ValueError as err:
        record_error(err, {"user": "alice"})

    log_text = caplog.text
    assert "boom" in log_text
    assert "user" in log_text
    assert "ValueError" in log_text
