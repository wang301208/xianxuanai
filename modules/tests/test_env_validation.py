from pathlib import Path
import importlib.util

import pytest
from pydantic import ValidationError

spec = importlib.util.spec_from_file_location(
    "validation", Path("autogpts/autogpt/autogpt/config/validation.py")
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
validate_env = validation.validate_env


def test_validate_env_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    validate_env()


def test_validate_env_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BRAIN_BACKEND", raising=False)
    with pytest.raises(ValidationError):
        validate_env()


def test_validate_env_allows_whole_brain(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BRAIN_BACKEND", "whole_brain")
    validate_env()
