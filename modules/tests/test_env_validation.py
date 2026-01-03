from pathlib import Path
import importlib.util

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
validation_path_candidates = (
    ROOT / "autogpts" / "autogpt" / "autogpt" / "config" / "validation.py",
    ROOT / "third_party" / "autogpt" / "autogpt" / "config" / "validation.py",
)
validation_path = next(
    (path for path in validation_path_candidates if path.is_file()), None
)
if validation_path is None:  # pragma: no cover - environment specific
    raise FileNotFoundError(
        "Unable to locate AutoGPT validation module; checked: "
        + ", ".join(str(p) for p in validation_path_candidates)
    )

spec = importlib.util.spec_from_file_location("validation", validation_path)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
validate_env = validation.validate_env


def test_validate_env_llm_success(monkeypatch):
    monkeypatch.setenv("BRAIN_BACKEND", "llm")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    validate_env()


def test_validate_env_llm_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BRAIN_BACKEND", "llm")
    with pytest.raises(ValidationError):
        validate_env()


def test_validate_env_default_allows_brain_simulation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BRAIN_BACKEND", raising=False)
    validate_env()


def test_validate_env_allows_whole_brain(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BRAIN_BACKEND", "whole_brain")
    validate_env()
