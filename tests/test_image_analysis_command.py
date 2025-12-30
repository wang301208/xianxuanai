from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("PIL")
from PIL import Image

from third_party.autogpt.autogpt.commands import image_analysis
from third_party.autogpt.autogpt.commands.image_analysis import analyze_image


class DummyWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root

    def get_path(self, rel: str | Path) -> Path:
        return self.root / Path(rel)


def test_analyze_image_records_features_and_summary(tmp_path, monkeypatch) -> None:
    class DummyExtractor:
        def __init__(self) -> None:
            self.called_with: dict[str, Any] = {}

        @classmethod
        def from_pretrained(cls, name: str):  # type: ignore[override]
            instance = cls()
            instance.model_name = name
            return instance

        def __call__(self, images=None, text=None, return_tensors=None):  # type: ignore[override]
            if images is not None:
                return {"pixel_values": [[[1.0, 0.0], [0.5, 0.5]], [[0.25, 0.75], [0.1, 0.2]]]}
            if text is not None:
                return {"text_embeds": [[0.1, 0.2, 0.3, 0.4]]}
            return {}

    monkeypatch.setattr(image_analysis, "CLIPFeatureExtractor", DummyExtractor)

    workspace = DummyWorkspace(tmp_path)
    captured: dict[str, Any] = {}

    def record_visual_observation(*, image=None, features=None, text=None):
        captured["image"] = image
        captured["features"] = features
        captured["text"] = text

    agent = SimpleNamespace(workspace=workspace, record_visual_observation=record_visual_observation)

    image_file = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color=(120, 60, 30)).save(image_file)

    result = analyze_image("sample.png", agent, description_style="detailed")

    assert result["description_style"] == "detailed"
    assert result["world_model_updated"] is True
    assert captured["features"] is not None
    assert captured["text"]["description"] == result["description"]

    analysis_file = workspace.root / result["workspace_file"]
    assert analysis_file.exists()
    summary = json.loads(analysis_file.read_text(encoding="utf-8"))
    assert summary["description"] == result["description"]
    assert summary["image_features"] == result["image_features"]


def test_analyze_image_handles_missing_transformers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(image_analysis, "CLIPFeatureExtractor", None)

    workspace = DummyWorkspace(tmp_path)

    class DummyWorldModel:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def add_visual_observation(self, agent_id: str, **payload):
            self.calls.append((agent_id, payload))

    world_model = DummyWorldModel()
    agent = SimpleNamespace(
        workspace=workspace,
        world_model=world_model,
        settings=SimpleNamespace(agent_id="agent-x"),
    )

    image_file = tmp_path / "fallback.png"
    Image.new("RGB", (2, 3), color=(10, 20, 30)).save(image_file)

    result = analyze_image("fallback.png", agent)

    assert result["clip_model"] is None
    assert result["world_model_updated"] is True
    assert len(result["image_features"]) > 0
    assert world_model.calls
    agent_id, payload = world_model.calls[0]
    assert agent_id == "agent-x"
    assert payload["text"]["description"] == result["description"]

    analysis_file = workspace.root / result["workspace_file"]
    assert analysis_file.exists()
    summary = json.loads(analysis_file.read_text(encoding="utf-8"))
    assert summary["style"] == result["description_style"]
