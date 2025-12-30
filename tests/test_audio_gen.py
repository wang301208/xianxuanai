from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from third_party.autogpt.autogpt.commands.audio_gen import speech_to_text, text_to_speech


class DummyWorkspace:
    def __init__(self, root: Path):
        self.root = root
    def get_path(self, rel: str) -> Path:
        return self.root / rel


def make_agent(tmp_path: Path):
    workspace = DummyWorkspace(tmp_path)
    openai_key = SimpleNamespace(get_secret_value=lambda: "test")
    legacy_config = SimpleNamespace(openai_credentials=SimpleNamespace(api_key=openai_key))
    return SimpleNamespace(workspace=workspace, legacy_config=legacy_config)


def test_text_to_speech(tmp_path):
    agent = make_agent(tmp_path)
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value.stream_to_file.side_effect = (
        lambda p: Path(p).write_bytes(b"audio")
    )
    with patch("autogpt.commands.audio_gen.OpenAI") as mock_openai:
        mock_openai.return_value.audio.speech.with_streaming_response.create.return_value = mock_ctx
        result = text_to_speech("hello", agent)
    path = Path(result.split(": ", 1)[1])
    assert path.exists()


def test_speech_to_text(tmp_path):
    agent = make_agent(tmp_path)
    audio_file = tmp_path / "input.mp3"
    audio_file.write_bytes(b"data")
    with patch("autogpt.commands.audio_gen.OpenAI") as mock_openai:
        mock_openai.return_value.audio.transcriptions.create.return_value.text = "hi"
        result = speech_to_text(str(audio_file), agent)
    assert result == "hi"
