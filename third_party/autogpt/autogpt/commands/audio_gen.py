"""Commands to convert between text and speech"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAI

from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent

COMMAND_CATEGORY = "audio"
COMMAND_CATEGORY_TITLE = "Audio"

logger = logging.getLogger(__name__)


@command(
    "text_to_speech",
    "Generate spoken audio from text",
    {
        "text": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The text to synthesize",
            required=True,
        ),
        "voice": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Voice selection for the provider",
            required=False,
        ),
    },
    lambda config: bool(config.openai_credentials),
    "Requires an OpenAI API key.",
)
def text_to_speech(text: str, agent: Agent, voice: str = "alloy") -> str:
    """Convert *text* to an audio file.

    Args:
        text: Text to synthesize.
        voice: Voice identifier understood by the provider.

    Returns:
        Path to the generated audio file inside the agent's workspace.
    """
    output_file = agent.workspace.root / f"{uuid.uuid4()}.mp3"

    client = OpenAI(
        api_key=agent.legacy_config.openai_credentials.api_key.get_secret_value()
    )
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
    ) as response:
        response.stream_to_file(output_file)

    logger.info("Audio generated for text: %s", text)
    return f"Saved to disk: {output_file}"


@command(
    "speech_to_text",
    "Transcribe audio to text",
    {
        "audio_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Path to the audio file relative to the workspace",
            required=True,
        ),
        "model": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Transcription model identifier",
            required=False,
        ),
    },
    lambda config: bool(config.openai_credentials),
    "Requires an OpenAI API key.",
)
def speech_to_text(audio_path: str, agent: Agent, model: str = "whisper-1") -> str:
    """Transcribe an audio file to text."""
    file_path = agent.workspace.get_path(audio_path)
    with file_path.open("rb") as audio_file:
        response = OpenAI(
            api_key=agent.legacy_config.openai_credentials.api_key.get_secret_value()
        ).audio.transcriptions.create(model=model, file=audio_file)
    return response.text
