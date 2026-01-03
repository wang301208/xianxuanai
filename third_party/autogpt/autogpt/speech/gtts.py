""" GTTS Voice. """
from __future__ import annotations

import os

try:  # optional dependency
    import gtts  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency absent
    gtts = None  # type: ignore[assignment]
from playsound import playsound

from autogpt.speech.base import VoiceBase


class GTTSVoice(VoiceBase):
    """GTTS Voice."""

    def _setup(self) -> None:
        pass

    def _speech(self, text: str, _: int = 0) -> bool:
        """Play the given text."""
        if gtts is None:
            raise RuntimeError("gtts is required for GTTSVoice but is not installed.")
        tts = gtts.gTTS(text)
        tts.save("speech.mp3")
        playsound("speech.mp3", True)
        os.remove("speech.mp3")
        return True
