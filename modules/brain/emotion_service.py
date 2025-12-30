"""Client for invoking an external emotion/sentiment classifier service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class EmotionServiceConfig:
    """Configuration for the remote emotion model endpoint."""

    endpoint: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: float = 8.0
    verify_ssl: bool = True
    enabled: bool = True

    def headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class EmotionServiceClient:
    """Lightweight HTTP client for a pretrained emotion classifier."""

    def __init__(self, config: EmotionServiceConfig) -> None:
        self.config = config

    def analyze(self, stimulus: str, context: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if not self.config.enabled:
            raise RuntimeError("Emotion service is disabled in the configuration")
        payload: Dict[str, Any] = {"text": stimulus}
        if context:
            payload["context"] = context
        if self.config.model:
            payload["model"] = self.config.model
        response = requests.post(
            self.config.endpoint,
            json=payload,
            headers=self.config.headers(),
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            logger.debug("Unexpected response type from emotion service: %s", type(result))
            raise ValueError("Emotion service response must be a JSON object")
        return result


__all__ = ["EmotionServiceClient", "EmotionServiceConfig"]
