"""Exceptions raised by the OpenAI stub."""


class OpenAIError(Exception):
    pass


class RateLimitError(OpenAIError):
    pass


class APIStatusError(OpenAIError):
    pass

