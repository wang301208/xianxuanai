"""Stub for the OpenAI base client used by logging helpers."""


class _DummyLogger:
    def warning(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


log = _DummyLogger()

