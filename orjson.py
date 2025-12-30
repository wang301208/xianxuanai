"""Minimal stub for :mod:`orjson` providing ``dumps`` and ``loads``."""

import json

OPT_INDENT_2 = 1 << 0
OPT_SORT_KEYS = 1 << 1
OPT_SERIALIZE_NUMPY = 1 << 2
OPT_SERIALIZE_DATACLASS = 1 << 3


def dumps(data, option=None):  # pragma: no cover - deterministic stub
    if option is None:
        option = 0
    indent = 2 if option & OPT_INDENT_2 else None
    sort_keys = bool(option & OPT_SORT_KEYS)
    return json.dumps(data, indent=indent, sort_keys=sort_keys, default=_default).encode("utf-8")


def loads(data):  # pragma: no cover - deterministic stub
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)


def _default(obj):  # pragma: no cover - deterministic stub
    try:
        return obj.to_dict()
    except AttributeError:
        try:
            return obj.__dict__
        except AttributeError:
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
