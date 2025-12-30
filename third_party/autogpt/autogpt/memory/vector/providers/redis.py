from __future__ import annotations

import logging
from typing import Iterator

import orjson
import redis

from autogpt.config import Config

from ..memory_item import MemoryItem
from .base import VectorMemoryProvider

logger = logging.getLogger(__name__)


class RedisMemory(VectorMemoryProvider):
    """Memory backend that stores memories in a Redis list."""

    def __init__(self, config: Config) -> None:
        """Initialize a Redis memory provider.

        Args:
            config: application configuration instance
        """
        self.key = config.memory_index
        self.client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            password=config.redis_password or None,
            db=0,
        )
        if config.wipe_redis_on_start:
            self.clear()
        logger.debug(
            f"Initialized {__class__.__name__} with key '{self.key}' at {config.redis_host}:{config.redis_port}"
        )

    def __iter__(self) -> Iterator[MemoryItem]:
        items = self.client.lrange(self.key, 0, -1)
        for item in items:
            try:
                yield MemoryItem.parse_raw(item)
            except Exception as e:
                logger.warning(f"Could not parse MemoryItem from redis: {e}")

    def __contains__(self, x: MemoryItem) -> bool:
        serialized = orjson.dumps(x.dict())
        return self.client.lpos(self.key, serialized) is not None

    def __len__(self) -> int:
        return self.client.llen(self.key) or 0

    def add(self, item: MemoryItem):
        serialized = orjson.dumps(item.dict())
        self.client.rpush(self.key, serialized)
        logger.debug(f"Adding item to memory: {item.dump()}")
        return len(self)

    def discard(self, item: MemoryItem):
        serialized = orjson.dumps(item.dict())
        self.client.lrem(self.key, 0, serialized)

    def clear(self):
        self.client.delete(self.key)
