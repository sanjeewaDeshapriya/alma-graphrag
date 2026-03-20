from __future__ import annotations

import json
from typing import Any, Optional
import redis

from src.config import REDIS_HOST, REDIS_PORT


class Cache:
    def __init__(self) -> None:
        self._client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    def get(self, key: str) -> Optional[Any]:
        try:
            value = self._client.get(key)
        except redis.RedisError:
            return None
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 900) -> None:
        try:
            self._client.setex(key, ttl_seconds, json.dumps(value))
        except redis.RedisError:
            return
