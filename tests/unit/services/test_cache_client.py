from unittest.mock import MagicMock

import pytest
import redis
from src.config import RedisSettings
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.cache import client as cache_client_module
from src.services.cache.client import CacheClient


@pytest.fixture
def redis_settings() -> RedisSettings:
    return RedisSettings(ttl_hours=1)


@pytest.fixture
def ask_request() -> AskRequest:
    return AskRequest(query="what is rag", model="llama3.2:3b")


async def test_find_cached_response_fails_open_on_redis_error(redis_settings, ask_request):
    redis_client = MagicMock()
    redis_client.get.side_effect = redis.ConnectionError("down")
    cache = CacheClient(redis_client, redis_settings)

    result = await cache.find_cached_response(ask_request)

    assert result is None


async def test_store_response_fails_open_on_redis_error(redis_settings, ask_request):
    redis_client = MagicMock()
    redis_client.set.side_effect = redis.ConnectionError("down")
    cache = CacheClient(redis_client, redis_settings)
    response = AskResponse(query="q", answer="a", sources=[], chunks_used=0, search_mode="hybrid")

    result = await cache.store_response(ask_request, response)

    assert result is False


async def test_circuit_opens_after_threshold_and_skips_redis(monkeypatch, redis_settings, ask_request):
    redis_client = MagicMock()
    redis_client.get.side_effect = redis.ConnectionError("down")
    cache = CacheClient(redis_client, redis_settings)

    fake_time = [0.0]
    monkeypatch.setattr(cache_client_module.time, "monotonic", lambda: fake_time[0])

    for _ in range(cache_client_module.CIRCUIT_FAILURE_THRESHOLD):
        await cache.find_cached_response(ask_request)

    assert redis_client.get.call_count == cache_client_module.CIRCUIT_FAILURE_THRESHOLD

    # Circuit is now open: further calls must not touch Redis at all.
    await cache.find_cached_response(ask_request)
    assert redis_client.get.call_count == cache_client_module.CIRCUIT_FAILURE_THRESHOLD

    # After the cooldown elapses, Redis is consulted again.
    fake_time[0] += cache_client_module.CIRCUIT_COOLDOWN_SECONDS
    redis_client.get.side_effect = None
    redis_client.get.return_value = None
    await cache.find_cached_response(ask_request)
    assert redis_client.get.call_count == cache_client_module.CIRCUIT_FAILURE_THRESHOLD + 1
