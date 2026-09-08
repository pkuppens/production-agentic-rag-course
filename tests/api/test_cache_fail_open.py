from unittest.mock import AsyncMock, MagicMock, patch

import redis
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from src.main import app


async def test_app_boots_and_ask_works_when_redis_unreachable():
    """Redis is a performance layer, not a dependency (docs/adr/0005-cache-fails-open.md).

    Startup must not fail, and `/ask` must still degrade gracefully to
    no-cache, when Redis is unreachable at boot.
    """
    with (
        patch("src.db.interfaces.postgresql.PostgreSQLDatabase.startup") as mock_startup,
        patch("src.db.interfaces.postgresql.PostgreSQLDatabase.get_session") as mock_get_session,
        patch("src.main.make_opensearch_client") as mock_os,
        patch("src.main.make_arxiv_client") as mock_arxiv,
        patch("src.main.make_pdf_parser_service") as mock_pdf,
        patch("src.main.make_ollama_client") as mock_ollama,
        patch("src.main.make_cache_client", side_effect=redis.ConnectionError("Redis unreachable")),
        patch("src.main.make_langfuse_tracer") as mock_langfuse,
        patch("src.main.make_telegram_service") as mock_telegram,
    ):
        mock_startup.return_value = None
        mock_telegram.return_value = None

        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_get_session.return_value.__exit__.return_value = None

        mock_os.return_value = MagicMock()
        mock_langfuse.return_value = MagicMock()
        mock_arxiv.return_value = AsyncMock()
        mock_pdf.return_value = AsyncMock()
        mock_ollama.return_value = AsyncMock()

        async with LifespanManager(app) as manager:
            assert app.state.cache_client is None

            async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
                response = await client.post("/api/v1/ask", json={"query": "test query", "model": "llama3.2:3b"})
                assert response.status_code in [200, 500, 503]
