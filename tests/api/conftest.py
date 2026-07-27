from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from src.main import app


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Async backend for testing."""
    return "asyncio"


@pytest.fixture
async def client():
    """HTTP client for API testing with mocked services.

    Patches target where each factory is *imported* (``src.main`` /
    ``src.routers.ping``), not where it's defined. ``main.py`` does
    ``from src.services.x.factory import make_x_client``, which binds its
    own name into ``src.main``'s namespace — patching the origin module
    leaves that reference untouched and the real, network-calling factory
    still runs during the app's lifespan.
    """
    with (
        patch("src.db.interfaces.postgresql.PostgreSQLDatabase.startup") as mock_startup,
        patch("src.db.interfaces.postgresql.PostgreSQLDatabase.get_session") as mock_get_session,
        patch("src.main.make_opensearch_client") as mock_os,
        patch("src.main.make_arxiv_client") as mock_arxiv,
        patch("src.main.make_pdf_parser_service") as mock_pdf,
        patch("src.main.make_ollama_client") as mock_ollama,
        patch("src.main.make_cache_client") as mock_cache,
        patch("src.main.make_langfuse_tracer") as mock_langfuse,
        patch("src.routers.ping.OllamaClient") as mock_ping_ollama,
        patch("src.repositories.paper.PaperRepository.get_by_arxiv_id") as mock_get_by_id,
    ):
        # Mock startup to do nothing
        mock_startup.return_value = None

        # Mock get_session to return a mock session
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_get_session.return_value.__exit__.return_value = None

        # Mock repository methods to return None (not found) by default
        mock_get_by_id.return_value = None

        # OpenSearch and Langfuse clients are fully synchronous
        mock_os.return_value = MagicMock()
        mock_langfuse.return_value = MagicMock()

        # These clients expose async methods
        mock_arxiv.return_value = AsyncMock()
        mock_pdf.return_value = AsyncMock()
        mock_ollama.return_value = AsyncMock()

        # Default to a cache miss so tests exercise the real ask/search logic
        # unless they explicitly configure a cache hit themselves.
        mock_cache.return_value = AsyncMock()
        mock_cache.return_value.find_cached_response = AsyncMock(return_value=None)
        mock_cache.return_value.store_response = AsyncMock(return_value=True)

        # ping.py constructs its own OllamaClient directly for the health check
        mock_ping_ollama.return_value.health_check = AsyncMock(return_value={"status": "healthy", "message": "mocked"})

        async with LifespanManager(app) as manager:
            async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
                yield client
