from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from src.config import Settings
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.services.ollama.embeddings_client import OllamaEmbeddingsClient


def make_settings() -> Settings:
    return Settings()


def mock_embed_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"embeddings": vectors}
    return response


@pytest.mark.parametrize("model", ["nomic-embed-text", "mxbai-embed-large", "bge-m3"])
@pytest.mark.asyncio
async def test_embed_query_for_each_model(model):
    """Each supported Ollama embedding model works via embed_query."""
    client = OllamaEmbeddingsClient(make_settings(), model=model)

    with patch("httpx.AsyncClient") as mock_client:
        response = mock_embed_response([[0.1, 0.2, 0.3]])
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=response)

        embedding = await client.embed_query("what is attention?")

        assert embedding == [0.1, 0.2, 0.3]
        call_args = mock_client.return_value.__aenter__.return_value.post.call_args
        assert call_args.kwargs["json"]["model"] == model
        assert call_args.kwargs["json"]["input"] == ["what is attention?"]


@pytest.mark.parametrize("model", ["nomic-embed-text", "mxbai-embed-large", "bge-m3"])
@pytest.mark.asyncio
async def test_embed_passages_for_each_model(model):
    """Each supported Ollama embedding model works via embed_passages."""
    client = OllamaEmbeddingsClient(make_settings(), model=model)

    with patch("httpx.AsyncClient") as mock_client:
        response = mock_embed_response([[0.1, 0.2], [0.3, 0.4]])
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=response)

        embeddings = await client.embed_passages(["passage one", "passage two"])

        assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_embed_passages_batches_requests():
    """embed_passages splits input into batches of the given size."""
    client = OllamaEmbeddingsClient(make_settings(), model="bge-m3")

    with patch("httpx.AsyncClient") as mock_client:
        post_mock = AsyncMock(
            side_effect=[
                mock_embed_response([[0.1], [0.2]]),
                mock_embed_response([[0.3]]),
            ]
        )
        mock_client.return_value.__aenter__.return_value.post = post_mock

        embeddings = await client.embed_passages(["a", "b", "c"], batch_size=2)

        assert embeddings == [[0.1], [0.2], [0.3]]
        assert post_mock.call_count == 2


@pytest.mark.asyncio
async def test_embed_query_connection_error():
    """A connection failure raises OllamaConnectionError."""
    client = OllamaEmbeddingsClient(make_settings(), model="bge-m3")

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        with pytest.raises(OllamaConnectionError):
            await client.embed_query("test")


@pytest.mark.asyncio
async def test_embed_query_timeout_error():
    """A timeout raises OllamaTimeoutError."""
    client = OllamaEmbeddingsClient(make_settings(), model="bge-m3")

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with pytest.raises(OllamaTimeoutError):
            await client.embed_query("test")


@pytest.mark.asyncio
async def test_embed_query_non_200_raises_ollama_exception():
    """A non-200 response raises OllamaException."""
    client = OllamaEmbeddingsClient(make_settings(), model="bge-m3")

    with patch("httpx.AsyncClient") as mock_client:
        response = MagicMock()
        response.status_code = 500
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=response)

        with pytest.raises(OllamaException):
            await client.embed_query("test")


def test_default_model_from_settings():
    """Without an explicit model override, the client uses settings.embeddings.ollama_embedding_model."""
    settings = make_settings()
    client = OllamaEmbeddingsClient(settings)

    assert client.model == settings.embeddings.ollama_embedding_model
