import logging
from typing import Any, Dict, List, Optional

import httpx
from src.config import Settings
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.services.embeddings.base import BaseEmbeddingsClient

logger = logging.getLogger(__name__)


class OllamaEmbeddingsClient(BaseEmbeddingsClient):
    """Client for Ollama's local embeddings API.

    Runs fully on-prem against a local Ollama instance: no document or query
    text is sent to any external service. Supports any embedding model
    pulled into Ollama (e.g. nomic-embed-text, mxbai-embed-large, bge-m3).
    """

    def __init__(self, settings: Settings, model: Optional[str] = None):
        """Initialize Ollama embeddings client.

        :param settings: Application settings
        :param model: Optional model override (defaults to settings.embeddings.ollama_embedding_model)
        """
        self.base_url = settings.ollama_host
        self.timeout = httpx.Timeout(float(settings.ollama_timeout))
        self.model = model or settings.embeddings.ollama_embedding_model
        logger.info(f"Ollama embeddings client initialized (model={self.model})")

    async def _embed(self, inputs: List[str]) -> List[List[float]]:
        """Call Ollama's /api/embed endpoint for one or more inputs.

        :param inputs: List of texts to embed
        :returns: List of embedding vectors, one per input
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": inputs},
                )

                if response.status_code != 200:
                    raise OllamaException(f"Ollama embeddings request failed: {response.status_code}")

                result: Dict[str, Any] = response.json()
                return result["embeddings"]

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama embeddings request timed out: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error generating embeddings with Ollama: {e}")

    async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed text passages for indexing.

        :param texts: List of text passages to embed
        :param batch_size: Number of texts to process in each API call
        :returns: List of embedding vectors
        """
        embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings.extend(await self._embed(batch))

        logger.info(f"Successfully embedded {len(texts)} passages via Ollama ({self.model})")
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a search query.

        :param query: Query text to embed
        :returns: Embedding vector for the query
        """
        embeddings = await self._embed([query])
        logger.debug(f"Embedded query: '{query[:50]}...'")
        return embeddings[0]

    async def close(self) -> None:
        """No persistent connection is held; nothing to release."""
