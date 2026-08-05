from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingsClient(ABC):
    """Interface for embeddings providers.

    Callers should depend on this interface, not a concrete provider, so the
    embeddings provider can be swapped via configuration.
    """

    @property
    @abstractmethod
    def model_label(self) -> str:
        """A stable "provider:model" label identifying what generated an embedding.

        Stored alongside every indexed chunk so a provider/model swap without
        reindexing can be detected instead of silently mixing incompatible
        vectors in the same index.
        """

    @abstractmethod
    async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed text passages for indexing.

        :param texts: List of text passages to embed
        :param batch_size: Number of texts to process in each API call
        :returns: List of embedding vectors
        """

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """Embed a search query.

        :param query: Query text to embed
        :returns: Embedding vector for the query
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the client (e.g. HTTP connections)."""
