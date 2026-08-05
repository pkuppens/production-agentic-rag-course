from typing import Optional

from src.config import Settings, get_settings

from .base import BaseEmbeddingsClient
from .jina_client import JinaEmbeddingsClient


def make_embeddings_client(settings: Optional[Settings] = None) -> BaseEmbeddingsClient:
    """Factory function to create an embeddings client for the configured provider.

    Creates a new client instance each time to avoid closed client issues.

    :param settings: Optional settings instance
    :returns: BaseEmbeddingsClient instance for the configured provider
    :raises ValueError: If an unsupported provider is configured
    """
    if settings is None:
        settings = get_settings()

    provider = settings.embeddings.provider

    if provider == "jina":
        return JinaEmbeddingsClient(api_key=settings.jina_api_key)

    raise ValueError(f"Unsupported embeddings provider: {provider!r}")


def make_embeddings_service(settings: Optional[Settings] = None) -> BaseEmbeddingsClient:
    """Factory function to create embeddings service (alias for make_embeddings_client).

    :param settings: Optional settings instance
    :returns: BaseEmbeddingsClient instance for the configured provider
    """
    return make_embeddings_client(settings)
