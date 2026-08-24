import logging
from typing import Optional

from src.config import get_settings
from src.services.slack.bot import SlackBot

logger = logging.getLogger(__name__)


def make_slack_service(
    opensearch_client,
    embeddings_client,
    ollama_client,
    cache_client=None,
    langfuse_tracer=None,
) -> Optional[SlackBot]:
    """
    Create Slack bot if enabled.

    Args:
        opensearch_client: OpenSearch client
        embeddings_client: Embeddings service client
        ollama_client: Ollama LLM client
        cache_client: Optional cache client
        langfuse_tracer: Optional Langfuse tracer (not used)

    Returns:
        SlackBot instance or None if disabled
    """
    settings = get_settings()

    if not settings.slack.enabled:
        logger.info("Slack bot is disabled")
        return None

    if not settings.slack.bot_token or not settings.slack.app_token:
        logger.warning("Slack bot/app token not configured")
        return None

    bot = SlackBot(
        bot_token=settings.slack.bot_token,
        app_token=settings.slack.app_token,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        ollama_client=ollama_client,
        cache_client=cache_client,
    )

    logger.info("Slack bot created successfully")
    return bot
