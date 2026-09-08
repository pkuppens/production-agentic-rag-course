from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.config import SlackSettings
from src.services.slack.bot import SlackBot
from src.services.slack.factory import make_slack_service


class TestSlackBot:
    """Test Slack bot."""

    def test_bot_creation(self):
        """Test creating bot instance."""
        bot = SlackBot(
            bot_token="xoxb-test",
            app_token="xapp-test",
            opensearch_client=MagicMock(),
            embeddings_client=MagicMock(),
            ollama_client=MagicMock(),
        )

        assert bot.bot_token == "xoxb-test"
        assert bot.app_token == "xapp-test"
        assert bot.opensearch is not None
        assert bot.embeddings is not None
        assert bot.ollama is not None

    async def test_search_command_empty_query(self):
        """Test /search with no keywords prompts usage."""
        bot = SlackBot(
            bot_token="xoxb-test",
            app_token="xapp-test",
            opensearch_client=MagicMock(),
            embeddings_client=AsyncMock(),
            ollama_client=AsyncMock(),
        )

        ack = AsyncMock()
        respond = AsyncMock()
        await bot._search_command(ack=ack, respond=respond, command={"text": ""})

        ack.assert_called_once()
        respond.assert_called_once()
        assert "Usage" in respond.call_args.args[0]

    async def test_search_command_returns_results(self):
        """Test /search formats and returns matching papers."""
        opensearch = MagicMock()
        opensearch.search_unified.return_value = {"hits": [{"arxiv_id": "1706.03762", "title": "Attention Is All You Need"}]}
        embeddings = AsyncMock()
        embeddings.embed_query.return_value = [0.1, 0.2]

        bot = SlackBot(
            bot_token="xoxb-test",
            app_token="xapp-test",
            opensearch_client=opensearch,
            embeddings_client=embeddings,
            ollama_client=AsyncMock(),
        )

        ack = AsyncMock()
        respond = AsyncMock()
        await bot._search_command(ack=ack, respond=respond, command={"text": "attention"})

        respond.assert_called_once()
        message = respond.call_args.args[0]
        assert "Attention Is All You Need" in message
        assert "1706.03762" in message

    async def test_handle_question_uses_cache_hit(self):
        """Test a cache hit short-circuits retrieval/generation."""
        from src.schemas.api.ask import AskResponse

        cached = AskResponse(query="q", answer="cached answer", sources=[], chunks_used=0, search_mode="hybrid")
        cache = AsyncMock()
        cache.find_cached_response.return_value = cached

        opensearch = MagicMock()
        ollama = AsyncMock()

        bot = SlackBot(
            bot_token="xoxb-test",
            app_token="xapp-test",
            opensearch_client=opensearch,
            embeddings_client=AsyncMock(),
            ollama_client=ollama,
            cache_client=cache,
        )

        say = AsyncMock()
        await bot._handle_question({"text": "what is rag?"}, say)

        say.assert_called_once()
        assert "cached answer" in say.call_args.args[0]
        opensearch.search_unified.assert_not_called()
        ollama.generate.assert_not_called()

    async def test_handle_direct_message_ignores_bot_messages(self):
        """Test bot-authored/edited events in DMs don't trigger the RAG pipeline."""
        bot = SlackBot(
            bot_token="xoxb-test",
            app_token="xapp-test",
            opensearch_client=MagicMock(),
            embeddings_client=AsyncMock(),
            ollama_client=AsyncMock(),
        )

        say = AsyncMock()
        await bot._handle_direct_message({"text": "hi", "bot_id": "B123", "channel_type": "im"}, say)
        await bot._handle_direct_message({"text": "hi", "channel_type": "channel"}, say)

        say.assert_not_called()


class TestSlackSettings:
    """Test Slack settings."""

    def test_default_settings(self):
        """Test default settings."""
        settings = SlackSettings(bot_token="", app_token="", enabled=False)
        assert settings.enabled is False
        assert settings.bot_token == ""
        assert settings.app_token == ""

    def test_custom_settings(self):
        """Test custom settings."""
        settings = SlackSettings(bot_token="xoxb-x", app_token="xapp-x", enabled=True)
        assert settings.enabled is True
        assert settings.bot_token == "xoxb-x"
        assert settings.app_token == "xapp-x"


class TestSlackFactory:
    """Test factory."""

    @patch("src.services.slack.factory.get_settings")
    def test_factory_disabled(self, mock_settings):
        """Test factory returns None when disabled."""
        mock_settings.return_value.slack.enabled = False
        bot = make_slack_service(
            opensearch_client=MagicMock(),
            embeddings_client=MagicMock(),
            ollama_client=MagicMock(),
        )
        assert bot is None

    @patch("src.services.slack.factory.get_settings")
    def test_factory_missing_tokens(self, mock_settings):
        """Test factory returns None without both tokens."""
        mock_settings.return_value.slack.enabled = True
        mock_settings.return_value.slack.bot_token = "xoxb-test"
        mock_settings.return_value.slack.app_token = ""
        bot = make_slack_service(
            opensearch_client=MagicMock(),
            embeddings_client=MagicMock(),
            ollama_client=MagicMock(),
        )
        assert bot is None

    @patch("src.services.slack.factory.get_settings")
    def test_factory_success(self, mock_settings):
        """Test factory creates bot."""
        mock_settings.return_value.slack.enabled = True
        mock_settings.return_value.slack.bot_token = "xoxb-test"
        mock_settings.return_value.slack.app_token = "xapp-test"
        bot = make_slack_service(
            opensearch_client=MagicMock(),
            embeddings_client=MagicMock(),
            ollama_client=MagicMock(),
        )
        assert bot is not None
        assert bot.bot_token == "xoxb-test"
        assert bot.app_token == "xapp-test"
