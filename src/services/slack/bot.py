import logging
from typing import Optional

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)


class SlackBot:
    """Simple Slack bot for Q&A, mirroring TelegramBot's shape (src/services/telegram/bot.py)."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        opensearch_client,
        embeddings_client,
        ollama_client,
        cache_client=None,
    ):
        """Initialize bot with required services."""
        self.bot_token = bot_token
        self.app_token = app_token
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.ollama = ollama_client
        self.cache = cache_client
        self.app: Optional[AsyncApp] = None
        self.handler: Optional[AsyncSocketModeHandler] = None

    async def start(self) -> None:
        """Start the bot in Socket Mode (no public HTTPS endpoint required)."""
        logger.info("Starting Slack bot...")
        self.app = AsyncApp(token=self.bot_token)

        self.app.event("app_mention")(self._handle_mention)
        self.app.event("message")(self._handle_direct_message)
        self.app.command("/search")(self._search_command)

        self.handler = AsyncSocketModeHandler(self.app, self.app_token)
        await self.handler.start_async()
        logger.info("Slack bot started successfully")

    async def stop(self) -> None:
        """Stop the bot."""
        if self.handler:
            await self.handler.close_async()
            logger.info("Slack bot stopped")

    async def _search_command(self, ack, respond, command) -> None:
        """Handle /search slash command."""
        await ack()

        query = (command.get("text") or "").strip()
        if not query:
            await respond("Usage: /search <keywords>\nExample: /search neural networks")
            return

        try:
            query_embedding = await self.embeddings.embed_query(query)

            results = self.opensearch.search_unified(
                query=query,
                query_embedding=query_embedding,
                size=10,
                use_hybrid=True,
            )

            hits = results.get("hits", [])
            if not hits:
                await respond("No papers found. Try different keywords.")
                return

            # Deduplicate by arxiv_id (since chunks may have same paper)
            seen_ids = set()
            unique_papers = []
            for hit in hits:
                arxiv_id = hit.get("arxiv_id", "")
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    unique_papers.append(hit)
                if len(unique_papers) >= 5:
                    break

            message = f"Found {len(unique_papers)} papers:\n\n"
            for idx, hit in enumerate(unique_papers, 1):
                title = hit.get("title", "Untitled")
                arxiv_id = hit.get("arxiv_id", "")
                url = f"https://arxiv.org/abs/{arxiv_id}"
                message += f"{idx}. {title}\n{url}\n\n"

            await respond(message)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            await respond(f"Search failed: {str(e)}")

    async def _handle_mention(self, event, say) -> None:
        """Handle @bot mentions in channels."""
        await self._handle_question(event, say)

    async def _handle_direct_message(self, event, say) -> None:
        """Handle direct messages to the bot (ignore bot's own messages/edits/etc.)."""
        if event.get("bot_id") or event.get("subtype"):
            return
        if event.get("channel_type") != "im":
            return
        await self._handle_question(event, say)

    async def _handle_question(self, event, say) -> None:
        """Handle a user question from either a mention or a DM."""
        query = event.get("text", "")

        try:
            ask_request = AskRequest(query=query, top_k=3, use_hybrid=True)

            if self.cache:
                try:
                    cached_response = await self.cache.find_cached_response(ask_request)
                    if cached_response:
                        await self._send_answer(say, cached_response)
                        return
                except Exception as e:
                    logger.warning(f"Cache lookup failed: {e}")

            from src.services.ollama.prompts import RAGPromptBuilder

            query_embedding = None
            if ask_request.use_hybrid:
                try:
                    query_embedding = await self.embeddings.embed_query(query)
                except Exception as e:
                    logger.warning(f"Failed to generate embeddings: {e}")

            search_results = self.opensearch.search_unified(
                query=query,
                query_embedding=query_embedding,
                size=ask_request.top_k,
                use_hybrid=ask_request.use_hybrid and query_embedding is not None,
            )

            chunks = []
            sources_set = set()
            for hit in search_results.get("hits", []):
                arxiv_id = hit.get("arxiv_id", "")
                chunks.append(
                    {
                        "arxiv_id": arxiv_id,
                        "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                    }
                )
                if arxiv_id:
                    arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

            sources = list(sources_set)

            if not chunks:
                await say("No relevant papers found. Try rephrasing your question.")
                return

            prompt = RAGPromptBuilder().create_rag_prompt(query=query, chunks=chunks)
            ollama_response = await self.ollama.generate(model="llama3.2:1b", prompt=prompt, stream=False)
            answer = ollama_response.get("response", "") if ollama_response else ""

            response = AskResponse(query=query, answer=answer, sources=sources, chunks_used=len(chunks), search_mode="hybrid")

            if self.cache:
                try:
                    await self.cache.store_response(ask_request, response)
                except Exception:
                    pass

            await self._send_answer(say, response)

        except Exception as e:
            logger.error(f"Question handling failed: {e}", exc_info=True)
            await say(f"Error: {str(e)}")

    async def _send_answer(self, say, response: AskResponse) -> None:
        """Send formatted answer using Slack mrkdwn."""
        message = f"*Answer:*\n{response.answer}\n"

        if response.sources:
            message += "\n*Sources:*\n"
            for idx, source_url in enumerate(response.sources[:5], 1):
                arxiv_id = source_url.split("/")[-1].replace(".pdf", "")
                message += f"{idx}. https://arxiv.org/abs/{arxiv_id}\n"

        await say(message)
