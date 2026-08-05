"""Shared fixtures for agentic RAG unit tests.

Provides mocked dependencies (OpenSearch, Ollama, embeddings) and sample
LangChain messages used across test_agentic_rag.py, test_nodes.py, and
test_tools.py.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.services.agents.context import Context


@pytest.fixture
def mock_opensearch_client():
    """Mock OpenSearch client with a default two-hit search_unified response."""
    client = Mock()
    client.search_unified = Mock(
        return_value={
            "hits": [
                {
                    "chunk_text": "Transformers are neural network architectures based on self-attention mechanisms.",
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "authors": "Ashish Vaswani, Noam Shazeer",
                    "score": 0.95,
                    "section_name": "Introduction",
                },
                {
                    "chunk_text": "Self-attention allows the model to weigh the importance of different words.",
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "authors": "Ashish Vaswani, Noam Shazeer",
                    "score": 0.87,
                    "section_name": "Model Architecture",
                },
            ]
        }
    )
    return client


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client. LLM calls are not awaitable by default, so agent
    nodes take their fallback/heuristic path — matching how these nodes
    behave when the underlying LLM call fails in production."""
    return Mock()


@pytest.fixture
def mock_jina_embeddings_client():
    """Mock Jina embeddings client."""
    client = Mock()
    client.embed_query = AsyncMock(return_value=[0.1] * 1024)
    client.embed_passages = AsyncMock(return_value=[[0.1] * 1024])
    return client


@pytest.fixture
def test_context(mock_opensearch_client, mock_ollama_client, mock_jina_embeddings_client):
    """Runtime context for agent node tests."""
    return Context(
        ollama_client=mock_ollama_client,
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        langfuse_tracer=None,
        trace=None,
        langfuse_enabled=False,
        model_name="llama3.2:1b",
        temperature=0.0,
        top_k=3,
        max_retrieval_attempts=2,
        guardrail_threshold=60,
    )


@pytest.fixture
def sample_human_message():
    return HumanMessage(content="What is machine learning?")


@pytest.fixture
def sample_ai_message():
    return AIMessage(content="Machine learning is a subset of artificial intelligence.")


@pytest.fixture
def sample_tool_message():
    return ToolMessage(
        content="Transformers are neural network architectures based on self-attention mechanisms.",
        tool_call_id="test_tool_call_1",
        name="retrieve_papers",
    )
