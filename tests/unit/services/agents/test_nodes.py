"""Tests for agentic RAG node functions using Runtime[Context] pattern."""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from src.services.agents.models import GradeDocuments, GuardrailScoring
from src.services.agents.nodes import (
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_guardrail_step,
    ainvoke_out_of_scope_step,
    ainvoke_retrieve_step,
    ainvoke_rewrite_query_step,
    continue_after_guardrail,
)
from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput
from src.services.agents.nodes.utils import get_latest_context, get_latest_query
from src.services.agents.state import AgentState


def stub_chat_model(context, response):
    """Stub `ollama_client.get_langchain_chat_model` so nodes take their real LLM path.

    Returns the mock chat model. Both `llm.ainvoke` and
    `llm.with_structured_output(...).ainvoke` resolve to `response`.
    """
    llm = Mock()
    llm.ainvoke = AsyncMock(return_value=response)
    llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=response)
    context.ollama_client.get_langchain_chat_model = Mock(return_value=llm)
    return llm


class TestGuardrailNode:
    """Tests for guardrail validation node."""

    def test_continue_after_guardrail_pass(self, test_context):
        """Test routing decision after guardrail pass."""
        state: AgentState = {
            "messages": [],
            "retrieval_attempts": 0,
            "guardrail_result": GuardrailScoring(score=75, reason="Pass"),
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = continue_after_guardrail(state, runtime)

        assert result == "continue"

    def test_continue_after_guardrail_fail(self, test_context):
        """Test routing decision after guardrail fail."""
        state: AgentState = {
            "messages": [],
            "retrieval_attempts": 0,
            "guardrail_result": GuardrailScoring(score=30, reason="Fail"),
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = continue_after_guardrail(state, runtime)

        assert result == "out_of_scope"

    @pytest.mark.asyncio
    async def test_guardrail_step_uses_llm_score(self, test_context, sample_human_message):
        """Guardrail uses the LLM's score, not the fallback (regression test for #41)."""
        llm = stub_chat_model(test_context, GuardrailScoring(score=90, reason="About ML research"))
        state: AgentState = {"messages": [sample_human_message], "retrieval_attempts": 0}
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_guardrail_step(state, runtime)

        assert result["guardrail_result"].score == 90
        test_context.ollama_client.get_langchain_chat_model.assert_called_once_with(
            model=test_context.model_name, temperature=0.0
        )
        llm.with_structured_output.assert_called_once_with(GuardrailScoring)

    @pytest.mark.asyncio
    async def test_guardrail_step_falls_back_when_llm_fails(self, test_context, sample_human_message):
        """If the LLM call fails, guardrail returns the conservative default score."""
        state: AgentState = {"messages": [sample_human_message], "retrieval_attempts": 0}
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_guardrail_step(state, runtime)

        assert result["guardrail_result"].score == 50
        assert "LLM validation failed" in result["guardrail_result"].reason


class TestRetrieveNode:
    """Tests for document retrieval node."""

    @pytest.mark.asyncio
    async def test_retrieve_creates_tool_call(self, test_context, sample_human_message):
        """Test retrieve node creates tool call."""
        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 0,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_retrieve_step(state, runtime)

        assert "retrieval_attempts" in result
        assert result["retrieval_attempts"] == 1
        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["messages"][0].tool_calls) > 0
        assert result["messages"][0].tool_calls[0]["name"] == "retrieve_papers"

    @pytest.mark.asyncio
    async def test_retrieve_max_attempts_reached(self, test_context, sample_human_message):
        """Test retrieve node when max attempts reached."""
        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 2,  # Already at max
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_retrieve_step(state, runtime)

        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        # Check that message indicates failure to find papers
        content_lower = result["messages"][0].content.lower()
        assert "apologize" in content_lower or "unable" in content_lower or "couldn't find" in content_lower


class TestGradeDocumentsNode:
    """Tests for document grading node."""

    @pytest.mark.asyncio
    async def test_grade_documents_relevant(self, test_context, sample_human_message, sample_tool_message):
        """Test grading node with relevant documents."""
        stub_chat_model(
            test_context, GradeDocuments(binary_score="yes", reasoning="Document discusses transformers which is relevant")
        )

        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_grade_documents_step(state, runtime)

        assert result["routing_decision"] == "generate_answer"
        assert result["grading_results"][0].reasoning == "Document discusses transformers which is relevant"

    @pytest.mark.asyncio
    async def test_grade_documents_not_relevant(self, test_context, sample_human_message, sample_tool_message):
        """Test grading node with irrelevant documents."""
        stub_chat_model(test_context, GradeDocuments(binary_score="no", reasoning="Document is not relevant to the query"))

        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_grade_documents_step(state, runtime)

        # The fallback heuristic would call this context relevant, so this proves the LLM verdict is used
        assert result["routing_decision"] == "rewrite_query"
        assert result["grading_results"][0].is_relevant is False


class TestRewriteQueryNode:
    """Tests for query rewriting node."""

    @pytest.mark.asyncio
    async def test_rewrite_query_success(self, test_context, sample_human_message):
        """Test query rewriting with LLM."""
        stub_chat_model(
            test_context,
            QueryRewriteOutput(
                rewritten_query="What are the key concepts in transformer neural network architectures?",
                reasoning="Made the query more specific",
            ),
        )

        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 1,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_rewrite_query_step(state, runtime)

        assert "messages" in result
        assert isinstance(result["messages"][0], HumanMessage)
        assert result["rewritten_query"] == "What are the key concepts in transformer neural network architectures?"
        assert result["messages"][0].content == result["rewritten_query"]


class TestGenerateAnswerNode:
    """Tests for answer generation node."""

    @pytest.mark.asyncio
    async def test_generate_answer_success(self, test_context, sample_human_message, sample_tool_message):
        """Test answer generation with context."""
        stub_chat_model(test_context, AIMessage(content="Based on the papers, transformers are neural network architectures."))

        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_generate_answer_step(state, runtime)

        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Based on the papers, transformers are neural network architectures."


class TestOutOfScopeNode:
    """Tests for out-of-scope handling node."""

    @pytest.mark.asyncio
    async def test_out_of_scope_response(self, test_context, sample_human_message):
        """Test out-of-scope helpful rejection."""
        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 0,
        }
        runtime = Mock(spec=Runtime)
        runtime.context = test_context

        result = await ainvoke_out_of_scope_step(state, runtime)

        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)


class TestNodeUtils:
    """Tests for node utility functions."""

    def test_get_latest_query(self, sample_human_message, sample_ai_message):
        """Test extracting latest query from messages."""
        messages = [sample_human_message, sample_ai_message]
        query = get_latest_query(messages)

        assert query == "What is machine learning?"

    def test_get_latest_query_with_multiple_human_messages(self):
        """Test extracting latest query with multiple human messages."""
        messages = [
            HumanMessage(content="First query"),
            AIMessage(content="First response"),
            HumanMessage(content="Second query"),
        ]
        query = get_latest_query(messages)

        assert query == "Second query"

    def test_get_latest_context(self, sample_tool_message):
        """Test extracting tool message context."""
        messages = [HumanMessage(content="Query"), sample_tool_message]
        context = get_latest_context(messages)

        assert context is not None
        assert "Transformers" in context

    def test_get_latest_context_no_tool_messages(self, sample_human_message):
        """Test extracting context when no tool messages exist."""
        messages = [sample_human_message]
        context = get_latest_context(messages)

        assert context == ""
