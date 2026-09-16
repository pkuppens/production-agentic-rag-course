from unittest.mock import AsyncMock, Mock

import pytest
from src.evaluation.dataset import GoldenQAItem
from src.evaluation.runner import print_results, run_eval_pass


class TestRunEvalPass:
    """Tests for run_eval_pass."""

    @pytest.mark.asyncio
    async def test_runs_each_item_through_the_injected_service(self):
        items = [
            GoldenQAItem(question="What is RAG?", expected_answer="Retrieval-augmented generation."),
            GoldenQAItem(question="What is a transformer?", expected_answer="An attention-based architecture."),
        ]
        agentic_rag = Mock()
        agentic_rag.ask = AsyncMock(
            side_effect=[
                {"answer": "RAG is retrieval-augmented generation."},
                {"answer": "A transformer is a neural network architecture."},
            ]
        )

        results = await run_eval_pass(items, agentic_rag=agentic_rag)

        assert results == [
            {
                "question": "What is RAG?",
                "expected": "Retrieval-augmented generation.",
                "actual": "RAG is retrieval-augmented generation.",
            },
            {
                "question": "What is a transformer?",
                "expected": "An attention-based architecture.",
                "actual": "A transformer is a neural network architecture.",
            },
        ]
        assert agentic_rag.ask.call_args_list[0].kwargs == {"query": "What is RAG?"}
        assert agentic_rag.ask.call_args_list[1].kwargs == {"query": "What is a transformer?"}

    @pytest.mark.asyncio
    async def test_empty_items_returns_empty_results(self):
        agentic_rag = Mock()
        agentic_rag.ask = AsyncMock()

        results = await run_eval_pass([], agentic_rag=agentic_rag)

        assert results == []
        agentic_rag.ask.assert_not_called()


class TestPrintResults:
    """Tests for print_results."""

    def test_prints_question_expected_and_actual(self, capsys):
        results = [
            {
                "question": "What is RAG?",
                "expected": "Retrieval-augmented generation.",
                "actual": "RAG combines retrieval and generation.",
            }
        ]

        print_results(results)

        captured = capsys.readouterr()
        assert "What is RAG?" in captured.out
        assert "Retrieval-augmented generation." in captured.out
        assert "RAG combines retrieval and generation." in captured.out

    def test_empty_results_prints_nothing(self, capsys):
        print_results([])

        captured = capsys.readouterr()
        assert captured.out == ""
