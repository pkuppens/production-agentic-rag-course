"""CLI entrypoint that drives the agentic RAG pipeline over the golden QA set.

Run via `uv run python -m src.evaluation.runner` (requires the local stack to be
up: OpenSearch, Ollama, ...); `make eval` is a shorthand for the same command on
platforms with `make` available. This is pure plumbing: it prints
question/expected/actual for each item without scoring the answers.
"""

import asyncio

from src.evaluation.dataset import GoldenQAItem, load_golden_set
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.factory import make_agentic_rag_service
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.factory import make_opensearch_client


def _build_agentic_rag_service() -> AgenticRAGService:
    """Construct the agentic RAG service from freshly built dependency clients."""
    return make_agentic_rag_service(
        opensearch_client=make_opensearch_client(),
        ollama_client=make_ollama_client(),
        embeddings_client=make_embeddings_service(),
        langfuse_tracer=make_langfuse_tracer(),
    )


async def run_eval_pass(items: list[GoldenQAItem], agentic_rag: AgenticRAGService | None = None) -> list[dict]:
    """Run each golden QA item through the agentic RAG pipeline.

    :param items: Golden QA items to evaluate
    :param agentic_rag: Agentic RAG service to drive; built from live clients if omitted
    :returns: List of dicts with question, expected, and actual answer
    """
    agentic_rag = agentic_rag or _build_agentic_rag_service()

    results = []
    for item in items:
        result = await agentic_rag.ask(query=item.question)
        results.append(
            {
                "question": item.question,
                "expected": item.expected_answer,
                "actual": result["answer"],
            }
        )
    return results


def print_results(results: list[dict]) -> None:
    """Print question/expected/actual for each eval result."""
    for i, r in enumerate(results, start=1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(results)}] {r['question']}")
        print(f"{'-' * 80}")
        print(f"Expected: {r['expected']}")
        print(f"{'-' * 80}")
        print(f"Actual:   {r['actual']}")


async def main() -> None:
    items = load_golden_set()
    print(f"Running {len(items)} golden QA items through the agentic RAG pipeline...")
    results = await run_eval_pass(items)
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
