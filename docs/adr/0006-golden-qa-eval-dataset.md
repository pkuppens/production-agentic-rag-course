# Golden QA set as a fixed input for eval

`src/evaluation/golden_qa.yaml` holds a small (10-20 item) hand-written set of CS.AI research questions and expected answers, loaded and validated via `GoldenQAItem` (`src/evaluation/dataset.py`) and driven end-to-end through the existing agentic RAG graph by `src/evaluation/runner.py`, run via `uv run python -m src.evaluation.runner` (or `make eval` as a shorthand where `make` is available — the course also supports Windows without it). This gives the eval harness a fixed, reviewable input rather than relying purely on live LLM-graded judgments with no ground truth to check them against.

`reference_paper_id` is optional and left unset on every current item: the `arxiv_ingestion` Airflow DAG fetches recent cs.AI papers by recency rather than a fixed list, so the locally ingested paper set (and therefore which arXiv IDs are actually retrievable) varies by when the DAG last ran. Items here are written to be answerable from cs.AI papers generally, not pinned to specific IDs that may not be indexed on a given machine.

This ticket is pure plumbing — dataset format, loader, and a runner that proves the pipeline can be driven from a fixed input set. No scoring is implemented yet; that lands in a follow-up ADR/ticket.
