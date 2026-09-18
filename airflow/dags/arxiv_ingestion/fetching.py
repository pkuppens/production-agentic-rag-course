import asyncio
import logging
from datetime import datetime, timedelta

from airflow.exceptions import AirflowFailException
from src.exceptions import ArxivAPIClientError
from src.schemas.arxiv.paper import ArxivPaper

from .common import get_cached_services

logger = logging.getLogger(__name__)


def _target_date(context: dict) -> str:
    """Determine the target ingestion date (defaults to yesterday)."""
    execution_date = context.get("execution_date")
    if execution_date:
        return (execution_date - timedelta(days=1)).strftime("%Y%m%d")
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")


def fetch_metadata(**context):
    """Fetch arXiv paper metadata for the target date.

    Deliberately does only the arXiv API call - no PDF download, parsing, or DB
    storage - so this is the only step retried when arXiv throttles a request. arXiv's
    export API answers overload with a bare HTTP 406 rather than 429 (a load-shedding
    signal, not a content-negotiation failure), so `ArxivAPIRateLimitError` covers both
    and is left to propagate for Airflow's normal task retries. `ArxivAPIClientError`
    (other 4xx, e.g. a malformed query) is not retryable and fails the task immediately.
    """
    arxiv_client, _, _, _, _ = get_cached_services()
    target_date = _target_date(context)

    logger.info(f"Fetching arXiv metadata for date: {target_date}")

    try:
        papers = asyncio.run(
            arxiv_client.fetch_papers(
                max_results=arxiv_client.max_results,
                from_date=target_date,
                to_date=target_date,
                sort_by="submittedDate",
                sort_order="descending",
            )
        )
    except ArxivAPIClientError as e:
        raise AirflowFailException(f"arXiv API rejected the request; failing without retry: {e}") from e

    logger.info(f"Fetched {len(papers)} papers for {target_date}")

    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="papers", value=[paper.model_dump() for paper in papers])
        ti.xcom_push(key="target_date", value=target_date)

    return {"papers_fetched": len(papers), "date": target_date}


def process_and_store_papers(**context):
    """Download/parse PDFs and store fetched papers to PostgreSQL.

    Reads paper metadata pushed by `fetch_metadata` via XCom, so retrying this task
    never re-hits the arXiv metadata endpoint. Already-processed papers (from a
    previous run) have their PDF download/parsing skipped - see
    `MetadataFetcher.process_and_store_papers`.
    """
    _, _, database, metadata_fetcher, _ = get_cached_services()

    ti = context.get("ti")
    papers_data = ti.xcom_pull(task_ids="fetch_metadata", key="papers") if ti else None
    target_date = ti.xcom_pull(task_ids="fetch_metadata", key="target_date") if ti else None

    if not papers_data:
        logger.info("No papers to process")
        results = {"papers_fetched": 0, "papers_stored": 0, "date": target_date}
        if ti:
            ti.xcom_push(key="fetch_results", value=results)
        return results

    papers = [ArxivPaper(**paper_data) for paper_data in papers_data]

    with database.get_session() as session:
        results = asyncio.run(
            metadata_fetcher.process_and_store_papers(
                papers=papers,
                process_pdfs=True,
                store_to_db=True,
                db_session=session,
            )
        )

    results["papers_fetched"] = len(papers)
    results["date"] = target_date
    logger.info(f"Processed and stored {results['papers_stored']} papers for {target_date}")

    if ti:
        ti.xcom_push(key="fetch_results", value=results)

    return results
