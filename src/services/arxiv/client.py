import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote, urlencode

import httpx
from src.config import ArxivSettings
from src.exceptions import (
    ArxivAPIClientError,
    ArxivAPIException,
    ArxivAPIRateLimitError,
    ArxivAPIServerError,
    ArxivAPITimeoutError,
    ArxivParseError,
    PDFDownloadException,
    PDFDownloadTimeoutError,
)
from src.schemas.arxiv.paper import ArxivPaper

logger = logging.getLogger(__name__)


class ArxivClient:
    """Client for fetching papers from arXiv API.

    arXiv's export API has known quirks under load: it answers overload with a bare
    HTTP 406 (not 429), and occasionally with an HTTP 200 wrapping a single
    "<entry title='Error'>" feed instead - both for requests whose parameters are
    individually valid, with the byte-identical request succeeding moments later.
    See `_get_with_retry` for how this is handled - it retries, but it isn't fast:
    a sustained overload window can take on the order of a minute to clear (worst
    case ~60-70s with the default `metadata_max_retries`/`metadata_max_retry_delay`,
    confirmed by live testing during #39), so callers should not assume this
    client resolves quickly.

    Separately, arXiv's `submittedDate:[...]` range-query syntax is unconditionally
    rejected with a 406, independent of the date/category/encoding - not throttling,
    and retrying it does not help (confirmed live: 19/19 retries failed over ~185s
    across 6 trials). `fetch_papers`'s `from_date`/`to_date` filtering no longer uses
    that syntax; see its docstring and docs/406.md for the full investigation.
    """

    def __init__(self, settings: ArxivSettings):
        self._settings = settings
        self._last_request_time: Optional[float] = None

    @cached_property
    def pdf_cache_dir(self) -> Path:
        """PDF cache directory."""
        cache_dir = Path(self._settings.pdf_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    @property
    def namespaces(self) -> dict:
        return self._settings.namespaces

    @property
    def rate_limit_delay(self) -> float:
        return self._settings.rate_limit_delay

    @property
    def timeout_seconds(self) -> int:
        return self._settings.timeout_seconds

    @property
    def max_results(self) -> int:
        return self._settings.max_results

    @property
    def search_category(self) -> str:
        return self._settings.search_category

    def _classify_http_status_error(self, status_code: int, message: str) -> ArxivAPIException:
        """Map an arXiv HTTP status code to a retryable/non-retryable exception.

        429 and 406 are both treated as throttling (arXiv's export API answers
        overload with a bare 406, not 429) and are retryable; other 4xx are
        permanent client errors; 5xx are retryable server errors.
        """
        if status_code in (429, 406):
            return ArxivAPIRateLimitError(message)
        if 500 <= status_code < 600:
            return ArxivAPIServerError(message)
        return ArxivAPIClientError(message)

    def _extract_embedded_error(self, xml_data: str) -> Optional[str]:
        """Detect arXiv's malformed-query error format.

        Instead of a 4xx status, arXiv sometimes answers with HTTP 200 and a feed
        containing exactly one entry titled "Error", linking to the API user manual,
        with the real message in <summary>. Observed in practice even for a query
        whose parameters are individually valid (e.g. sortOrder=descending rejected
        with "sortOrder must be in: ascending, descending") while the byte-identical
        request succeeds moments later - i.e. the same server-side flakiness as the
        406s, just surfaced as a 200 instead of an HTTP error status. Returns the
        embedded error message, or None if this isn't that shape.
        """
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            return None

        entries = root.findall("atom:entry", self.namespaces)
        if len(entries) != 1:
            return None

        entry = entries[0]
        title = self._get_text(entry, "atom:title")
        entry_id = self._get_text(entry, "atom:id")
        if title == "Error" and "user-manual" in entry_id:
            return self._get_text(entry, "atom:summary") or "(no message)"
        return None

    def _worst_case_wait_seconds(self) -> float:
        """Total backoff time `_get_with_retry` sleeps across a fully-exhausted retry
        run (excludes the per-request round trip itself), for error messages/logging -
        so "it's slow, not stuck" is stated in seconds, not just implied."""
        max_retries = self._settings.metadata_max_retries
        return sum(
            min(self.rate_limit_delay * (2**attempt), self._settings.metadata_max_retry_delay)
            for attempt in range(max_retries - 1)
        )

    async def _get_with_retry(self, url: str, context: str = "") -> str:
        """GET an arXiv API URL, honoring arXiv's documented rate-limit policy.

        arXiv's API manual requires at least 3 seconds between requests and a
        single connection at a time (https://info.arxiv.org/help/api/user-manual.html) -
        `rate_limit_delay` enforces that spacing before every attempt, including
        retries. On top of that policy, this retries a bounded number of times with
        backoff when arXiv itself signals throttling/overload (406, 429, 5xx - see
        `_classify_http_status_error` - or its embedded-error-in-a-200 shape, see
        `_extract_embedded_error`), since a single request already complying with
        the documented policy can still be throttled by server-side load shedding.
        Non-retryable errors (other 4xx, timeouts) propagate immediately.

        This is deliberately slow to give up: with the default settings
        (`metadata_max_retries=8`, `metadata_max_retry_delay=10.0`), a fully-exhausted
        retry run sleeps ~59s before raising - confirmed against a real arXiv overload
        window while fixing #39, where a plain query failed 406 on 8/8 consecutive
        attempts over ~70s (including request time) before succeeding on the very
        next attempt. Callers (e.g. the Airflow `fetch_metadata` task) should expect
        this call to occasionally take the better part of a minute, not fail fast.

        :param url: Fully-built arXiv API query URL
        :param context: Optional message suffix for logging/errors (e.g. " for paper X")
        :returns: Raw XML response text
        """
        max_retries = self._settings.metadata_max_retries

        for attempt in range(max_retries):
            if self._last_request_time is not None:
                time_since_last = time.time() - self._last_request_time
                if time_since_last < self.rate_limit_delay:
                    await asyncio.sleep(self.rate_limit_delay - time_since_last)

            self._last_request_time = time.time()
            error: Optional[ArxivAPIException] = None

            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    xml_data = response.text

                embedded_error = self._extract_embedded_error(xml_data)
                if embedded_error is None:
                    return xml_data
                error = ArxivAPIServerError(f"arXiv API returned an embedded error{context}: {embedded_error}")
            except httpx.TimeoutException as e:
                logger.error(f"arXiv API timeout{context}: {e}")
                raise ArxivAPITimeoutError(f"arXiv API request timed out{context}: {e}") from e
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                error = self._classify_http_status_error(status_code, f"arXiv API returned error {status_code}{context}: {e}")
            except Exception as e:
                logger.error(f"Failed to fetch from arXiv{context}: {e}")
                raise ArxivAPIException(f"Unexpected error fetching from arXiv{context}: {e}") from e

            if isinstance(error, ArxivAPIClientError):
                logger.error(str(error))
                raise error
            if attempt == max_retries - 1:
                # Re-raise as the same (retryable) exception type, so callers that
                # distinguish ArxivAPIRateLimitError/ArxivAPIServerError still can -
                # just with the message enriched to say this is arXiv's known
                # transient-overload quirk, not a permanent rejection of this query.
                enriched_message = (
                    f"{error} (gave up after {max_retries} attempts{context}, ~{self._worst_case_wait_seconds():.0f}s "
                    "of retrying). This matches arXiv's known transient overload behavior (bare 406s / "
                    "embedded-error-in-a-200 responses - see ArxivClient docstring) which normally clears within "
                    "about a minute, rather than a permanent rejection of this query; retrying the whole ingestion "
                    "run later will likely succeed."
                )
                logger.error(enriched_message)
                raise type(error)(enriched_message) from error
            # Generous but bounded: this isn't time-critical, so keep retrying, but
            # cap the interval rather than letting exponential backoff run into minutes.
            # arXiv's overload windows are known to run tens of seconds, not milliseconds,
            # so don't expect this to resolve on the first or second retry.
            wait_time = min(self.rate_limit_delay * (2**attempt), self._settings.metadata_max_retry_delay)
            logger.warning(
                f"{error} (attempt {attempt + 1}/{max_retries}) - known arXiv overload quirk, not necessarily a bad "
                f"query; retrying in {wait_time:.0f}s"
            )
            await asyncio.sleep(wait_time)

        raise ArxivAPIException(f"arXiv API request failed{context}: metadata_max_retries is set to {max_retries}")

    async def fetch_papers(
        self,
        max_results: Optional[int] = None,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[ArxivPaper]:
        """
        Fetch papers from arXiv for the configured category.

        Args:
            max_results: Maximum number of papers to fetch (uses settings default if None)
            start: Starting index for pagination
            sort_by: Sort criteria (submittedDate, lastUpdatedDate, relevance)
            sort_order: Sort order (ascending, descending)
            from_date: Filter papers submitted after this date (format: YYYYMMDD)
            to_date: Filter papers submitted before this date (format: YYYYMMDD)

        Returns:
            List of ArxivPaper objects for the configured category

        Date filtering does NOT use arXiv's `submittedDate:[...]` range-query syntax -
        that syntax is currently rejected with an unconditional 406, independent of
        the date value, category, or bracket encoding (see docs/406.md). It also
        turns out to be an especially bad match for arXiv's throttling behavior:
        every `submittedDate:[...]` query is uniquely parameterized (different bounds
        every call), so it almost never hits arXiv's shared response cache, and a
        cache-miss query gets a 406 with an empty body on every attempt while this
        host is throttled - confirmed live: 19/19 retries failed across 6
        independent trials (~185s each, 0% success), while the plain, cacheable
        `cat:{category}` query (no date bounds) succeeded 6/6. So instead, this
        fetches that same plain/cacheable query over a wider window
        (`date_filter_scan_results` results), sorted by submittedDate descending,
        and filters to the requested date range client-side.
        """
        if max_results is None:
            max_results = self.max_results

        search_query = f"cat:{self.search_category}"

        # Scan a wider window than requested when date-filtering, then trim to
        # max_results after filtering - see docstring above for why this can't be
        # done with arXiv's submittedDate:[...] syntax.
        fetch_count = max(max_results, self._settings.date_filter_scan_results) if (from_date or to_date) else max_results

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(fetch_count, 2000),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        # `[`/`]` must NOT be left unescaped: arXiv's edge rejects literal brackets in
        # the query string with a 406 (confirmed by direct A/B testing - not a
        # transient/throttle issue, a hard requirement of arXiv's request handling).
        # Kept here for any other bracket-using callers of this safe-charset; date
        # ranges no longer use it (see docstring above).
        safe = ":+"  # Don't encode :, + characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        logger.info(f"Fetching {fetch_count} {self.search_category} papers from arXiv")
        xml_data = await self._get_with_retry(url)
        papers = self._parse_response(xml_data)

        if from_date or to_date:
            papers = [p for p in papers if self._in_date_range(p.published_date, from_date, to_date)]
            papers = papers[:max_results]

        logger.info(f"Fetched {len(papers)} papers")

        return papers

    @staticmethod
    def _in_date_range(published_date: str, from_date: Optional[str], to_date: Optional[str]) -> bool:
        """Check an ISO `published_date` (e.g. "2026-09-20T12:34:56Z") against an
        inclusive YYYYMMDD `from_date`/`to_date` range. Used to filter client-side
        instead of arXiv's broken `submittedDate:[...]` range query - see
        `fetch_papers`'s docstring."""
        date_part = published_date[:10].replace("-", "")
        if from_date and date_part < from_date:
            return False
        if to_date and date_part > to_date:
            return False
        return True

    async def fetch_papers_with_query(
        self,
        search_query: str,
        max_results: Optional[int] = None,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> List[ArxivPaper]:
        """
        Fetch papers from arXiv using a custom search query.

        Args:
            search_query: Custom arXiv search query (e.g., "au:LeCun AND cat:cs.AI").
                Do NOT include a `submittedDate:[...]`/`lastUpdatedDate:[...]` range
                clause here - that syntax is currently rejected with an
                unconditional 406 by arXiv's export API, regardless of the date
                value or encoding (see docs/406.md). Use `fetch_papers`'s
                `from_date`/`to_date` instead, which filters client-side.
            max_results: Maximum number of papers to fetch (uses settings default if None)
            start: Starting index for pagination
            sort_by: Sort criteria (submittedDate, lastUpdatedDate, relevance)
            sort_order: Sort order (ascending, descending)

        Returns:
            List of ArxivPaper objects matching the search query

        Examples:
            # Papers by specific author
            "au:LeCun AND cat:cs.AI"

            # Papers with specific keywords in title
            "ti:transformer AND cat:cs.AI"
        """
        if max_results is None:
            max_results = self.max_results

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(max_results, 2000),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        # `[`/`]` must NOT be left unescaped - see fetch_papers for why.
        safe = ":+*"  # Don't encode :, +, * characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        xml_data = await self._get_with_retry(url)
        papers = self._parse_response(xml_data)
        logger.info(f"Query returned {len(papers)} papers")

        return papers

    async def fetch_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Fetch a specific paper by its arXiv ID.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2507.17748v1" or "2507.17748")

        Returns:
            ArxivPaper object or None if not found
        """
        # Clean the arXiv ID (remove version if needed for search)
        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
        params = {"id_list": clean_id, "max_results": 1}

        # `[`/`]` must NOT be left unescaped - see fetch_papers for why.
        safe = ":+*"  # Don't encode :, +, * characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        xml_data = await self._get_with_retry(url, context=f" for paper {arxiv_id}")
        papers = self._parse_response(xml_data)

        if papers:
            return papers[0]
        else:
            logger.warning(f"Paper {arxiv_id} not found")
            return None

    def _parse_response(self, xml_data: str) -> List[ArxivPaper]:
        """
        Parse arXiv API XML response into ArxivPaper objects.

        Args:
            xml_data: Raw XML response from arXiv API

        Returns:
            List of parsed ArxivPaper objects
        """
        try:
            root = ET.fromstring(xml_data)
            entries = root.findall("atom:entry", self.namespaces)

            papers = []
            for entry in entries:
                paper = self._parse_single_entry(entry)
                if paper:
                    papers.append(paper)

            return papers

        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML response: {e}")
            raise ArxivParseError(f"Failed to parse arXiv XML response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing arXiv response: {e}")
            raise ArxivParseError(f"Unexpected error parsing arXiv response: {e}")

    def _parse_single_entry(self, entry: ET.Element) -> Optional[ArxivPaper]:
        """
        Parse a single entry from arXiv XML response.

        Args:
            entry: XML entry element

        Returns:
            ArxivPaper object or None if parsing fails
        """
        try:
            # Extract basic metadata
            arxiv_id = self._get_arxiv_id(entry)
            if not arxiv_id:
                return None

            title = self._get_text(entry, "atom:title", clean_newlines=True)
            authors = self._get_authors(entry)
            abstract = self._get_text(entry, "atom:summary", clean_newlines=True)
            published = self._get_text(entry, "atom:published")
            categories = self._get_categories(entry)
            pdf_url = self._get_pdf_url(entry)

            return ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published_date=published,
                categories=categories,
                pdf_url=pdf_url,
            )

        except Exception as e:
            logger.error(f"Failed to parse entry: {e}")
            return None

    def _get_text(self, element: ET.Element, path: str, clean_newlines: bool = False) -> str:
        """
        Extract text from XML element safely.

        Args:
            element: Parent XML element
            path: XPath to find the text element
            clean_newlines: Whether to replace newlines with spaces

        Returns:
            Extracted text or empty string
        """
        elem = element.find(path, self.namespaces)
        if elem is None or elem.text is None:
            return ""

        text = elem.text.strip()
        return text.replace("\n", " ") if clean_newlines else text

    def _get_arxiv_id(self, entry: ET.Element) -> Optional[str]:
        """
        Extract arXiv ID from entry.

        Args:
            entry: XML entry element

        Returns:
            arXiv ID or None
        """
        id_elem = entry.find("atom:id", self.namespaces)
        if id_elem is None or id_elem.text is None:
            return None
        return id_elem.text.split("/")[-1]

    def _get_authors(self, entry: ET.Element) -> List[str]:
        """
        Extract author names from entry.

        Args:
            entry: XML entry element

        Returns:
            List of author names
        """
        authors = []
        for author in entry.findall("atom:author", self.namespaces):
            name = self._get_text(author, "atom:name")
            if name:
                authors.append(name)
        return authors

    def _get_categories(self, entry: ET.Element) -> List[str]:
        """
        Extract categories from entry.

        Args:
            entry: XML entry element

        Returns:
            List of category terms
        """
        categories = []
        for category in entry.findall("atom:category", self.namespaces):
            term = category.get("term")
            if term:
                categories.append(term)
        return categories

    def _get_pdf_url(self, entry: ET.Element) -> str:
        """
        Extract PDF URL from entry links.

        Args:
            entry: XML entry element

        Returns:
            PDF URL or empty string (always HTTPS)
        """
        for link in entry.findall("atom:link", self.namespaces):
            if link.get("type") == "application/pdf":
                url = link.get("href", "")
                # Convert HTTP to HTTPS for arXiv URLs
                if url.startswith("http://arxiv.org/"):
                    url = url.replace("http://arxiv.org/", "https://arxiv.org/")
                return url
        return ""

    async def download_pdf(self, paper: ArxivPaper, force_download: bool = False) -> Optional[Path]:
        """
        Download PDF for a given paper to local cache.

        Args:
            paper: ArxivPaper object containing PDF URL
            force_download: Force re-download even if file exists

        Returns:
            Path to downloaded PDF file or None if download failed
        """
        if not paper.pdf_url:
            logger.error(f"No PDF URL for paper {paper.arxiv_id}")
            return None

        pdf_path = self._get_pdf_path(paper.arxiv_id)

        # Return cached PDF if exists
        if pdf_path.exists() and not force_download:
            logger.info(f"Using cached PDF: {pdf_path.name}")
            return pdf_path

        # Download with retry
        if await self._download_with_retry(paper.pdf_url, pdf_path):
            return pdf_path
        else:
            return None

    def _get_pdf_path(self, arxiv_id: str) -> Path:
        """
        Get the local path for a PDF file.

        Args:
            arxiv_id: arXiv paper ID

        Returns:
            Path object for the PDF file
        """
        safe_filename = arxiv_id.replace("/", "_") + ".pdf"
        return self.pdf_cache_dir / safe_filename

    async def _download_with_retry(self, url: str, path: Path, max_retries: Optional[int] = None) -> bool:
        """Download a file with retry logic."""
        if max_retries is None:
            max_retries = self._settings.download_max_retries

        logger.info(f"Downloading PDF from {url}")

        # Respect rate limits
        await asyncio.sleep(self.rate_limit_delay)

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=float(self.timeout_seconds)) as client:
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        with open(path, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
                logger.info(f"Successfully downloaded to {path.name}")
                return True

            except httpx.TimeoutException as e:
                if attempt < max_retries - 1:
                    wait_time = self._settings.download_retry_delay_base * (attempt + 1)
                    logger.warning(f"PDF download timeout (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"PDF download failed after {max_retries} attempts due to timeout: {e}")
                    raise PDFDownloadTimeoutError(f"PDF download timed out after {max_retries} attempts: {e}")
            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = self._settings.download_retry_delay_base * (attempt + 1)  # Exponential backoff
                    logger.warning(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise PDFDownloadException(f"PDF download failed after {max_retries} attempts: {e}")
            except Exception as e:
                logger.error(f"Unexpected download error: {e}")
                raise PDFDownloadException(f"Unexpected error during PDF download: {e}")

        # Clean up partial download
        if path.exists():
            path.unlink()

        return False
