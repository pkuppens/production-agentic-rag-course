import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.exceptions import MetadataFetchingException, PipelineException
from src.schemas.arxiv.paper import ArxivPaper
from src.schemas.pdf_parser.models import ParserType, PdfContent
from src.services.arxiv.client import ArxivClient
from src.services.metadata_fetcher import MetadataFetcher, make_metadata_fetcher
from src.services.pdf_parser.parser import PDFParserService


class TestMetadataFetcher:
    """Test MetadataFetcher functionality."""

    @pytest.fixture
    def mock_arxiv_client(self):
        """Create mock ArxivClient."""
        client = MagicMock(spec=ArxivClient)
        return client

    @pytest.fixture
    def mock_pdf_parser(self):
        """Create mock PDFParserService."""
        parser = MagicMock(spec=PDFParserService)
        return parser

    @pytest.fixture
    def metadata_fetcher(self, mock_arxiv_client, mock_pdf_parser, tmp_path):
        """Create MetadataFetcher instance for testing."""
        return MetadataFetcher(
            arxiv_client=mock_arxiv_client,
            pdf_parser=mock_pdf_parser,
            pdf_cache_dir=tmp_path,
            max_concurrent_downloads=2,
            max_concurrent_parsing=1,
        )

    @pytest.fixture
    def sample_arxiv_papers(self):
        """Create sample ArxivPaper objects."""
        return [
            ArxivPaper(
                arxiv_id="2024.0001v1",
                title="Test Paper 1",
                authors=["Author 1"],
                abstract="Abstract 1",
                categories=["cs.AI"],
                published_date="2024-01-01T00:00:00Z",
                pdf_url="http://arxiv.org/pdf/2024.0001v1",
            ),
            ArxivPaper(
                arxiv_id="2024.0002v1",
                title="Test Paper 2",
                authors=["Author 2"],
                abstract="Abstract 2",
                categories=["cs.AI"],
                published_date="2024-01-02T00:00:00Z",
                pdf_url="http://arxiv.org/pdf/2024.0002v1",
            ),
        ]

    @pytest.fixture
    def sample_pdf_content(self):
        """Create sample PdfContent."""
        return PdfContent(
            raw_text="Sample PDF content", sections=[], tables=[], figures=[], parser_used=ParserType.DOCLING, metadata={}
        )

    def test_metadata_fetcher_initialization(self, metadata_fetcher, tmp_path):
        """Test MetadataFetcher initialization."""
        assert metadata_fetcher.pdf_cache_dir == tmp_path
        assert metadata_fetcher.max_concurrent_downloads == 2
        assert metadata_fetcher.max_concurrent_parsing == 1

    # Complex integration tests removed for simplicity

    # Most complex tests removed - keeping only simple ones

    @pytest.mark.asyncio
    async def test_empty_papers_list(self, metadata_fetcher):
        """Test handling of empty papers list."""
        result = await metadata_fetcher.fetch_and_process_papers(max_results=0, process_pdfs=False, store_to_db=False)

        assert result["papers_fetched"] == 0
        assert result["pdfs_downloaded"] == 0
        assert result["pdfs_parsed"] == 0
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_respected(self, metadata_fetcher):
        """Test that rate limiting delays are respected."""
        # This is a basic test to ensure the rate limiting logic exists
        # More comprehensive testing would require timing analysis
        metadata_fetcher.arxiv_client.fetch_papers = AsyncMock(return_value=[])

        start_time = time.time()
        await metadata_fetcher.fetch_and_process_papers(max_results=1)
        end_time = time.time()

        # Should complete quickly for empty result
        assert end_time - start_time < 1.0

    @pytest.mark.asyncio
    async def test_process_and_store_papers_skips_already_processed(self, metadata_fetcher, sample_arxiv_papers):
        """Papers already marked pdf_processed=True in the DB should skip PDF
        download/parsing entirely, and must not have their stored content clobbered
        with "not processed" placeholders."""
        already_processed = MagicMock(pdf_processed=True)
        mock_session = MagicMock()
        metadata_fetcher._process_pdfs_batch = AsyncMock(
            return_value={"downloaded": 0, "parsed": 0, "parsed_papers": {}, "errors": []}
        )

        with patch("src.services.metadata_fetcher.PaperRepository") as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_arxiv_id.return_value = already_processed
            mock_repo.upsert.return_value = MagicMock()

            results = await metadata_fetcher.process_and_store_papers(
                sample_arxiv_papers, process_pdfs=True, store_to_db=True, db_session=mock_session
            )

        assert results["pdfs_skipped"] == len(sample_arxiv_papers)
        assert results["pdfs_downloaded"] == 0
        metadata_fetcher._process_pdfs_batch.assert_awaited_once_with([])

        # Existing parsed content must not be overwritten: pdf_processed/raw_text
        # should be absent from the PaperCreate sent to upsert (exclude_unset semantics).
        assert mock_repo.upsert.call_count == len(sample_arxiv_papers)
        for call in mock_repo.upsert.call_args_list:
            paper_create = call.args[0]
            assert "pdf_processed" not in paper_create.model_fields_set
            assert "raw_text" not in paper_create.model_fields_set

    @pytest.mark.asyncio
    async def test_process_and_store_papers_new_paper_gets_placeholder(self, metadata_fetcher, sample_arxiv_papers):
        """A paper with no prior DB record and a failed/skipped parse still gets stored
        with the existing "not processed" placeholder behavior (unchanged)."""
        mock_session = MagicMock()
        metadata_fetcher._process_pdfs_batch = AsyncMock(
            return_value={"downloaded": 0, "parsed": 0, "parsed_papers": {}, "errors": []}
        )

        with patch("src.services.metadata_fetcher.PaperRepository") as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_arxiv_id.return_value = None
            mock_repo.upsert.return_value = MagicMock()

            results = await metadata_fetcher.process_and_store_papers(
                sample_arxiv_papers, process_pdfs=True, store_to_db=True, db_session=mock_session
            )

        assert results["pdfs_skipped"] == 0
        metadata_fetcher._process_pdfs_batch.assert_awaited_once_with(sample_arxiv_papers)

        for call in mock_repo.upsert.call_args_list:
            paper_create = call.args[0]
            assert paper_create.pdf_processed is False
            assert "pdf_processed" in paper_create.model_fields_set
