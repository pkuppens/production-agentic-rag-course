from pathlib import Path

import pytest
from pydantic import ValidationError
from src.evaluation.dataset import DEFAULT_GOLDEN_SET_PATH, GoldenQAItem, load_golden_set


class TestGoldenQAItem:
    """Tests for the GoldenQAItem schema."""

    def test_valid_item_with_reference_paper_id(self):
        item = GoldenQAItem(
            question="What is RAG?", expected_answer="Retrieval-augmented generation.", reference_paper_id="2401.00001"
        )
        assert item.question == "What is RAG?"
        assert item.reference_paper_id == "2401.00001"

    def test_reference_paper_id_defaults_to_none(self):
        item = GoldenQAItem(question="What is RAG?", expected_answer="Retrieval-augmented generation.")
        assert item.reference_paper_id is None

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            GoldenQAItem(question="What is RAG?")


class TestLoadGoldenSet:
    """Tests for load_golden_set."""

    def test_loads_default_golden_set(self):
        items = load_golden_set()
        assert 10 <= len(items) <= 20
        assert all(isinstance(item, GoldenQAItem) for item in items)
        assert all(item.question.strip() for item in items)
        assert all(item.expected_answer.strip() for item in items)

    def test_default_path_points_at_shipped_file(self):
        assert DEFAULT_GOLDEN_SET_PATH.name == "golden_qa.yaml"
        assert DEFAULT_GOLDEN_SET_PATH.exists()

    def test_missing_file_raises_file_not_found_error(self, tmp_path: Path):
        missing_path = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_golden_set(missing_path)

    def test_loads_from_explicit_path(self, tmp_path: Path):
        custom_path = tmp_path / "custom_golden_qa.yaml"
        custom_path.write_text("- question: Custom question?\n  expected_answer: Custom answer.\n", encoding="utf-8")

        items = load_golden_set(custom_path)

        assert len(items) == 1
        assert items[0].question == "Custom question?"
        assert items[0].expected_answer == "Custom answer."

    def test_invalid_item_schema_raises_validation_error(self, tmp_path: Path):
        bad_path = tmp_path / "bad_golden_qa.yaml"
        bad_path.write_text("- question: Missing expected answer\n", encoding="utf-8")

        with pytest.raises(ValidationError):
            load_golden_set(bad_path)
