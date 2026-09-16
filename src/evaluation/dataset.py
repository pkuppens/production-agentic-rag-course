from pathlib import Path

import yaml
from pydantic import BaseModel, TypeAdapter

DEFAULT_GOLDEN_SET_PATH = Path(__file__).parent / "golden_qa.yaml"


class GoldenQAItem(BaseModel):
    """A single hand-written question/expected-answer pair used to drive eval runs."""

    question: str
    expected_answer: str
    reference_paper_id: str | None = None


def load_golden_set(path: str | Path = DEFAULT_GOLDEN_SET_PATH) -> list[GoldenQAItem]:
    """Load and validate the golden QA set from a YAML file.

    :param path: Path to the golden QA YAML file
    :returns: List of validated GoldenQAItem instances
    :raises FileNotFoundError: If the file does not exist
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden QA dataset not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        raw_items = yaml.safe_load(f) or []

    return TypeAdapter(list[GoldenQAItem]).validate_python(raw_items)
