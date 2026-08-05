from unittest.mock import MagicMock

import pytest
from src.config import Settings
from src.exceptions import ConfigurationError
from src.services.opensearch.client import OpenSearchClient


@pytest.fixture
def client() -> OpenSearchClient:
    return OpenSearchClient(host="http://127.0.0.1:9200", settings=Settings())


def make_aggregation_response(buckets: list[dict]) -> dict:
    return {"aggregations": {"embedding_models": {"buckets": buckets}}}


class TestGetIndexedEmbeddingModels:
    def test_returns_empty_dict_when_index_missing(self, client):
        client.client = MagicMock()
        client.client.indices.exists.return_value = False

        assert client.get_indexed_embedding_models() == {}
        client.client.search.assert_not_called()

    def test_returns_label_to_count_mapping(self, client):
        client.client = MagicMock()
        client.client.indices.exists.return_value = True
        client.client.search.return_value = make_aggregation_response([{"key": "ollama:bge-m3", "doc_count": 42}])

        assert client.get_indexed_embedding_models() == {"ollama:bge-m3": 42}

    def test_returns_empty_dict_on_search_error(self, client):
        client.client = MagicMock()
        client.client.indices.exists.return_value = True
        client.client.search.side_effect = Exception("cluster unavailable")

        assert client.get_indexed_embedding_models() == {}


class TestValidateEmbeddingModelConsistency:
    def test_passes_when_index_empty(self, client):
        client.client = MagicMock()
        client.client.indices.exists.return_value = False

        client.validate_embedding_model_consistency("ollama:bge-m3")  # should not raise

    def test_passes_when_only_expected_label_present(self, client):
        client.client = MagicMock()
        client.client.indices.exists.return_value = True
        client.client.search.return_value = make_aggregation_response([{"key": "ollama:bge-m3", "doc_count": 10}])

        client.validate_embedding_model_consistency("ollama:bge-m3")  # should not raise

    def test_raises_when_different_label_present(self, client):
        client.client = MagicMock()
        client.client.indices.exists.return_value = True
        client.client.search.return_value = make_aggregation_response([{"key": "ollama:mxbai-embed-large", "doc_count": 100}])

        with pytest.raises(ConfigurationError, match="ollama:mxbai-embed-large"):
            client.validate_embedding_model_consistency("ollama:bge-m3")

    def test_raises_when_mixed_labels_present(self, client):
        client.client = MagicMock()
        client.client.indices.exists.return_value = True
        client.client.search.return_value = make_aggregation_response(
            [
                {"key": "ollama:bge-m3", "doc_count": 10},
                {"key": "ollama:nomic-embed-text", "doc_count": 5},
            ]
        )

        with pytest.raises(ConfigurationError, match="ollama:nomic-embed-text"):
            client.validate_embedding_model_consistency("ollama:bge-m3")
