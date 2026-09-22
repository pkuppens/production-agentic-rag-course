from langchain_ollama import ChatOllama
from src.config import Settings
from src.services.ollama.client import OllamaClient


def make_settings() -> Settings:
    return Settings()


def test_get_langchain_chat_model_returns_chat_ollama_bound_to_client_host():
    """The LangGraph agent nodes need a BaseChatModel wired to this client's Ollama host."""
    client = OllamaClient(make_settings())

    model = client.get_langchain_chat_model(model="llama3.2:1b", temperature=0.3)

    assert isinstance(model, ChatOllama)
    assert model.model == "llama3.2:1b"
    assert model.base_url == client.base_url
    assert model.temperature == 0.3


def test_get_langchain_chat_model_defaults_temperature_to_zero():
    client = OllamaClient(make_settings())

    model = client.get_langchain_chat_model(model="llama3.2:1b")

    assert model.temperature == 0.0
