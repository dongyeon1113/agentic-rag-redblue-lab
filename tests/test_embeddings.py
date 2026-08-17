import pytest
from langchain_ollama import OllamaEmbeddings

from services.common.embeddings import (
    DeterministicHashEmbeddings,
    create_embeddings,
)


def test_embedding_factory_defaults_to_deterministic(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    assert isinstance(create_embeddings(), DeterministicHashEmbeddings)


def test_embedding_factory_supports_ollama(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_EMBEDDING_BASE_URL", "http://ollama:11434/")
    assert isinstance(create_embeddings(), OllamaEmbeddings)


def test_embedding_factory_rejects_unknown_backend(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "unknown")
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_BACKEND"):
        create_embeddings()
