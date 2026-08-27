from __future__ import annotations

import math
from hashlib import blake2b
from typing import Protocol

import httpx


class EmbeddingClient(Protocol):
    @property
    def model(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbeddingClient:
    """Synchronous embedding client used while indexing and querying Chroma."""

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = httpx.post(
                f"{self._base_url}/api/embed",
                json={
                    "model": self._model,
                    "input": texts,
                    "truncate": True,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama embedding model {self._model} is unavailable: {exc}"
            ) from exc
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding batch")
        return [[float(value) for value in vector] for vector in embeddings]


class DeterministicEmbeddingClient:
    """Offline test double; production apps always use OllamaEmbeddingClient."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    @property
    def model(self) -> str:
        return "deterministic-test-embedding"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.casefold().replace("\n", " ").split():
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector

