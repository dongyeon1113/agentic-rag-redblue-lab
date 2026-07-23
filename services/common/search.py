import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from services.common.schemas import SearchHit

TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")
STOPWORDS = {
    "a",
    "an",
    "are",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for raw_token in TOKEN_PATTERN.findall(text)
        if (token := raw_token.lower()) not in STOPWORDS
    ]


def lexical_score(query: str, document: str) -> float:
    query_counts = Counter(tokenize(query))
    document_counts = Counter(tokenize(document))
    if not query_counts or not document_counts:
        return 0.0

    overlap = sum(
        min(query_count, document_counts[token])
        for token, query_count in query_counts.items()
    )
    return round(overlap / sum(query_counts.values()), 6)


class JsonDocumentStore:
    def __init__(self, data_file: Path) -> None:
        self.data_file = data_file
        self.documents = self._load()

    def _load(self) -> list[dict[str, Any]]:
        with self.data_file.open(encoding="utf-8") as file:
            documents = json.load(file)
        if not isinstance(documents, list):
            raise ValueError(f"{self.data_file} must contain a JSON list")
        return documents

    def search(self, query: str, limit: int) -> list[SearchHit]:
        ranked: list[SearchHit] = []
        for document in self.documents:
            score = lexical_score(query, document["text"])
            if score <= 0:
                continue
            ranked.append(
                SearchHit(
                    document_id=document["id"],
                    source=document["source"],
                    trust=document.get("trust", "unknown"),
                    text=document["text"],
                    score=score,
                )
            )
        ranked.sort(key=lambda hit: (-hit.score, hit.document_id))
        return ranked[:limit]
