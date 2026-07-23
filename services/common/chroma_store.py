from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from services.common.embeddings import DeterministicHashEmbeddings
from services.common.schemas import SearchHit
from services.common.search import load_json_documents


class ChromaDocumentStore:
    def __init__(
        self,
        data_file: Path,
        *,
        collection_name: str,
        persist_directory: Path | None = None,
        embedding: Embeddings | None = None,
    ) -> None:
        self.data_file = data_file
        self.embedding = embedding or DeterministicHashEmbeddings()
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding,
            persist_directory=(
                str(persist_directory) if persist_directory is not None else None
            ),
        )
        self._index_documents()

    def _index_documents(self) -> None:
        records = load_json_documents(self.data_file)
        documents = [
            Document(
                page_content=record["text"],
                metadata={
                    "document_id": record["id"],
                    "source": record["source"],
                    "trust": record.get("trust", "unknown"),
                },
            )
            for record in records
        ]
        self.vector_store.add_documents(
            documents=documents,
            ids=[record["id"] for record in records],
        )

    def search(self, query: str, limit: int) -> list[SearchHit]:
        results = self.vector_store.similarity_search_with_score(query, k=limit)
        hits: list[SearchHit] = []
        for document, distance in results:
            metadata = document.metadata
            # Chroma returns a distance where a lower value is a closer match.
            score = 1.0 / (1.0 + max(float(distance), 0.0))
            hits.append(
                SearchHit(
                    document_id=str(metadata["document_id"]),
                    source=str(metadata["source"]),
                    trust=str(metadata.get("trust", "unknown")),
                    text=document.page_content,
                    score=round(score, 6),
                )
            )
        return hits
