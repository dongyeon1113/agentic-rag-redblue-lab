import json
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
                    "tags": json.dumps(record.get("tags", [])),
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
                    tags=self._decode_tags(metadata.get("tags", "[]")),
                    text=document.page_content,
                    score=round(score, 6),
                )
            )
        return hits

    def add_document(
        self,
        *,
        document_id: str,
        source: str,
        trust: str,
        tags: list[str],
        text: str,
    ) -> None:
        if self.contains(document_id):
            raise ValueError(f"Document already exists: {document_id}")

        self.vector_store.add_documents(
            documents=[
                Document(
                    page_content=text,
                    metadata={
                        "document_id": document_id,
                        "source": source,
                        "trust": trust,
                        "tags": json.dumps(tags),
                    },
                )
            ],
            ids=[document_id],
        )

    def contains(self, document_id: str) -> bool:
        return bool(self.vector_store.get(ids=[document_id]).get("ids", []))

    def count(self) -> int:
        return self.vector_store._collection.count()

    def delete_untrusted_documents(self) -> int:
        records = self.vector_store.get(
            where={"trust": "untrusted"},
            include=[],
        )
        document_ids = [str(item) for item in records.get("ids", [])]
        if document_ids:
            self.vector_store.delete(ids=document_ids)
        return len(document_ids)

    def delete_untrusted_document(self, document_id: str) -> bool:
        records = self.vector_store.get(
            ids=[document_id],
            include=["metadatas"],
        )
        ids = records.get("ids", [])
        if not ids:
            return False
        metadatas = records.get("metadatas", [])
        metadata = metadatas[0] if metadatas else {}
        if metadata.get("trust") != "untrusted":
            raise ValueError("Only untrusted experiment documents can be deleted")
        self.vector_store.delete(ids=[document_id])
        return True

    @staticmethod
    def _decode_tags(value: object) -> list[str]:
        if not isinstance(value, str):
            return []
        try:
            tags = json.loads(value)
        except json.JSONDecodeError:
            return []
        if not isinstance(tags, list):
            return []
        return [str(tag) for tag in tags]
