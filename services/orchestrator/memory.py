import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from services.common.schemas import MemoryRecord, SearchHit
from services.common.search import lexical_score

MEMORY_SOURCE = "agent-memory"


class ConversationMemory:
    """Append-only long-term memory of past orchestrator turns.

    Records are persisted as JSONL so memory survives container restarts.
    A record is `trusted` only when every passage used for the answer was
    trusted; poisoned turns stay `untrusted` and are hidden in defended mode.
    """

    def __init__(self, memory_file: Path, *, max_records: int = 500) -> None:
        self.memory_file = memory_file
        self.max_records = max_records
        self.records = self._load()

    def _load(self) -> list[MemoryRecord]:
        if not self.memory_file.is_file():
            return []
        records = []
        for line in self.memory_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(MemoryRecord.model_validate_json(line))
        return records[-self.max_records :]

    def _rewrite(self) -> None:
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(
            "".join(
                f"{record.model_dump_json()}\n"
                for record in self.records
            ),
            encoding="utf-8",
        )

    def append(
        self,
        *,
        session_id: str,
        query: str,
        answer: str,
        trust: str,
    ) -> MemoryRecord:
        record = MemoryRecord(
            memory_id=f"memory-{uuid4().hex[:12]}",
            session_id=session_id,
            query=query,
            answer=answer,
            trust=trust,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]
            self._rewrite()
            return record

        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_file.open("a", encoding="utf-8") as file:
            file.write(f"{record.model_dump_json()}\n")
        return record

    def recall(
        self,
        query: str,
        *,
        session_id: str,
        limit: int,
        trusted_only: bool,
    ) -> list[MemoryRecord]:
        scored: list[MemoryRecord] = []
        for record in self.records:
            if record.session_id != session_id:
                continue
            if trusted_only and record.trust != "trusted":
                continue
            score = lexical_score(query, f"{record.query} {record.answer}")
            if score <= 0:
                continue
            scored.append(record.model_copy(update={"score": score}))
        scored.sort(key=lambda record: (-record.score, record.created_at))
        return scored[:limit]

    def list(self, *, session_id: str | None, limit: int) -> list[MemoryRecord]:
        records = [
            record
            for record in self.records
            if session_id is None or record.session_id == session_id
        ]
        return records[-limit:]

    def clear(self, *, session_id: str | None) -> int:
        kept = [
            record
            for record in self.records
            if session_id is not None and record.session_id != session_id
        ]
        deleted_count = len(self.records) - len(kept)
        self.records = kept
        self._rewrite()
        return deleted_count


def memory_to_hit(record: MemoryRecord) -> SearchHit:
    return SearchHit(
        document_id=record.memory_id,
        source=MEMORY_SOURCE,
        trust=record.trust,
        tags=["memory", record.session_id],
        text=f"Earlier turn. Question: {record.query} Answer: {record.answer}",
        score=record.score,
    )
