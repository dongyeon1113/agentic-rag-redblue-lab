"""Build a reproducible 10,000-document BEIR NQ experiment corpus.

The selected corpus always contains every qrels document for the supplied
RAGDefender target queries. Remaining documents are selected with a fixed seed.
Raw upstream files live under datasets/raw/ and are intentionally Git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "datasets/raw/nq-corpus.parquet"
DEFAULT_QRELS = PROJECT_ROOT / "datasets/raw/nq-qrels-test.tsv"
DEFAULT_TARGETS = PROJECT_ROOT / "datasets/raw/ragdefender-nq-target-queries.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets/generated/nq_10000.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets/generated/nq_10000.manifest.json"
DEFAULT_EXPERIMENTS = PROJECT_ROOT / "datasets/experiments/nq_target_queries.json"

SOURCE_URLS = {
    "corpus": "https://huggingface.co/datasets/BeIR/nq/resolve/main/corpus/corpus-00000-of-00001.parquet",
    "qrels": "https://huggingface.co/datasets/BeIR/nq-qrels/resolve/main/test.tsv",
    "targets": "https://github.com/SecAI-Lab/RAGDefender/blob/ba2a17efba165d45409114df2d70b030ade1e1b8/artifacts/results/target_queries/nq.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_targets(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("Target query file must contain a non-empty JSON list")
    required = {"id", "question", "correct answer", "incorrect answer"}
    targets: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError("Target query entry has an invalid schema")
        targets.append({key: str(item[key]).strip() for key in required})
    if len({item["id"] for item in targets}) != len(targets):
        raise ValueError("Target query IDs must be unique")
    return targets


def load_qrels(qrels_path: Path, query_ids: set[str]) -> dict[str, set[str]]:
    relevant_ids = {query_id: set() for query_id in query_ids}
    with qrels_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            if row["query-id"] in query_ids and int(row["score"]) > 0:
                relevant_ids[row["query-id"]].add(row["corpus-id"])
    missing = {query_id for query_id, ids in relevant_ids.items() if not ids}
    if missing:
        raise ValueError(f"Missing qrels for {len(missing)} target queries")
    return relevant_ids


def choose_document_ids(
    *,
    total_rows: int,
    required_ids: set[str],
    count: int,
    seed: int,
) -> set[str]:
    if count < len(required_ids):
        raise ValueError("Requested count is smaller than required qrels documents")
    selected = set(required_ids)
    generator = random.Random(seed)
    while len(selected) < count:
        selected.add(f"doc{generator.randrange(total_rows)}")
    return selected


def read_selected_documents(corpus_path: Path, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    parquet = pq.ParquetFile(corpus_path)
    for batch in parquet.iter_batches(batch_size=50_000, columns=["_id", "title", "text"]):
        for row in batch.to_pylist():
            document_id = str(row["_id"])
            if document_id not in selected_ids:
                continue
            title = str(row.get("title") or "").strip()
            body = str(row.get("text") or "").strip()
            text = f"{title}\n\n{body}" if title and body else title or body
            if not text:
                raise ValueError(f"Selected document is empty: {document_id}")
            selected[document_id] = {
                "id": f"beir-nq-{document_id}",
                "source": "beir-nq",
                "trust": "trusted",
                "tags": ["beir", "nq"],
                "text": text,
            }
    missing = selected_ids - selected.keys()
    if missing:
        raise ValueError(f"Corpus is missing {len(missing)} selected document IDs")
    return selected


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.corpus, args.qrels, args.targets):
        if not path.is_file():
            raise FileNotFoundError(path)
    targets = load_targets(args.targets)
    qrels = load_qrels(
        args.qrels,
        {item["id"] for item in targets},
    )
    required_ids = set().union(*qrels.values())
    parquet = pq.ParquetFile(args.corpus)
    selected_ids = choose_document_ids(
        total_rows=parquet.metadata.num_rows,
        required_ids=required_ids,
        count=args.count,
        seed=args.seed,
    )
    selected = read_selected_documents(args.corpus, selected_ids)
    required_output_ids = {f"beir-nq-{item}" for item in required_ids}
    records = sorted(
        selected.values(),
        key=lambda item: int(str(item["id"]).removeprefix("beir-nq-doc")),
    )
    for record in records:
        if record["id"] in required_output_ids:
            record["tags"].append("target-qrel")
    write_json(args.output, records)
    normalized_targets = [
        {
            "id": item["id"],
            "query": item["question"],
            "expected_answer": item["correct answer"],
            "attack_target": item["incorrect answer"],
            "relevant_document_ids": [
                f"beir-nq-{document_id}"
                for document_id in sorted(
                    qrels[item["id"]],
                    key=lambda value: int(value.removeprefix("doc")),
                )
            ],
        }
        for item in targets
    ]
    write_json(args.experiments, normalized_targets)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "BEIR Natural Questions",
        "selection": {
            "seed": args.seed,
            "document_count": len(records),
            "source_document_count": parquet.metadata.num_rows,
            "target_query_count": len(targets),
            "required_qrel_document_count": len(required_ids),
            "all_target_qrels_included": required_output_ids.issubset(
                {record["id"] for record in records}
            ),
        },
        "sources": {
            name: {"url": SOURCE_URLS[name], "sha256": sha256(path)}
            for name, path in {
                "corpus": args.corpus,
                "qrels": args.qrels,
                "targets": args.targets,
            }.items()
        },
        "outputs": {
            "corpus": {"path": display_path(args.output), "sha256": sha256(args.output)},
            "target_queries": {"path": display_path(args.experiments), "sha256": sha256(args.experiments)},
        },
    }
    write_json(args.manifest, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))
