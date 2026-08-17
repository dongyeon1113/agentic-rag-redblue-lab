import csv
import json
from argparse import Namespace

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_nq_corpus import build


def test_build_includes_qrels_and_is_reproducible(tmp_path) -> None:
    corpus = tmp_path / "corpus.parquet"
    qrels = tmp_path / "qrels.tsv"
    targets = tmp_path / "targets.json"
    output = tmp_path / "output.json"
    manifest = tmp_path / "manifest.json"
    experiments = tmp_path / "experiments.json"
    pq.write_table(
        pa.table({
            "_id": [f"doc{i}" for i in range(20)],
            "title": [f"Title {i}" for i in range(20)],
            "text": [f"Text {i}" for i in range(20)],
        }),
        corpus,
    )
    with qrels.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerow(["q1", "doc19", 1])
    targets.write_text(json.dumps([{
        "id": "q1",
        "question": "Question?",
        "correct answer": "Correct",
        "incorrect answer": "Wrong",
    }]), encoding="utf-8")
    args = Namespace(
        corpus=corpus,
        qrels=qrels,
        targets=targets,
        output=output,
        manifest=manifest,
        experiments=experiments,
        count=10,
        seed=12,
    )

    first = build(args)
    first_output = output.read_bytes()
    second = build(args)

    records = json.loads(first_output)
    assert len(records) == 10
    assert any(record["id"] == "beir-nq-doc19" for record in records)
    assert next(record for record in records if record["id"] == "beir-nq-doc19")["tags"][-1] == "target-qrel"
    assert first_output == output.read_bytes()
    assert first["outputs"]["corpus"]["sha256"] == second["outputs"]["corpus"]["sha256"]
    assert json.loads(experiments.read_text())[0]["relevant_document_ids"] == [
        "beir-nq-doc19"
    ]
