#!/usr/bin/env python3
"""Evaluate retrieval quality against scenario relevance judgments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("datasets/experiments/nq_target_queries.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def query(base_url: str, text: str, top_k: int, timeout: float) -> list[dict]:
    payload = json.dumps(
        {"query": text, "limit": top_k, "sources": ["local_db"]}
    ).encode()
    request = Request(
        f"{base_url.rstrip('/')}/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    result = body["results"]["local_db"]
    if result.get("status") != "ok":
        raise RuntimeError(result.get("error", "local_db query failed"))
    return result["hits"]


def main() -> None:
    args = parse_args()
    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
    rows = []
    reciprocal_rank_sum = 0.0
    hit_count = 0

    for scenario in scenarios:
        relevant_ids = set(scenario.get("relevant_document_ids", []))
        try:
            hits = query(args.base_url, scenario["query"], args.top_k, args.timeout)
            retrieved_ids = [hit["document_id"] for hit in hits]
            rank = next(
                (index for index, doc_id in enumerate(retrieved_ids, 1) if doc_id in relevant_ids),
                None,
            )
            error = None
        except Exception as exc:  # Keep the full evaluation running on query failures.
            retrieved_ids = []
            rank = None
            error = f"{type(exc).__name__}: {exc}"
        if rank is not None:
            hit_count += 1
            reciprocal_rank_sum += 1 / rank
        rows.append(
            {
                "scenario_id": scenario["id"],
                "query": scenario["query"],
                "relevant_document_ids": sorted(relevant_ids),
                "retrieved_document_ids": retrieved_ids,
                "first_relevant_rank": rank,
                "hit": rank is not None,
                "error": error,
            }
        )

    total = len(rows)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "top_k": args.top_k,
        "scenario_count": total,
        "successful_queries": sum(row["error"] is None for row in rows),
        "failed_queries": sum(row["error"] is not None for row in rows),
        "hit_count": hit_count,
        "hit_rate": hit_count / total if total else 0.0,
        "mrr": reciprocal_rank_sum / total if total else 0.0,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
