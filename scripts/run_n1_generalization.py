"""Run a type-balanced N=1 evaluation over expanded scenarios."""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            values = [json.loads(line) for line in source if line.strip()]
        return [
            item
            for item in values
            if item.get("status", "accepted") == "accepted"
            and item.get("baseline_valid", True) is True
        ]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("scenario file must contain a JSON list or JSONL records")
    return value


def balanced_sample(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    generator = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("question_type", "unspecified"))].append(record)
    for group in groups.values():
        generator.shuffle(group)
    selected: list[dict[str, Any]] = []
    categories = sorted(groups)
    while len(selected) < min(count, len(records)):
        progressed = False
        for category in categories:
            if groups[category] and len(selected) < count:
                selected.append(groups[category].pop())
                progressed = True
        if not progressed:
            break
    return selected


def post_json(url: str, value: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = balanced_sample(load_scenarios(args.scenarios), args.count, args.seed)
    scenario_types = {item["id"]: item.get("question_type", "unspecified") for item in scenarios}
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for start in range(0, len(scenarios), 20):
        chunk = scenarios[start : start + 20]
        payload = {
            "scenarios": [
                {
                    "name": item["id"],
                    "query": item["query"],
                    "expected_answer": item["expected_answer"],
                    "attack_target": item["attack_target"],
                }
                for item in chunk
            ],
            "poison_counts": [1],
            "repetitions": args.repetitions,
            "top_k": args.top_k,
            "max_generation_trials": args.max_generation_trials,
            "passage_word_count": args.passage_word_count,
            "generation_temperature": args.temperature,
            "candidate_multiplier": args.candidate_multiplier,
            "fixed_poison_pool": True,
            "retrieval_defenses": ["none"],
        }
        result = post_json(
            f"{args.base_url.rstrip('/')}/experiments/poisoned-rag/benchmark",
            payload,
            timeout=args.timeout,
        )
        runs.extend(result.get("runs", []))
        failures.extend(result.get("failures", []))
        print(f"completed{min(start + 20, len(scenarios))}/{len(scenarios)}", flush=True)

    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for item in runs:
        category = scenario_types.get(str(item.get("scenario_name")), "unspecified")
        by_type[category][str(item["attacked"]["outcome"])] += 1
        by_type[category]["retrieved"] += int(item["metrics"]["poison_in_top_k"] > 0)
        by_type[category]["runs"] += 1
    summary = {
        "configuration": {
            "n": 1,
            "top_k": args.top_k,
            "scenario_count": len(scenarios),
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "completed_runs": len(runs),
        "failed_runs": len(failures),
        "outcomes": dict(Counter(item["attacked"]["outcome"] for item in runs)),
        "retrieval_rate": (
            sum(item["metrics"]["poison_in_top_k"] > 0 for item in runs) / len(runs)
            if runs else 0.0
        ),
        "adoption_rate": (
            sum(item["attacked"]["outcome"] == "attack_succeeded" for item in runs) / len(runs)
            if runs else 0.0
        ),
        "by_question_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "scenario_ids": [item["id"] for item in scenarios],
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-generation-trials", type=int, default=10)
    parser.add_argument("--passage-word-count", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
