import argparse
import json
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one controlled multi-document PoisonedRAG experiment."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--query", required=True)
    parser.add_argument("--expected-answer", required=True)
    parser.add_argument("--attack-target", required=True)
    parser.add_argument("--poison-count", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-generation-trials", type=int, default=10)
    parser.add_argument("--passage-word-count", type=int, default=30)
    parser.add_argument("--generation-temperature", type=float, default=1.0)
    args = parser.parse_args()

    request = Request(
        f"{args.base_url.rstrip('/')}/experiments/poisoned-rag",
        data=json.dumps(
            {
                "query": args.query,
                "expected_answer": args.expected_answer,
                "attack_target": args.attack_target,
                "poison_count": args.poison_count,
                "top_k": args.top_k,
                "max_generation_trials": args.max_generation_trials,
                "passage_word_count": args.passage_word_count,
                "generation_temperature": args.generation_temperature,
                "cleanup_before_run": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        print(json.dumps(json.load(response), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
