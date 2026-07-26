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
    parser.add_argument(
        "--ratio",
        type=int,
        choices=[1, 2, 4, 6],
        default=2,
    )
    args = parser.parse_args()

    request = Request(
        f"{args.base_url.rstrip('/')}/experiments/poisoned-rag",
        data=json.dumps(
            {
                "query": args.query,
                "expected_answer": args.expected_answer,
                "attack_target": args.attack_target,
                "poison_ratio": args.ratio,
                "limit": 5,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        print(json.dumps(json.load(response), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
