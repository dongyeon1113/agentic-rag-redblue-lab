import argparse
import json
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one controlled keyword-stuffing experiment."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--query", required=True)
    parser.add_argument("--expected-answer", required=True)
    parser.add_argument("--attack-target", required=True)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument(
        "--keyword-only",
        action="store_true",
        help="Disable the indirect prompt-injection instruction.",
    )
    args = parser.parse_args()

    payload = {
        "query": args.query,
        "expected_answer": args.expected_answer,
        "attack_target": args.attack_target,
        "repetitions": args.repetitions,
        "include_prompt_injection": not args.keyword_only,
        "limit": 3,
    }
    request = Request(
        f"{args.base_url.rstrip('/')}/experiments/keyword-stuffing",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        print(json.dumps(json.load(response), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
