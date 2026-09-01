from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentdojo_benchmark.runner import DEFENSE_PROFILES, BenchmarkOptions, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the repository's ToolCallingAgent in official AgentDojo "
            "environments with every suite tool permitted and optional defenses."
        )
    )
    parser.add_argument(
        "--suite",
        action="append",
        dest="suites",
        choices=("workspace", "slack", "travel", "banking", "all"),
        help="Suite to run; repeat for multiple suites. Default: workspace.",
    )
    parser.add_argument("--attack", default="important_instructions")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--user-task", action="append", default=[])
    parser.add_argument("--injection-task", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/agentdojo"))
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--attack-only", action="store_true")
    parser.add_argument("--benign-only", action="store_true")
    parser.add_argument(
        "--defense",
        action="append",
        choices=tuple(DEFENSE_PROFILES),
        help="Defense profile; repeat to compare profiles. Default: baseline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.attack_only and args.benign_only:
        raise SystemExit("--attack-only and --benign-only cannot be combined")
    selected = args.suites or ["workspace"]
    suites = (
        ("workspace", "slack", "travel", "banking")
        if "all" in selected
        else tuple(dict.fromkeys(selected))
    )
    report = run_benchmark(BenchmarkOptions(
        suites=suites,
        attack=args.attack,
        benchmark_version=args.benchmark_version,
        user_tasks=tuple(args.user_task),
        injection_tasks=tuple(args.injection_task),
        output_dir=args.output_dir,
        force_rerun=args.force_rerun,
        include_benign=not args.attack_only,
        include_attack=not args.benign_only,
        defense_profiles=tuple(dict.fromkeys(args.defense or ["baseline"])),
    ))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
