from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent_system.config import AgentSettings
from agentdojo_benchmark.adapter import CurrentAgentPipeline
from defense import DefenseConfig

try:
    from agentdojo.attacks.attack_registry import ATTACKS, load_attack
    from agentdojo.benchmark import (
        SuiteResults,
        benchmark_suite_with_injections,
        benchmark_suite_without_injections,
    )
    from agentdojo.logging import OutputLogger
    from agentdojo.task_suite.load_suites import get_suite, get_suites
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "AgentDojo is required. Install it with: pip install -e '.[benchmark]'"
    ) from exc


@dataclass(frozen=True)
class BenchmarkOptions:
    suites: tuple[str, ...] = ("workspace",)
    attack: str = "important_instructions"
    benchmark_version: str = "v1.2.2"
    user_tasks: tuple[str, ...] = ()
    user_tasks_by_suite: dict[str, tuple[str, ...]] | None = None
    injection_tasks: tuple[str, ...] = ()
    output_dir: Path = Path("runs/agentdojo")
    force_rerun: bool = False
    include_benign: bool = True
    include_attack: bool = True
    defense_profiles: tuple[str, ...] = ("baseline",)


DEFENSE_COMPONENTS = (
    "regex",
    "prompt_guard",
    "task_shield",
    "spotlighting:delimiting",
    "spotlighting:datamarking",
    "spotlighting:encoding",
)

DEFENSE_PROFILES = {
    "baseline": DefenseConfig(block_indirect_actions=False),
    "regex": DefenseConfig(regex_filter=True, block_indirect_actions=False),
    "spotlighting:delimiting": DefenseConfig(
        spotlighting=["delimiting"], block_indirect_actions=False
    ),
    "spotlighting:datamarking": DefenseConfig(
        spotlighting=["datamarking"], block_indirect_actions=False
    ),
    "spotlighting:encoding": DefenseConfig(
        spotlighting=["encoding"], block_indirect_actions=False
    ),
    "prompt_guard": DefenseConfig(prompt_guard=True, block_indirect_actions=False),
    "task_shield": DefenseConfig(task_shield=True, block_indirect_actions=False),
    "all": DefenseConfig(
        regex_filter=True,
        prompt_guard=True,
        task_shield=True,
        spotlighting=["delimiting"],
        block_indirect_actions=False,
    ),
}


def defense_config(profile: str) -> DefenseConfig:
    if profile in DEFENSE_PROFILES:
        return DEFENSE_PROFILES[profile].model_copy(deep=True)
    components = profile.split("+")
    if not components or any(item not in DEFENSE_COMPONENTS for item in components):
        raise ValueError(f"Unknown defense combination {profile!r}")
    if len(set(components)) != len(components):
        raise ValueError(f"Duplicate defense in combination {profile!r}")
    spotlighting = [
        item.split(":", 1)[1]
        for item in components
        if item.startswith("spotlighting:")
    ]
    if len(spotlighting) > 1:
        raise ValueError("A combination can contain only one spotlighting method")
    return DefenseConfig(
        regex_filter="regex" in components,
        prompt_guard="prompt_guard" in components,
        task_shield="task_shield" in components,
        spotlighting=spotlighting,
        block_indirect_actions=False,
    )


def _rate(values: dict[Any, bool]) -> float | None:
    return sum(values.values()) / len(values) if values else None


def _serialize_results(results: SuiteResults) -> dict[str, Any]:
    return {
        "utility_results": {
            "::".join(key): value
            for key, value in results["utility_results"].items()
        },
        "attack_success_results": {
            "::".join(key): value
            for key, value in results["security_results"].items()
        },
        "injection_task_capability": results["injection_tasks_utility_results"],
    }


def _load_model_aware_attack(
    name: str,
    suite: Any,
    pipeline: CurrentAgentPipeline,
) -> Any:
    if name not in ATTACKS:
        raise ValueError(
            f"Unknown attack {name!r}. Available attacks: {', '.join(sorted(ATTACKS))}"
        )
    actual_name = pipeline.name
    try:
        # Some official attacks require a model name known to AgentDojo while
        # constructing their prompt. Keep the official attack implementation,
        # then replace only its prose label with the real local model name.
        pipeline.name = f"gpt-4o-2024-05-13-{actual_name}"
        attack = load_attack(name, suite, pipeline)
    finally:
        pipeline.name = actual_name
    if hasattr(attack, "model_name"):
        attack.model_name = pipeline.settings.ollama_model
    return attack


def run_benchmark(
    options: BenchmarkOptions,
    *,
    settings: AgentSettings | None = None,
) -> dict[str, Any]:
    # Load project-local configuration without overriding variables supplied by
    # a service manager, container runtime, or the current process.
    load_dotenv(override=False)
    if not options.include_benign and not options.include_attack:
        raise ValueError("At least one of benign or attack evaluation must be enabled")
    available_suites = get_suites(options.benchmark_version)
    unknown = set(options.suites) - set(available_suites)
    if unknown:
        raise ValueError(f"Unknown suites: {', '.join(sorted(unknown))}")
    if options.user_tasks and options.user_tasks_by_suite is not None:
        raise ValueError("Use either user_tasks or user_tasks_by_suite, not both")
    if options.user_tasks and len(options.suites) != 1:
        raise ValueError("--user-task can only be used with one suite")
    if options.user_tasks_by_suite is not None:
        unknown_selection_suites = (
            set(options.user_tasks_by_suite) - set(options.suites)
        )
        if unknown_selection_suites:
            raise ValueError(
                "User-task selection contains suites outside this run: "
                + ", ".join(sorted(unknown_selection_suites))
            )
    if options.injection_tasks and len(options.suites) != 1:
        raise ValueError("--injection-task can only be used with one suite")

    settings = settings or AgentSettings.from_env()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "benchmark_version": options.benchmark_version,
        "agentdojo_attack": options.attack,
        "model": settings.ollama_model,
        "all_tool_permissions": True,
        "approvals_required": False,
        "defense_profiles": list(options.defense_profiles),
        "user_task_selection": (
            {
                suite: list(task_ids)
                for suite, task_ids in options.user_tasks_by_suite.items()
            }
            if options.user_tasks_by_suite is not None
            else None
        ),
        "profiles": {},
    }

    for profile in options.defense_profiles:
        config = defense_config(profile)
        pipeline = CurrentAgentPipeline(
            settings=settings, defense=config, profile_name=profile
        )
        profile_dir = options.output_dir / profile.replace(":", "-")
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_report: dict[str, Any] = {
            "config": config.model_dump(mode="json"),
            "suites": {},
        }
        with OutputLogger(str(profile_dir)):
            for suite_name in options.suites:
                selected_user_tasks: tuple[str, ...] | None = (
                    options.user_tasks or None
                )
                if options.user_tasks_by_suite is not None:
                    selected_user_tasks = options.user_tasks_by_suite.get(
                        suite_name,
                        (),
                    )
                    if not selected_user_tasks:
                        continue
                suite = get_suite(options.benchmark_version, suite_name)
                suite_report: dict[str, Any] = {}
                if options.include_benign:
                    benign = benchmark_suite_without_injections(
                        pipeline,
                        suite,
                        logdir=profile_dir,
                        force_rerun=options.force_rerun,
                        user_tasks=selected_user_tasks,
                        benchmark_version=options.benchmark_version,
                    )
                    suite_report["benign_utility"] = _rate(benign["utility_results"])
                    suite_report["benign"] = _serialize_results(benign)
                if options.include_attack:
                    attack = _load_model_aware_attack(options.attack, suite, pipeline)
                    attacked = benchmark_suite_with_injections(
                        pipeline,
                        suite,
                        attack,
                        logdir=profile_dir,
                        force_rerun=options.force_rerun,
                        user_tasks=selected_user_tasks,
                        injection_tasks=options.injection_tasks or None,
                        verbose=False,
                        benchmark_version=options.benchmark_version,
                    )
                    suite_report["utility_under_attack"] = _rate(
                        attacked["utility_results"]
                    )
                    suite_report["targeted_asr"] = _rate(
                        attacked["security_results"]
                    )
                    suite_report["attacked"] = _serialize_results(attacked)
                profile_report["suites"][suite_name] = suite_report
        report["profiles"][profile] = profile_report

    summary_path = options.output_dir / "current-agent-summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["summary_path"] = str(summary_path)
    return report
