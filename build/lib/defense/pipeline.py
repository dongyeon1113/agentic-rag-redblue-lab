from __future__ import annotations

import asyncio
import math
from copy import deepcopy
from time import perf_counter
from typing import Any

from agent_system.contracts import AgentTask, Capability, RiskLevel, TaskResult
from defense.models import DefenseConfig, DefenseFinding, DefenseReport
from defense.ragpart import combination_vectors, partition_text
from defense.regex_prompt_injection import RegexPromptInjectionFilter
from defense.spotlighting import SimplifiedSpotlighting

_RECORD_LIST_KEYS = ("documents", "messages", "items")
_RECORD_KEYS = ("document", "message", "item")
_TEXT_KEYS = ("content", "body", "text")
_ID_KEYS = ("document_id", "message_id", "item_id")


class DefensePipeline:
    def __init__(
        self,
        prompt_guard_detector: Any | None = None,
        embedding: Any | None = None,
    ) -> None:
        self._regex = RegexPromptInjectionFilter()
        self._prompt_guard = prompt_guard_detector
        self._embedding = embedding

    async def inspect_result(
        self,
        task: AgentTask,
        result: TaskResult,
        config: DefenseConfig,
        report: DefenseReport,
    ) -> tuple[TaskResult, list[str]]:
        if result.error is not None or not result.output:
            return result, []
        output = deepcopy(result.output)
        instructions: list[str] = []
        records, container_key, singular = self._records(output)
        if not records:
            return result, []

        if config.ragpart and len(records) > 1:
            records = await asyncio.to_thread(
                self._ragpart_rerank,
                records,
                str(task.parameters.get("query", "")),
                report,
            )

        kept: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            report.inspected_records += 1
            record_id = self._record_id(record, index)
            text_key, text = self._record_text(record)
            trust = str(record.get("metadata", {}).get("trust", "unknown")).casefold()
            if trust == "untrusted":
                report.untrusted_data_seen = True

            blocked = False
            if config.regex_filter:
                started = perf_counter()
                finding = self._regex.inspect(text)
                report.detector_latency_ms += (perf_counter() - started) * 1000
                if finding.is_suspicious:
                    blocked = True
                    report.findings.append(DefenseFinding(
                        defense="regex",
                        record_id=record_id,
                        action="blocked",
                        reason=", ".join(match.rule_name for match in finding.matches),
                    ))

            if not blocked and config.prompt_guard:
                if self._prompt_guard is None:
                    raise RuntimeError(
                        "Prompt Guard is enabled but no detector is configured. "
                        "Install requirements-defense.txt and set PROMPT_GUARD_ENABLED=true."
                    )
                pg = await asyncio.to_thread(self._prompt_guard.inspect, text)
                report.detector_latency_ms += pg.latency_ms
                if pg.blocked:
                    blocked = True
                    report.findings.append(DefenseFinding(
                        defense="prompt_guard",
                        record_id=record_id,
                        action="blocked",
                        reason=pg.reason,
                        metadata={"label": pg.label},
                    ))

            if blocked:
                report.blocked_records += 1
                continue

            transformed = text
            for method in config.spotlighting:
                spotlighted = SimplifiedSpotlighting(method).apply(transformed)
                transformed = spotlighted.transformed_document
                instructions.append(spotlighted.system_instruction)
                report.transformed_records += 1
                report.findings.append(DefenseFinding(
                    defense=f"spotlighting:{method}",
                    record_id=record_id,
                    action="transformed",
                    reason="Untrusted retrieval content was spotlighted.",
                    metadata=spotlighted.metadata,
                ))
            if text_key is not None:
                record[text_key] = transformed
            kept.append(record)

        if container_key is not None:
            if singular:
                output[container_key] = kept[0] if kept else None
            else:
                output[container_key] = kept
        return result.model_copy(update={"output": output}), list(dict.fromkeys(instructions))

    def _ragpart_rerank(
        self,
        records: list[dict[str, Any]],
        query: str,
        report: DefenseReport,
    ) -> list[dict[str, Any]]:
        if self._embedding is None:
            raise RuntimeError("RAGPart requires the configured embedding client.")
        texts = [self._record_text(record)[1] for record in records]
        fragments = [partition_text(text, 5) for text in texts]
        flat = [piece for group in fragments for piece in group]
        vectors = self._embedding.embed([query, *flat])
        query_vector, fragment_vectors = vectors[0], vectors[1:]
        scored: list[tuple[float, int, dict[str, Any]]] = []
        offset = 0
        for index, (record, pieces) in enumerate(zip(records, fragments, strict=True)):
            current = fragment_vectors[offset : offset + len(pieces)]
            offset += len(pieces)
            combinations = combination_vectors(current, 3)
            similarities = [self._cosine(query_vector, vector) for vector in combinations]
            score = sum(similarities) / len(similarities) if similarities else 0.0
            scored.append((score, index, record))
            report.findings.append(DefenseFinding(
                defense="ragpart",
                record_id=self._record_id(record, index),
                action="reranked",
                reason="Ranked by five-fragment, size-three combination consensus.",
                metadata={"score": round(score, 6), "combinations": len(combinations)},
            ))
        return [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    @staticmethod
    def blocks_indirect_action(
        capability: Capability,
        config: DefenseConfig,
        report: DefenseReport,
    ) -> bool:
        return (
            config.block_indirect_actions
            and report.untrusted_data_seen
            and capability.risk is not RiskLevel.READ
        )

    @staticmethod
    def _records(output: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None, bool]:
        for key in _RECORD_LIST_KEYS:
            value = output.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)], key, False
        for key in _RECORD_KEYS:
            value = output.get(key)
            if isinstance(value, dict):
                return [value], key, True
        return [], None, False

    @staticmethod
    def _record_text(record: dict[str, Any]) -> tuple[str | None, str]:
        for key in _TEXT_KEYS:
            value = record.get(key)
            if isinstance(value, str):
                return key, value
        return None, ""

    @staticmethod
    def _record_id(record: dict[str, Any], index: int) -> str:
        for key in _ID_KEYS:
            if key in record:
                return str(record[key])
        return f"record-{index}"
