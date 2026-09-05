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
_TEXT_KEYS = frozenset({
    "body",
    "content",
    "description",
    "message",
    "messages",
    "review",
    "reviews",
    "snippet",
    "subject",
    "text",
    "title",
    "web_content",
})
_ID_KEYS = ("document_id", "message_id", "item_id", "id_", "id")


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

        # Reuse one marker/delimiter per tool result. This keeps the boundary
        # instruction stable and prevents one system message per record.
        spotlighters = {
            method: SimplifiedSpotlighting(method)
            for method in dict.fromkeys(config.spotlighting)
        }
        kept: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            report.inspected_records += 1
            record_id = self._record_id(record, index)
            _, text = self._record_text(record)
            trust = str(record.get("metadata", {}).get("trust", "unknown")).casefold()
            explicitly_trusted = trust == "trusted"
            if trust == "untrusted":
                report.untrusted_data_seen = True

            blocked = False
            if not explicitly_trusted and text and config.regex_filter:
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

            if (
                not explicitly_trusted
                and text
                and not blocked
                and config.prompt_guard
            ):
                if self._prompt_guard is None:
                    raise RuntimeError(
                        "Prompt Guard is enabled but no detector is configured. "
                        "Build with PROMPT_GUARD_ENABLED=true."
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

            if not explicitly_trusted:
                for method, spotlighter in spotlighters.items():
                    record, spotlighted = self._spotlight_record(record, spotlighter)
                    if not spotlighted:
                        continue
                    instructions.append(spotlighted[0].system_instruction)
                    report.transformed_records += 1
                    metadata = dict(spotlighted[0].metadata)
                    metadata["transformed_fields"] = len(spotlighted)
                    report.findings.append(DefenseFinding(
                        defense=f"spotlighting:{method}",
                        record_id=record_id,
                        action="transformed",
                        reason="Untrusted retrieval content was spotlighted.",
                        metadata=metadata,
                    ))
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

    @classmethod
    def _record_text(cls, record: dict[str, Any]) -> tuple[str | None, str]:
        return None, "\n".join(cls._all_strings(record))

    @classmethod
    def _all_strings(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [
                text
                for item in value
                for text in cls._all_strings(item)
            ]
        if isinstance(value, dict):
            return [
                text
                for key, item in value.items()
                if str(key).casefold() != "metadata"
                for text in cls._all_strings(item)
            ]
        return []

    @classmethod
    def _spotlight_record(
        cls,
        record: dict[str, Any],
        spotlighter: SimplifiedSpotlighting,
    ) -> tuple[dict[str, Any], list[Any]]:
        transformed, results = cls._spotlight_value(
            record,
            spotlighter,
            content_field=False,
        )
        return transformed, results

    @classmethod
    def _spotlight_value(
        cls,
        value: Any,
        spotlighter: SimplifiedSpotlighting,
        *,
        content_field: bool,
    ) -> tuple[Any, list[Any]]:
        if isinstance(value, str):
            if not content_field:
                return value, []
            result = spotlighter.apply(value)
            return result.transformed_document, [result]
        if isinstance(value, list):
            transformed: list[Any] = []
            results: list[Any] = []
            for item in value:
                current, current_results = cls._spotlight_value(
                    item,
                    spotlighter,
                    content_field=content_field,
                )
                transformed.append(current)
                results.extend(current_results)
            return transformed, results
        if isinstance(value, dict):
            transformed_dict: dict[str, Any] = {}
            results = []
            for key, item in value.items():
                key_name = str(key).casefold()
                if key_name == "metadata":
                    transformed_dict[key] = item
                    continue
                current, current_results = cls._spotlight_value(
                    item,
                    spotlighter,
                    content_field=content_field or key_name in _TEXT_KEYS,
                )
                transformed_dict[key] = current
                results.extend(current_results)
            return transformed_dict, results
        return value, []

    @staticmethod
    def _record_id(record: dict[str, Any], index: int) -> str:
        for key in _ID_KEYS:
            if key in record:
                return str(record[key])
        return f"record-{index}"
