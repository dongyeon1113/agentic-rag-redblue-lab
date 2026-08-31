from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern


@dataclass(frozen=True)
class RegexRule:
    name: str
    pattern: Pattern[str]
    description: str


@dataclass(frozen=True)
class RegexMatch:
    rule_name: str
    description: str
    matched_text: str
    start: int
    end: int


@dataclass(frozen=True)
class FilterResult:
    is_suspicious: bool
    matches: list[RegexMatch]


def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


DEFAULT_PROMPT_INJECTION_RULES = [
    RegexRule(
        name="ignore_instructions",
        pattern=_compile(
            r"\b(ignore|disregard|forget|override)\b.{0,80}"
            r"\b(previous|prior|above|earlier|system|developer)\b.{0,40}"
            r"\b(instruction|instructions|prompt|message|messages|rules)\b"
        ),
        description="Attempts to ignore, override, or discard higher-priority instructions.",
    ),
    RegexRule(
        name="retrieved_passage_override",
        pattern=_compile(
            r"\b(ignore|disregard)\b.{0,80}"
            r"\b(other|all|conflicting)\b.{0,40}"
            r"\b(retrieved|passage|passages|evidence|documents|context)\b"
        ),
        description="Attempts to make one retrieved passage override other retrieved evidence.",
    ),
    RegexRule(
        name="role_reassignment",
        pattern=_compile(
            r"\b(you are now|act as|pretend to be|assume the role of)\b"
        ),
        description="Attempts to assign a new role to the answer generator.",
    ),
    RegexRule(
        name="system_prompt_claim",
        pattern=_compile(
            r"\b(system prompt|developer message|hidden instruction|"
            r"confidential instruction|internal instruction)\b"
        ),
        description="References privileged prompt or instruction channels.",
    ),
    RegexRule(
        name="exact_answer_command",
        pattern=_compile(
            r"\b(answer|respond|reply)\b.{0,80}"
            r"(\b(exactly|only)\b.{0,40}(:|as|the following|this)\b|"
            r"\bwith\b.{0,20}\b(exactly|only)\b.{0,20}:)"
        ),
        description="Commands the model to emit a specific answer format or content.",
    ),
    RegexRule(
        name="do_not_disclose_instruction",
        pattern=_compile(
            r"\b(do not|don't|never)\b.{0,60}"
            r"\b(mention|reveal|disclose|cite|acknowledge)\b.{0,60}"
            r"\b(this instruction|these instructions|this prompt|this message)\b"
        ),
        description="Attempts to hide malicious or controlling instructions.",
    ),
]


class RegexPromptInjectionFilter:
    def __init__(
        self,
        rules: list[RegexRule] | None = None,
        *,
        max_match_text_length: int = 160,
    ) -> None:
        self.rules = rules or DEFAULT_PROMPT_INJECTION_RULES
        self.max_match_text_length = max_match_text_length

    def inspect(self, text: str) -> FilterResult:
        matches: list[RegexMatch] = []
        for rule in self.rules:
            for match in rule.pattern.finditer(text):
                matched_text = match.group(0)
                if len(matched_text) > self.max_match_text_length:
                    matched_text = matched_text[: self.max_match_text_length].rstrip()
                matches.append(
                    RegexMatch(
                        rule_name=rule.name,
                        description=rule.description,
                        matched_text=matched_text,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return FilterResult(is_suspicious=bool(matches), matches=matches)

    def is_suspicious(self, text: str) -> bool:
        return self.inspect(text).is_suspicious


DEFAULT_PROMPT_INJECTION_FILTER = RegexPromptInjectionFilter()


def detect_prompt_injection(text: str) -> FilterResult:
    return DEFAULT_PROMPT_INJECTION_FILTER.inspect(text)


def contains_prompt_injection(text: str) -> bool:
    return DEFAULT_PROMPT_INJECTION_FILTER.is_suspicious(text)
