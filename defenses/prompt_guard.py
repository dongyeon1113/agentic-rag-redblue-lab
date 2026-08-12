"""Meta Prompt Guard 86M adapter for retrieved-document filtering."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Any

DEFAULT_MODEL_ID = "meta-llama/Prompt-Guard-86M"
BLOCKED_LABELS = frozenset({"injection", "jailbreak"})

@dataclass(frozen=True)
class PromptGuardChunkResult:
    chunk_index: int
    label: str
    scores: dict[str, float]
    blocked: bool

@dataclass(frozen=True)
class PromptGuardResult:
    label: str
    scores: dict[str, float]
    blocked: bool
    reason: str
    chunks: list[PromptGuardChunkResult]
    latency_ms: float

class PromptGuardDetector:
    """Classify token chunks; any injection or jailbreak blocks the document.

    The model loads once during construction and is reused. Korean and other
    non-English corpora require separate quality evaluation.
    """
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, *, tokenizer: Any | None = None,
                 model: Any | None = None, torch_module: Any | None = None,
                 device: str | None = None, max_length: int | None = None,
                 stride: int = 32, threshold: float = 0.90) -> None:
        if tokenizer is None or model is None or torch_module is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Prompt Guard requires 'transformers' and 'torch'.") from exc
            torch_module = torch
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load {model_id}. Accept its Hugging Face license and authenticate "
                    f"with `huggingface-cli login` or set HF_TOKEN. Original error: {exc}"
                ) from exc
        self.tokenizer, self.model, self.torch = tokenizer, model, torch_module
        self.device = device or ("cuda" if self.torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        configured_limit = getattr(tokenizer, "model_max_length", 512)
        if not isinstance(configured_limit, int) or configured_limit > 100_000:
            configured_limit = 512
        self.max_length = max_length or configured_limit
        if self.max_length < 2 or stride < 0 or stride >= self.max_length:
            raise ValueError("Invalid max_length or stride")
        self.stride = stride
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.id2label = {int(i): str(label) for i, label in self.model.config.id2label.items()}
        if not self.id2label:
            raise RuntimeError("Prompt Guard model config has no id2label map")

    def inspect(self, value: str | Any) -> PromptGuardResult:
        text = value if isinstance(value, str) else getattr(value, "text", None)
        if not isinstance(text, str):
            raise TypeError("Input must be text or have a text field")
        started = perf_counter()
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length,
            stride=self.stride, return_overflowing_tokens=True, padding=True,
            return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in encoded.items()
                  if k in {"input_ids", "attention_mask", "token_type_ids"}}
        with self.torch.no_grad():
            probabilities = self.torch.softmax(self.model(**inputs).logits, dim=-1).detach().cpu()
        chunks = []
        for index, row in enumerate(probabilities.tolist()):
            scores = {self.id2label.get(i, f"label_{i}"): float(score)
                      for i, score in enumerate(row)}
            label = max(scores, key=scores.get)
            danger_score = max(
                (score for name, score in scores.items() if name.casefold() in BLOCKED_LABELS),
                default=0.0,
            )
            chunks.append(PromptGuardChunkResult(
                index, label, scores, danger_score >= self.threshold
            ))
        if not chunks:
            raise RuntimeError("Prompt Guard tokenizer produced no chunks")
        blocked_chunks = [chunk for chunk in chunks if chunk.blocked]
        decisive = max(blocked_chunks or chunks, key=lambda c: c.scores[c.label])
        blocked = bool(blocked_chunks)
        reason = (f"Prompt Guard classified chunk {decisive.chunk_index} as {decisive.label}."
                  if blocked else "All chunks were classified as benign.")
        return PromptGuardResult(decisive.label, decisive.scores, blocked, reason,
            chunks, (perf_counter() - started) * 1000)
