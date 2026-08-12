# Defense integrations

## Meta Prompt Guard 86M

Prompt Guard 86M is Meta Purple Llama's prompt-injection and jailbreak
classifier. It does not replace the Ollama generation model. It runs after
retrieval and before retrieved passages enter the LLM prompt:

```text
query -> retriever -> Prompt Guard -> safe passages -> Ollama LLM
```

Enable it per request with `"prompt_guard": true`. The detector reads label
names from `model.config.id2label`; `benign` passes, while `injection` and
`jailbreak` block the complete document. Long documents are tokenized into
overlapping chunks and one unsafe chunk blocks the whole document. API logs
include per-chunk scores and `detector_latency_ms`; these internal scores
should not be shown in user-facing answers.

Install dependencies with `pip install -r requirements.txt`. The model may be
gated: accept its license, then run `huggingface-cli login` or set `HF_TOKEN`.
`PROMPT_GUARD_MODEL` can override the model ID and `PROMPT_GUARD_DEVICE` can
force `cpu` or `cuda`. CPU inference is supported. Korean-language quality
must be measured separately rather than assumed from English benchmarks.

Example:

```bash
curl -X POST 'http://localhost:8000/answer' \
  -H 'content-type: application/json' \
  -d '{"query":"What is the capital of France?","sources":["local_db"],"mode":"vulnerable","prompt_guard":true}'
```

Run the unit tests without downloading the model:

```bash
python3 -m pytest -q tests/test_prompt_guard.py tests/test_rag.py
```

Official references:

- https://huggingface.co/meta-llama/Prompt-Guard-86M
- https://github.com/meta-llama/PurpleLlama/blob/main/Prompt-Guard/MODEL_CARD.md

## Existing regex filter

The regex adapter uses the same post-retrieval, pre-generation integration
point and can be enabled independently with `regex_filter`.
