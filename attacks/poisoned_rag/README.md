# PoisonedRAG black-box reproduction

This is the only attack workflow exposed by `main`. The previous mixed attack
dashboard is preserved in `codex/all-attacks-backup-2026-08-02`.

The implementation follows the paper's black-box construction:

1. Generate an independent instruction passage `I` for target question `Q`
   and attacker-selected answer `R`.
2. Ask the victim answer model using only `I` as context.
3. Retry until the answer contains `R` or the configured `L` limit is reached.
4. Inject each verified malicious text as `P = Q || I`.
5. Measure answer ASR and retrieval precision, recall, and F1 at Top-K.

The dashboard also runs multiple questions over `N = 0, 1, 3, 5` and exports
the raw results as JSON plus aggregate metrics as CSV.

Run one controlled trial from the command line:

```bash
python3 attacks/poisoned_rag/run.py \
  --query "What is the capital of France?" \
  --expected-answer Paris \
  --attack-target Lyon \
  --poison-count 5 \
  --top-k 5 \
  --max-generation-trials 10
```

Open `http://localhost:8000/demo` for a single-run view or an N-sweep across
the committed, synthetic scenarios. Each run clears earlier untrusted
experiment documents before measuring its baseline.
