# Vendored BIPIA document-generation subset

Source: https://github.com/microsoft/BIPIA

Pinned commit: `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`

This directory contains the official Email QA and Table QA test contexts plus
the text attack set. The orchestrator reads these files directly to reproduce
the original retrieval-free BIPIA prompt flow. The unused Code task and Python
builder package are intentionally excluded.

The upstream license and NOTICE are included. Benchmark datasets have their own
licenses summarized in `LICENSE`; preserve attribution and verify source terms.
Web QA and Summarization are excluded because upstream requires separately
obtained source datasets.
