# Keyword-stuffing retrieval attack

This directory will contain the first red-team experiment.

## Goal

Inject an untrusted passage whose query terms are repeated often enough to rank
above a trusted passage, then measure whether the final answer changes.

## Planned artifacts

- `poison.json`: controlled, non-sensitive poison passages.
- `inject.py`: inserts poison into the local corpus.
- `evaluate.py`: records retrieval rank, score, and attack success.
- `results/`: generated experiment output, not hand-edited claims.

The baseline service does not yet accept writes. Corpus mutation and the
RAGDefender comparison are the next milestone.
