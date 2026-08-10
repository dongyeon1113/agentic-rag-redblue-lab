# Architecture

## Service responsibilities

| Service | Current responsibility | Next integration |
| --- | --- | --- |
| Orchestrator | Fan out queries and generate a Qwen RAG answer | Routing policy |
| Local DB Agent | Index and search an NQ-style fixture in Chroma | Complete BEIR corpus |
| Gmail Agent | Index and search dummy email in Chroma | Gmail API with a dedicated dummy account |
| Drive Agent | Index and search dummy documents in Chroma | Google Drive API with a dedicated dummy account |

## Retrieval layer

Search agents convert fixture records to LangChain `Document` objects and
preserve `document_id`, `source`, and `trust` as Chroma metadata. The default
offline embedding keeps tests deterministic. A later Ollama embedding adapter
can replace it without changing the search API.

## Answer generation

`POST /query` returns retrieval results for inspection. `POST /answer` formats
the highest-ranked passages and invokes `qwen3:8b` through LangChain
`ChatOllama`. Vulnerable mode includes trusted and untrusted passages. Defended
mode removes untrusted passages and uses an instruction-isolation prompt.

## Retrieval-stage defense

`retrieval_defense: "ragpart"` switches the search agents to a second Chroma
collection holding `C(N, k)` mean-pooled fragment vectors per document. Each
combination is queried independently and the resulting top-p lists are merged
by majority vote (arXiv:2512.24268). It runs before generation and uses no
trust metadata, so it composes with either `mode`. See
`docs/ragpart-ragmask.ko.md` for the algorithm, measurements, and its current
limitation under the offline hash embedding.

## Trust boundary

All retrieved content is data, not executable instructions. Future connectors
must preserve source and trust metadata so attack and defense evaluation can
distinguish trusted fixtures from injected content.

Secrets belong in `.env` or mounted credential files. They must never be
committed to Git or included in test fixtures.
