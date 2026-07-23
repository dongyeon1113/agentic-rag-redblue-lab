# Architecture

## Service responsibilities

| Service | Current responsibility | Next integration |
| --- | --- | --- |
| Orchestrator | Fan out queries and combine responses | Routing policy and answer generation |
| Local DB Agent | Search a small NQ-style JSON corpus | BEIR corpus and vector index |
| Gmail Agent | Search dummy email fixtures | Gmail API with a dedicated dummy account |
| Drive Agent | Search dummy document fixtures | Google Drive API with a dedicated dummy account |

## Trust boundary

All retrieved content is data, not executable instructions. Future connectors
must preserve source and trust metadata so attack and defense evaluation can
distinguish trusted fixtures from injected content.

Secrets belong in `.env` or mounted credential files. They must never be
committed to Git or included in test fixtures.
