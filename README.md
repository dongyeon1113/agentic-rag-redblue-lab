# Agentic RAG Red/Blue Lab

Controlled research environment for testing retrieval poisoning attacks and
defenses across a small multi-agent RAG system.

## Current milestone

The first milestone provides four independently deployable services:

- `orchestrator`: routes a query to selected search agents.
- `local-db-agent`: searches a small BEIR NQ-style sample.
- `gmail-agent`: searches non-sensitive dummy email.
- `drive-agent`: searches non-sensitive dummy Drive documents.

The Gmail and Drive services use local JSON fixtures for now. Real Google API
credentials and the complete BEIR datasets are intentionally not part of this
milestone.

## Architecture

```text
Client
  |
  v
Orchestrator :8000
  |--------- Local DB Agent :8001
  |--------- Gmail Agent    :8002
  `--------- Drive Agent    :8003
```

## Run with Docker

```bash
cp .env.example .env
sudo docker compose up -d --build
sudo docker compose ps
```

Check the complete topology:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/agents
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the capital of France?","sources":["local_db"]}'
```

Stop the services:

```bash
sudo docker compose down
```

## Run tests without Docker

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## Repository layout

```text
services/       Four HTTP services and shared search code
datasets/       Small committed fixtures only
attacks/        Red-team scenarios and reproducible payloads
defenses/       Blue-team adapters and evaluation notes
docs/           Architecture and experiment documentation
tests/          Unit and API tests
```

Never commit `.env`, OAuth credentials, API tokens, private keys, downloaded
BEIR corpora, or model weights.
