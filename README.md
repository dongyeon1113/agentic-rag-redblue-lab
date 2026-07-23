# Agentic RAG Red/Blue Lab

Controlled research environment for retrieval-poisoning and defense experiments
in a small multi-agent RAG system.

## Current system

```text
Client
  |
  v
Orchestrator :8000 ---- Ollama / Qwen3:8b
  |--------- Local DB Agent :8001 ---- ChromaDB
  |--------- Gmail Agent    :8002 ---- ChromaDB
  `--------- Drive Agent    :8003 ---- ChromaDB
```

- The orchestrator fans out a query and generates a cited RAG answer.
- The Local DB agent indexes a small BEIR NQ-style fixture.
- Gmail and Drive agents index non-sensitive dummy fixtures.
- Each search agent has an independent persistent Chroma volume.
- `vulnerable` mode uses trusted and untrusted passages.
- `defended` mode removes untrusted passages before answer generation.

Real Google credentials, private data, and complete research datasets are not
included.

## Requirements

- Linux host with Docker Engine and Docker Compose
- NVIDIA GPU, driver, and NVIDIA Container Toolkit
- User access to the Docker daemon

The default deployment binds all host ports to `127.0.0.1`. Use an SSH tunnel
for remote access; do not expose this unauthenticated lab directly to the
internet.

## First run

```bash
./scripts/bootstrap.sh
```

This creates `.env`, builds the containers, starts the stack, and automatically
pulls the configured Ollama model. The first run downloads Qwen3:8b and can take
several minutes.

Check readiness:

```bash
docker compose ps
python3 scripts/smoke_test.py
```

The five long-running services should report `healthy`. The
`ollama-model-init` service is expected to complete and exit successfully after
checking or downloading the model.

Open the orchestrator API documentation at `http://localhost:8000/docs`.

## API examples

Inspect retrieval:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the capital of France?","sources":["local_db"]}'
```

Generate a RAG answer:

```bash
curl -X POST http://localhost:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the capital of France?","sources":["local_db"],"mode":"vulnerable"}'
```

Use `GET /agents` for search-agent health and `GET /model` for Ollama model
readiness.

## Operations

```bash
docker compose ps
docker compose logs -f orchestrator
docker compose restart
docker compose down
```

Do not run `docker compose down -v` unless persistent Chroma data and downloaded
Ollama models should be deleted.

## Tests

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Pull requests and pushes to `main` or `codex/**` run the same tests in GitHub
Actions.

## Repository layout

```text
services/       HTTP services and shared retrieval code
datasets/       Small committed fixtures only
attacks/        Controlled red-team experiments
defenses/       Blue-team adapters and evaluation notes
docs/           Architecture and team runbooks
scripts/        Bootstrap and deployment validation
tests/          Unit and API tests
```

The Korean team runbook is at
[`docs/team-quickstart.ko.md`](docs/team-quickstart.ko.md).

Never commit `.env`, OAuth credentials, API tokens, private keys, downloaded
corpora, model weights, real email, or real Drive documents.
