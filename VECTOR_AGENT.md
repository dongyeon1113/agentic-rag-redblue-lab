# Complete nomic-embed agent

This is the canonical completed configuration. It combines the agent stack
with the Chroma and `nomic-embed-text` search backend.

```bash
cp .env.example .env
docker compose \
  -f compose.agent.yaml \
  -f compose.vector.override.yaml \
  up --build
```

The first run performs the following initialization:

1. Pull `qwen3:8b` and `nomic-embed-text` into the Ollama model volume.
2. Convert the JSON seeds into mutable service working copies.
3. Encode all Local DB, Gmail, and Drive text with `nomic-embed-text`.
4. Store embeddings and metadata in each service's private Chroma directory.
5. Serve queries by encoding the query with the same model and performing
   cosine-distance vector search.

Persistent data:

```text
local-db-data:/app/data/local_db/
├── documents.json
└── chroma/

gmail-data:/app/data/gmail/
├── inbox.json
├── sent.json
└── chroma/

drive-data:/app/data/drive/
├── items.json
└── chroma/
```

New, updated, sent, moved, and deleted objects are synchronized with both JSON
and their Chroma collection. The original files in `mock_data/` remain seeds.

The Local DB contains 10,000 NQ documents, so first-run embedding can take a
while. Existing IDs are skipped on later starts. If the embedding model or seed
content changes, use a new collection name or reset the corresponding volume.

Natural-language query:

```bash
curl -X POST http://localhost:19000/v1/agent/query \
  -H 'content-type: application/json' \
  -d '{
    "user_id": "user-1",
    "session_id": "session-1",
    "query": "로컬 문서에서 Chicago Fire 시즌 4의 첫 방영일을 찾아줘"
  }'
```

