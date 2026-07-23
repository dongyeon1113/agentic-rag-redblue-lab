#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable. Check that Docker is running and your user belongs to the docker group." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example."
fi

docker compose up -d --build
docker compose ps

echo
echo "Run: python3 scripts/smoke_test.py"
