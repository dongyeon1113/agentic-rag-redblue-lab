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

compose_files=(-f compose.yaml)
if grep -Eq '^DRIVE_SYNC_ENABLED=(1|true|yes)$' .env; then
  if [[ -z "${GOOGLE_SERVICE_ACCOUNT_FILE:-}" ]]; then
    google_key="$(
      sed -n 's/^GOOGLE_SERVICE_ACCOUNT_FILE=//p' .env | tail -n 1
    )"
  else
    google_key="$GOOGLE_SERVICE_ACCOUNT_FILE"
  fi
  if [[ -z "$google_key" || ! -f "$google_key" ]]; then
    echo "Google Drive sync is enabled, but GOOGLE_SERVICE_ACCOUNT_FILE is missing." >&2
    exit 1
  fi
  compose_files+=(-f compose.google.yaml)
  echo "Google Drive folder sync enabled."
fi
if grep -Eq '^GMAIL_SYNC_ENABLED=(1|true|yes)$' .env; then
  gmail_token="$(
    sed -n 's/^GOOGLE_GMAIL_TOKEN_FILE=//p' .env | tail -n 1
  )"
  if [[ -z "$gmail_token" || ! -f "$gmail_token" ]]; then
    echo "Gmail sync is enabled, but GOOGLE_GMAIL_TOKEN_FILE is missing." >&2
    echo "Run: python3 scripts/authorize_gmail.py" >&2
    exit 1
  fi
  compose_files+=(-f compose.gmail.yaml)
  echo "Google Gmail sync enabled."
fi
docker_root_dir="$(docker info --format '{{.DockerRootDir}}')"
if [[ "$docker_root_dir" == /var/snap/docker/* ]]; then
  compose_files+=(-f compose.snap.yaml)
  echo "Detected Snap Docker; applying the AppArmor compatibility override."
fi

docker compose "${compose_files[@]}" up -d --build
docker compose "${compose_files[@]}" ps

echo
echo "Run: python3 scripts/smoke_test.py"
