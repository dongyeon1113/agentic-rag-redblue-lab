# JSON mock-data deployment

모의 JSON 파일을 사용하는 기본 실험 구성입니다. 기존 프로젝트 경로를
런타임에 참조하지 않으며 필요한 파일은 모두 `mock_data/`에 있습니다.

```text
mock_data/
├── local_db/
│   ├── nq_10000.json
│   └── secrets.json
├── gmail/
│   ├── inbox.json
│   └── sent.json
└── drive/
    └── items.json
```

실행:

```bash
docker compose \
  -f compose.agent.yaml \
  -f compose.vector.override.yaml \
  up --build
```

컨테이너는 이미지에 포함된 `mock_data/`를 읽기 전용 seed처럼 사용합니다.
첫 실행 시 각 서비스의 named volume에 쓰기 가능한 작업 JSON을 생성합니다.

| 서비스 | 작업 데이터 |
|---|---|
| Local DB | `local-db-data:/app/data/local_db/documents.json` |
| Gmail | `gmail-data:/app/data/gmail/{inbox,sent}.json` |
| Drive | `drive-data:/app/data/drive/items.json` |
| Orchestrator | `orchestrator-data:/app/data/orchestrator` |

JSON seed를 수정한 뒤 새 상태로 다시 실험하려면 기존 작업 볼륨도 초기화합니다.

```bash
docker compose \
  -f compose.agent.yaml \
  -f compose.vector.override.yaml \
  down -v
docker compose \
  -f compose.agent.yaml \
  -f compose.vector.override.yaml \
  up --build
```

`down -v`는 이 Compose 프로젝트의 작업 볼륨을 삭제하므로 기존 실험 변경 사항도
사라집니다. `mock_data/`의 seed 파일은 삭제되지 않습니다.

