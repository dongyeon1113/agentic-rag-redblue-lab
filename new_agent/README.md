# New Agent Architecture

기존 `services/` 구현을 유지한 채 새로 개발하는 독립 프로젝트 스켈레톤입니다.
현재 구조는 하나의 LLM 오케스트레이터를 전제로 하며, Local DB, Gmail,
Google Drive는 LLM 없는 FastAPI 도구 서비스로 동작합니다.

## Architecture

```text
Client
  |
  v
Orchestrator :8000
  |-- HTTP --> Local DB tool service :8001
  |-- HTTP --> Gmail tool service    :8002
  `-- HTTP --> Drive tool service    :8003
```

- `application/`: 오케스트레이션과 워크플로 유스케이스
- `contracts/`: 서비스 사이의 안정적인 HTTP DTO
- `ports/`: LLM, 메모리, 실행기 인터페이스
- `infrastructure/`: HTTP 클라이언트 및 메모리 어댑터
- `tool_runtime/`: 세 도구 서비스가 공유하는 실행·권한·승인 런타임
- `services/`: Local DB, Gmail, Drive 도구와 FastAPI 앱
- `security/`: 향후 TaskShield를 연결할 `AgentGuard` 경계

## Current scope

스켈레톤에서 데이터 어댑터는 인메모리 구현입니다. 오케스트레이터의
`ExplicitTaskPlanner`도 테스트용이며, 명시적인 `requested_tasks`만 실행합니다.
자연어 계획은 추후 `TaskPlanner` 구현을 LLM 기반 플래너로 교체하여 추가합니다.

기존 코드에서 다음 구현을 포트 뒤로 이식할 수 있습니다.

| 기존 코드 | 새 위치 |
|---|---|
| `services/common/chroma_store.py` | `DocumentRepository` 구현체 |
| `services/gmail_agent/google_gmail.py` | `GmailGateway` 구현체 |
| `services/drive_agent/google_drive.py` | `DriveGateway` 구현체 |
| Ollama 라우터·생성 코드 | `LanguageModel`, `TaskPlanner` 구현체 |
| 기존 세션 메모리 | 메모리 Repository 구현체 |

새 프로젝트가 안정화될 때까지 기존 모듈을 직접 import하지 않는 것이 원칙입니다.
필요한 코드는 새 인터페이스에 맞게 이식합니다.

## Run

저장소 루트에서:

```bash
cd new_agent
docker compose up --build
```

로컬 Python 환경에서는 각 터미널에서 다음을 실행합니다.

```bash
cd new_agent
python -m pip install -e '.[dev]'
uvicorn agent_system.services.local_db.app:app --port 8001
uvicorn agent_system.services.gmail.app:app --port 8002
uvicorn agent_system.services.drive.app:app --port 8003
uvicorn agent_system.api.orchestrator:app --port 8000
```

API 문서:

- Orchestrator: `http://localhost:19000/docs`
- Local DB: `http://localhost:19001/docs`
- Gmail: `http://localhost:19002/docs`
- Drive: `http://localhost:19003/docs`

## Tool API

모든 도구 서비스는 공통 엔드포인트를 제공합니다.

```text
GET  /health
GET  /v1/capabilities
POST /v1/tasks
POST /v1/tools/{action}
```

Local DB 검색 예시:

```bash
curl -X POST http://localhost:19001/v1/tools/document_search \
  -H 'content-type: application/json' \
  -d '{
    "parameters": {"query": "example", "namespace": "knowledge"},
    "principal": {
      "user_id": "user-1",
      "session_id": "session-1",
      "permissions": ["document:read"]
    }
  }'
```

쓰기·삭제·메일 발송처럼 승인이 필요한 작업은 오케스트레이터가 고정된
`task_id`, `action`, `parameters`에 대한 `ApprovalReceipt`를 발급한 후
`POST /v1/tasks`로 실행하는 흐름을 사용합니다. 현재 digest 검사는 로컬
스켈레톤이며 운영 구현에서는 서명된 승인 토큰으로 교체해야 합니다.

## Test

```bash
cd new_agent
pytest
```

