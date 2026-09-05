# New Agent Architecture

이 저장소의 현재 에이전트 구현입니다.
하나의 LLM 오케스트레이터를 전제로 하며, Local DB, Gmail,
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

- `application/`: LLM 도구 호출 루프와 메모리 유스케이스
- `contracts/`: 서비스 사이의 안정적인 HTTP DTO
- `ports/`: 메모리와 실행기 인터페이스
- `infrastructure/`: HTTP 클라이언트 및 메모리 어댑터
- `tool_runtime/`: 세 도구 서비스가 공유하는 실행·권한·승인 런타임
- `services/`: Local DB, Gmail, Drive 도구와 FastAPI 앱
- `security/`: 권한·승인용 `AgentGuard` 경계
- `defense/`: TaskShield와 검색 결과 방어 파이프라인

## Current scope

중앙 `ToolCallingAgent`가 Ollama로 사용자 요청을 해석하고, 각 도구 서비스의
capability를 바탕으로 필요한 작업을 단계적으로 선택·실행합니다. Local DB,
Gmail, Drive 검색은 JSON 원본 데이터를 Chroma에 인덱싱한 벡터 검색만 사용합니다.
세션 메모리는 호스트의 `data/orchestrator/sessions.json`에서 시간순으로 조회합니다. 장기 메모리는 방어 실험용 `context1`, `context2` 같은 독립 슬롯으로 구분하며, 각 JSON 원본과 사용자별 Chroma 벡터 검색을 사용합니다.
쓰기·삭제·메일 발송은 사용자 승인 후 실행됩니다.

## Run

저장소 루트에서:

```bash
docker compose up --build
```

로컬 Python 환경에서는 각 터미널에서 다음을 실행합니다.

```bash
python -m pip install -e '.[dev]'
uvicorn agent_system.services.local_db.vector_app:app --port 8001
uvicorn agent_system.services.gmail.vector_app:app --port 8002
uvicorn agent_system.services.drive.vector_app:app --port 8003
uvicorn agent_system.api.agent:app --port 8000
```

API 문서:

- Orchestrator: `http://localhost:19000/docs`

대화형 CLI는 서비스 시작 후 다음처럼 실행합니다.

```bash
python -m pip install -e ".[dev]"
agent-cli --base-url http://localhost:19000 --user-id user-1
```

동일한 CLI 프로세스에서는 같은 세션으로 후속 대화가 이어지며, 승인 필요 작업은
파라미터를 표시한 후 사용자 확인을 받습니다. Qwen3 tool calling은 `OLLAMA_THINK=true`가 기본이며, 서버가 부여한 권한과 상세 도구 선택 규칙이 모델 컨텍스트에 포함됩니다. 자세한 명령은 `/help` 또는
`AGENT_USAGE.md`를 참고하십시오.

## Security experiment GUI

기존 방어·공격 실험을 새 도구 호출 파이프라인에 통합했습니다. GUI는
`http://localhost:19010`에서 열 수 있습니다. 기존 `nq-defense-demo`처럼 NQ 시나리오를 선택하고 공격 문서 수, Top-K, Q+I, 공격 유형과 방어 조합을 설정해 취약/방어 결과를 한 화면에서 비교합니다. 자세한 구조와 Prompt Guard 실행법은
`SECURITY_EXPERIMENTS.md`를 참고하십시오.

## AgentDojo benchmark

`ToolCallingAgent` 자체를 AgentDojo 공식 suite에서 평가할 수 있습니다. 모든 suite
도구 권한을 부여한 격리 환경에서 baseline과 TaskShield, Regex, Prompt Guard, Spotlighting
프로필의 benign utility, utility under attack, targeted ASR을 비교합니다. GUI는
파일 기반 서버 작업, 진행 조회, 중단·재개와 결과 다운로드를 지원합니다. 설치와
실행 방법은 `AGENTDOJO_BENCHMARK.md`를 참고하십시오.

## Tool API

Docker 내부의 모든 도구 서비스는 공통 엔드포인트를 제공합니다.

```text
GET  /health
GET  /v1/capabilities
POST /v1/tasks
POST /v1/tools/{action}
```

하위 도구 API는 호스트에 공개하지 않고 Docker 내부 네트워크에서 오케스트레이터만 호출합니다.
쓰기·삭제·메일 발송처럼 승인이 필요한 작업은 오케스트레이터가 고정된
`task_id`, `action`, `parameters`에 대한 `ApprovalReceipt`를 발급한 후
`POST /v1/tasks`로 실행하는 흐름을 사용합니다. 현재 digest 검사는 로컬
스켈레톤이며 운영 구현에서는 서명된 승인 토큰으로 교체해야 합니다.

## Test

```bash
pytest
```
