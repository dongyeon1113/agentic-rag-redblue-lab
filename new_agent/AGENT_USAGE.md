# Complete agent usage

## Start

```bash
cd new_agent
cp .env.example .env
docker compose -f compose.agent.yaml up --build
```

첫 실행에서는 Ollama가 `qwen3:8b` 모델을 내려받으므로 시간이 걸릴 수 있습니다.
준비 상태는 다음 API로 확인합니다.

```bash
curl http://localhost:19000/ready
```

## Ask a natural-language question

일반 문서, Gmail, Drive 읽기 권한은 모의 환경의 기본 권한입니다.

```bash
curl -X POST http://localhost:19000/v1/agent/query \
  -H 'content-type: application/json' \
  -d '{
    "user_id": "user-1",
    "session_id": "session-1",
    "query": "로컬 문서에서 Chicago Fire 시즌 4가 언제 방영됐는지 찾아줘"
  }'
```

Gmail 예시:

```bash
curl -X POST http://localhost:19000/v1/agent/query \
  -H 'content-type: application/json' \
  -d '{
    "user_id": "user-1",
    "session_id": "session-1",
    "query": "수신함에서 AgentDojo 실험 일정에 관한 메일을 찾아 요약해줘"
  }'
```

Drive 예시:

```bash
curl -X POST http://localhost:19000/v1/agent/query \
  -H 'content-type: application/json' \
  -d '{
    "user_id": "user-1",
    "session_id": "session-1",
    "query": "Drive에서 architecture 문서를 찾아 내용을 설명해줘"
  }'
```

## Approval flow

메일 발송, 삭제, Drive 이동은 먼저 `awaiting_approval`로 반환됩니다. 쓰기
권한도 요청에 명시해야 합니다.

```bash
curl -X POST http://localhost:19000/v1/agent/query \
  -H 'content-type: application/json' \
  -d '{
    "user_id": "user-1",
    "session_id": "session-1",
    "query": "researcher@example.com으로 테스트 완료 메일을 보내줘",
    "permissions": ["document:read", "gmail:read", "drive:read", "gmail:send"]
  }'
```

응답의 `workflow_id`와 `approval_requests[].task_id`를 사용합니다.

```bash
curl -X POST http://localhost:19000/v1/agent/workflows/WORKFLOW_ID/approve \
  -H 'content-type: application/json' \
  -d '{
    "user_id": "user-1",
    "session_id": "session-1",
    "approved_task_ids": ["TASK_ID"]
  }'
```

승인 대기 작업을 취소할 수도 있습니다.

```bash
curl -X DELETE \
  'http://localhost:19000/v1/agent/workflows/WORKFLOW_ID?user_id=user-1&session_id=session-1'
```

## Long-term memory API

```text
GET    /v1/memories/{user_id}
POST   /v1/memories
PUT    /v1/memories/{memory_id}
DELETE /v1/memories/{memory_id}?user_id={user_id}
```

세션 대화는 `orchestrator-data` 볼륨의 `sessions.json`에 자동 저장됩니다.
장기 메모리는 API로 관리하며 관련성이 있는 경우 에이전트 컨텍스트에 로드됩니다.

## Reset mock state

```bash
docker compose -f compose.agent.yaml down -v
```

이 명령은 실행 중 생성·수정된 작업 데이터와 Ollama 모델 볼륨까지 삭제합니다.
Ollama 모델을 유지하려면 데이터 볼륨만 개별적으로 초기화하십시오.

