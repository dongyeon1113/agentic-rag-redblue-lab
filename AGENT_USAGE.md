# Complete agent usage

## Start

```bash
cp .env.example .env
docker compose up --build
```

첫 실행에서는 Ollama가 `qwen3:8b` 모델을 내려받으므로 시간이 걸릴 수 있습니다.
준비 상태는 다음 API로 확인합니다.

```bash
curl http://localhost:19000/ready
```

도구 라우팅 정확도를 위해 Qwen3 thinking은 기본 활성화됩니다. 필요하면 `.env`에서 끌 수 있습니다.

```text
OLLAMA_THINK=true
```

모델에는 매 요청의 실제 권한 목록, search/get/list 선택 기준, `secret` namespace의 조건부 권한이 함께 전달됩니다. 명백한 외부 데이터 요청에 모델이 도구 없이 답하면 에이전트 루프가 한 번만 도구 선택을 재판단시킵니다.

## Interactive Agent CLI

서비스를 시작한 뒤 별도 터미널에서 대화형 클라이언트를 실행합니다. 패키지를
editable 모드로 설치하면 `agent-cli` 명령이 등록됩니다.

```bash
python -m pip install -e ".[dev]"
agent-cli --base-url http://localhost:19000 --user-id user-1
```

설치하지 않고 실행하려면 다음 명령을 사용할 수 있습니다.

```bash
PYTHONPATH=src python -m agent_system.cli --base-url http://localhost:19000
```

CLI 프로세스가 실행되는 동안 같은 `session_id`를 사용하므로 후속 질문이 같은
대화로 연결됩니다. 재실행 후에도 기존 세션을 이어가려면
`--session-id experiment-1`처럼 명시합니다. 권한은 오케스트레이터의
`AGENT_PERMISSIONS` 설정에서만 부여하며, 쓰기 작업은 에이전트가 표시하는
작업·위험도·파라미터를 확인한 뒤 별도로 승인합니다.

```text
you> /permissions
agent> 서버 권한: document:read, drive:read, gmail:read, gmail:send
you> researcher@example.com으로 테스트 완료 메일을 보내줘
approval> 사용자 승인이 필요한 작업입니다.
위 작업을 모두 승인할까요? [y/N] y
agent> 메일을 보냈습니다.
```

주요 명령은 `/help`, `/new`, `/context context2`, `/permissions`,
`/memories`, `/exit`입니다. `--show-results`를 지정하면 원본 도구 실행 결과도
출력합니다. 기본 접속 설정은 `AGENT_API_URL`, `AGENT_USER_ID`,
`AGENT_SESSION_ID`, `AGENT_MEMORY_CONTEXT` 환경변수로도 지정할 수 있습니다.
`AGENT_PERMISSIONS`는 CLI가 아니라 오케스트레이터 컨테이너 설정입니다.

## Ask a natural-language question

일반 문서, Gmail, Drive 읽기 권한은 모의 환경의 기본 권한입니다.

```bash
curl -X POST http://localhost:19000/v1/agent/query \
  -H 'content-type: application/json' \
  -d '{
    "user_id": "user-1",
    "session_id": "session-1",
    "memory_context": "context1",
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

메일 발송, 삭제, Drive 이동은 서버 `AGENT_PERMISSIONS`에 해당 권한이 있을 때
먼저 `awaiting_approval`로 반환됩니다. 클라이언트 요청에는 권한을 넣을 수 없으며,
추가 필드는 422로 거부됩니다.

```bash
curl -X POST http://localhost:19000/v1/agent/query \
  -H 'content-type: application/json' \
  -d '{
    "user_id": "user-1",
    "session_id": "session-1",
    "query": "researcher@example.com으로 테스트 완료 메일을 보내줘"
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

## Long-term memory contexts and automatic storage

각 에이전트 요청에서 연결할 장기 메모리 컨텍스트를 지정합니다. 생략하면 `context1`입니다.

```json
{
  "user_id": "user-1",
  "session_id": "session-1",
  "memory_context": "context1",
  "query": "앞으로 주간 보고서는 간결하게 작성해줘"
}
```

저장 위치는 다음과 같습니다.

```text
/app/data/orchestrator/
├─ sessions.json                  # 모든 세션 대화
├─ long_term.json                 # context1 장기 메모리
├─ contexts/context2.json        # context2 장기 메모리
├─ contexts/context3.json        # context3 장기 메모리
└─ chroma/                       # 컨텍스트별 독립 Chroma 컬렉션
```

`memory_context`에는 `context1`, `context2`, ... 형식의 번호 기반 실험 ID만 허용됩니다. 경로나 임의 호스트 파일은 지정할 수 없습니다. 새로운 컨텍스트 ID를 지정하면
해당 JSON 파일과 Chroma 컬렉션이 지연 생성됩니다. 같은 사용자가 같은 세션을
사용하더라도 `memory_context`를 바꾸면 다른 장기 기억을 불러옵니다.

대화가 완료되면 LLM은 사용자 직접 진술 중 장기간 유효한 선호, 프로필, 프로젝트,
반복 워크플로와 제약만 추출하고 각 항목을 `preference`, `profile`, `project`, `workflow`, `constraint` 중 하나로 분류합니다. 이 분류는 같은 `contextN` 파일 내부의 메모리 항목 분류이며 파일 선택 기준은 아닙니다. 비밀정보, 일회성 요청, 도구·문서·메일 결과, 낮은 신뢰도 후보는 저장하지 않습니다. 특히 `secret` namespace 도구 결과가 포함된 실행은 LLM 추출 자체를 건너뛰고 세션 및 장기 메모리 양쪽에 저장하지 않습니다. 저장 직전에도 비밀번호·API 키·토큰 패턴을 다시 검사합니다. 저장된 항목은 응답의
`stored_memories`에서 확인할 수 있습니다. 자동 저장 실패는 원래 에이전트 응답을
실패시키지 않으며 서버 로그에 기록됩니다.

```text
GET    /v1/memory-contexts
GET    /v1/memories/{user_id}?memory_context=context1
POST   /v1/memories                  # body에 memory_context
PUT    /v1/memories/{memory_id}    # body에 memory_context
DELETE /v1/memories/{memory_id}?user_id={user_id}&memory_context=context1
```

일반 세션 대화는 bind mount된 호스트의 `data/orchestrator/sessions.json`에 시간순으로 자동 저장됩니다. 민감 질문과 해당 답변은 세션 JSON에 저장하지 않습니다. 기존 파일에 남아 있는 민감 턴과 장기 메모리는 다음 LLM 컨텍스트에서 제외됩니다. 장기 메모리는 컨텍스트별 JSON 원본과 Chroma 임베딩에 동기화되고, `user_id` 필터가 적용된 의미 검색으로
에이전트 컨텍스트에 로드됩니다.

자동 저장 정책은 다음 환경변수로 조정합니다.

```text
AUTO_MEMORY_ENABLED=true
AUTO_MEMORY_MIN_CONFIDENCE=0.8
AUTO_MEMORY_MAX_ITEMS=3
```

## Reset mock state

```bash
docker compose down -v
```

이 명령은 실행 중 생성·수정된 작업 데이터와 Ollama 모델 볼륨까지 삭제합니다.
Ollama 모델을 유지하려면 데이터 볼륨만 개별적으로 초기화하십시오.

