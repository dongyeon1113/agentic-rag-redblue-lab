# 에이전트 앱 API와 CLI 요청 가이드

이 문서는 Local DB, Gmail, Drive 검색 에이전트가 제공하는 API와 Bash CLI에서 요청하는 방법을 정리한다. 모든 예시는 저장소 루트에서 Docker Compose 스택을 실행한 상태를 기준으로 한다.

## 1. 에이전트 주소

| 앱 | Compose 서비스 이름 | 호스트 기본 URL | 컨테이너 네트워크 URL |
| --- | --- | --- | --- |
| Orchestrator | `orchestrator` | `http://127.0.0.1:8000` | `http://orchestrator:8000` |
| Local DB Agent | `local-db-agent` | `http://127.0.0.1:8001` | `http://local-db-agent:8000` |
| Gmail Agent | `gmail-agent` | `http://127.0.0.1:8002` | `http://gmail-agent:8000` |
| Drive Agent | `drive-agent` | `http://127.0.0.1:8003` | `http://drive-agent:8000` |

호스트 포트는 `.env`의 `ORCHESTRATOR_PORT`, `LOCAL_DB_PORT`, `GMAIL_PORT`, `DRIVE_PORT`로 변경할 수 있다. Compose 네트워크에서는 모든 앱이 컨테이너 포트 `8000`을 사용하며 서비스 이름으로 접근한다.

## 2. 공통 API

`services.common.agent_factory.create_search_agent()`가 세 에이전트에 공통으로 `/health`와 `/search`를 등록한다.

### `GET /health`

프로세스의 기본 상태와 앱 버전을 반환한다.

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8001/health
```

응답 예시:

```json
{
  "status": "ok",
  "service": "local-db-agent",
  "version": "0.1.0"
}
```

이 API는 프로세스가 응답하는지만 확인한다. 전체 시스템 상태는 오케스트레이터의 `GET /agents`, 모델 상태는 `GET /model`로 확인한다.

### `POST /search`

에이전트가 소유한 데이터에서 Top-K 문서를 검색한다.

요청 스키마:

| 필드 | 타입 | 필수 | 제약·기본값 |
| --- | --- | --- | --- |
| `query` | string | 예 | 1~500자 |
| `limit` | integer | 아니요 | 기본값 3, 범위 1~20 |

Local DB 직접 검색:

```bash
curl --fail-with-body --silent --show-error \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{"query":"What is the capital of France?","limit":3}' \
  http://127.0.0.1:8001/search
```

Gmail 검색:

```bash
curl --fail-with-body --silent --show-error \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{"query":"project meeting schedule","limit":5}' \
  http://127.0.0.1:8002/search
```

Drive 검색:

```bash
curl --fail-with-body --silent --show-error \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{"query":"security experiment report","limit":5}' \
  http://127.0.0.1:8003/search
```

응답의 각 `hit`은 다음 필드를 가진다.

| 필드 | 의미 |
| --- | --- |
| `document_id` | 저장소 내 문서 식별자 |
| `source` | 데이터 출처 또는 커넥터 이름 |
| `trust` | `trusted`, `untrusted`, `unknown` 등의 신뢰 표시 |
| `tags` | 문서 태그 배열 |
| `text` | 검색된 본문 |
| `score` | 유사도 점수; 이 구현에서는 클수록 관련성이 높음 |

응답 구조:

```json
{
  "service": "local-db-agent",
  "query": "What is the capital of France?",
  "hits": [
    {
      "document_id": "...",
      "source": "...",
      "trust": "trusted",
      "tags": [],
      "text": "...",
      "score": 0.81
    }
  ]
}
```

검색 호출 흐름은 `FastAPI search()` → `app.state.document_store.search()` → `ChromaDocumentStore.search()` → `Chroma.similarity_search_with_score()` 순서다.

## 3. Local DB 전용 API

Local DB 에이전트만 코퍼스 통계와 통제된 실험 문서 변경 API를 제공한다.

### `GET /stats`

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8001/stats
```

응답에는 `dataset`, `backend`, `counts.trusted`, `counts.untrusted`, `counts.total`이 포함된다.

### `POST /documents`

보안 실험용 문서를 Chroma에 추가한다. 서버가 `trust=untrusted`를 강제하므로 요청에서 trust를 지정할 수 없다.

```bash
curl --fail-with-body --silent --show-error \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{
    "document_id":"cli-experiment-001",
    "source":"cli-lab",
    "tags":["controlled","test"],
    "text":"Controlled experiment document added from the CLI."
  }' \
  http://127.0.0.1:8001/documents
```

요청 제약:

- `document_id`: 1~120자, 영문자 또는 숫자로 시작하고 이후 `A-Z`, `a-z`, `0-9`, `.`, `_`, `:`, `-`만 허용
- `source`: 1~200자
- `tags`: 최대 20개
- `text`: 1~10,000자
- 정의되지 않은 추가 필드 금지

정상 응답은 HTTP `201 Created`다. 같은 ID가 이미 있으면 `409 Conflict`, Chroma가 아닌 백엔드에서는 `501 Not Implemented`를 반환한다.

### `DELETE /documents/{document_id}`

특정 untrusted 실험 문서만 삭제한다.

```bash
curl --fail-with-body --silent --show-error \
  --request DELETE \
  http://127.0.0.1:8001/documents/cli-experiment-001
```

문서가 없으면 오류 대신 `deleted=false`를 반환한다. trusted 문서를 삭제하려 하면 `403 Forbidden`이다.

### `DELETE /documents/untrusted`

모든 untrusted 실험 문서를 삭제한다. 원본 trusted 코퍼스는 유지된다.

```bash
curl --fail-with-body --silent --show-error \
  --request DELETE \
  http://127.0.0.1:8001/documents/untrusted
```

이 작업은 해당 Local DB 컬렉션의 모든 untrusted 문서를 대상으로 하므로 공유 실험 환경에서는 실행 전에 `/stats`로 개수를 확인해야 한다.

## 4. Gmail·Drive 전용 API

### `GET /source-status`

현재 데이터가 샘플 fixture인지 실제 Google 동기화 결과인지 확인한다.

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8002/source-status

curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8003/source-status
```

샘플 모드 응답:

```json
{"source":"gmail","mode":"sample","connected":false}
```

`GMAIL_SYNC_ENABLED=true` 또는 `DRIVE_SYNC_ENABLED=true`로 시작해 동기화에 성공한 앱은 `mode=live`, `connected=true`를 반환한다. 이 값은 시작 시 읽은 환경 설정을 나타내며 매 요청마다 Google API 연결을 재검사하지는 않는다.

## 5. CLI 환경 변수로 요청하기

반복 호출에서는 URL을 작업 전용 셸 변수로 지정하면 편리하다.

```bash
ORCH_API='http://127.0.0.1:8000'
LOCAL_DB_API='http://127.0.0.1:8001'
GMAIL_API='http://127.0.0.1:8002'
DRIVE_API='http://127.0.0.1:8003'
```

현재 셸의 모든 하위 명령에서 사용하려면 `export LOCAL_DB_API`처럼 내보낼 수 있다. 비밀 값은 명령행이나 저장소 파일에 넣지 않는다.

변수 사용 예시:

```bash
curl --fail-with-body --silent --show-error \
  --header 'Content-Type: application/json' \
  --data '{"query":"retrieval poisoning defense","limit":5}' \
  "${LOCAL_DB_API}/search"
```

`jq`가 설치되어 있으면 결과를 읽기 쉽게 출력할 수 있다.

```bash
curl --fail-with-body --silent --show-error \
  --header 'Content-Type: application/json' \
  --data '{"query":"retrieval poisoning defense","limit":5}' \
  "${LOCAL_DB_API}/search" | jq .
```

프로젝트의 포트 설정을 바꾸려면 먼저 `.env.example`을 `.env`로 복사하고 값을 수정한 뒤 Compose를 다시 생성한다.

```bash
cp .env.example .env
docker compose up --detach --build
docker compose ps
```

`.env`는 Compose가 읽는 런타임 설정이다. 위의 `LOCAL_DB_API` 같은 셸 변수는 CLI 요청 편의를 위한 것이며 애플리케이션 설정을 변경하지 않는다.

## 6. 오케스트레이터를 통한 권장 호출

일반 사용자는 개별 에이전트보다 오케스트레이터를 호출하는 편이 적합하다. 오케스트레이터는 여러 소스를 병렬 검색하고, 결과 병합과 방어 정책을 적용한다.

전체 소스 검색:

```bash
curl --fail-with-body --silent --show-error \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{
    "query":"project security status",
    "sources":["local_db","gmail","drive"],
    "limit":3
  }' \
  http://127.0.0.1:8000/query
```

방어 모드 답변:

```bash
curl --fail-with-body --silent --show-error \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{
    "query":"What is the capital of France?",
    "sources":["local_db"],
    "limit":5,
    "mode":"defended",
    "regex_filter":true,
    "prompt_guard":false
  }' \
  http://127.0.0.1:8000/answer
```

오케스트레이터 경유 실험 문서 생성:

```bash
curl --fail-with-body --silent --show-error \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{
    "document_id":"cli-experiment-002",
    "source":"cli-lab",
    "tags":["controlled"],
    "text":"Controlled test passage."
  }' \
  http://127.0.0.1:8000/experiments/documents
```

직접 에이전트 호출은 검색 계층 단위 테스트와 장애 진단에 유용하다. 오케스트레이터 경유 호출은 소스 팬아웃, trust 필터, 정규식, Prompt Guard, Spotlighting, 답변 생성이 필요한 종단간 사용에 적합하다.

## 7. 컨테이너 네트워크에서 요청하기

호스트가 아니라 Compose 컨테이너 안에서는 `127.0.0.1:8001`이 Local DB 에이전트를 가리키지 않는다. 서비스 DNS 이름과 내부 포트를 사용해야 한다.

오케스트레이터 컨테이너에서 Python 표준 라이브러리로 Local DB 상태 확인:

```bash
docker compose exec orchestrator \
  python -c "import urllib.request; print(urllib.request.urlopen('http://local-db-agent:8000/health').read().decode())"
```

컨테이너 간 주소:

```text
http://local-db-agent:8000
http://gmail-agent:8000
http://drive-agent:8000
http://ollama:11434
```

## 8. 문서 UI와 빠른 점검

각 FastAPI 앱은 기본 OpenAPI 문서를 제공한다.

```text
http://127.0.0.1:8000/docs  # Orchestrator
http://127.0.0.1:8001/docs  # Local DB
http://127.0.0.1:8002/docs  # Gmail
http://127.0.0.1:8003/docs  # Drive
```

전체 상태를 CLI에서 확인하는 최소 절차:

```bash
docker compose ps
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/agents
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/model
curl --fail-with-body --silent --show-error http://127.0.0.1:8001/stats
```

## 9. 일반적인 오류

| 증상 | 확인 사항 |
| --- | --- |
| 연결 거부 | `docker compose ps`, `.env` 포트, `HOST_BIND` 확인 |
| `422 Unprocessable Entity` | JSON 문법, `query` 길이, `limit` 1~20, 허용 필드 확인 |
| `409 Conflict` | 실험 `document_id` 중복 여부 확인 |
| `403 Forbidden` | trusted 문서를 삭제하려 했는지 확인 |
| `501 Not Implemented` | Local DB가 Chroma가 아닌 lexical 백엔드로 실행 중인지 확인 |
| 검색 결과 없음 | `/stats`, Gmail/Drive `/source-status`, 질의 및 `limit` 확인 |
| 오케스트레이터의 소스별 `status:error` | 해당 에이전트 `/health`를 직접 호출하고 `docker compose logs <service>` 확인 |
| 모델 답변 `503` | `/model`과 Ollama 상태 확인; 검색 API 자체는 모델 없이도 호출 가능 |

직접 에이전트 API에는 인증이 없다. 기본 `HOST_BIND=127.0.0.1`을 유지하고, 이 연구용 스택을 인증·TLS 없이 외부 네트워크에 공개하지 않는다.
