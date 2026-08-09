# 팀원용 빠른 시작

## 현재 범위

이 저장소는 연구용 멀티 에이전트 RAG 기준 환경이다. Orchestrator가 Local
DB, Gmail, Drive 검색 에이전트에 질의를 보내고, 검색 결과를 Ollama의
Qwen3:8b에 전달해 답변을 생성한다.

현재 Gmail과 Drive는 실제 Google 계정이 아닌 더미 JSON 데이터를 사용한다.
외부 인터넷에 공개하는 서비스가 아니라 승인된 연구 서버 내부에서 사용한다.

## 서버에서 처음 실행

필요 조건:

- Docker Engine과 Docker Compose
- NVIDIA GPU, 드라이버, NVIDIA Container Toolkit
- 현재 사용자가 `docker` 그룹에 포함되어 있을 것

```bash
git clone <TEAM_REPOSITORY_URL> team-project
cd team-project
./scripts/bootstrap.sh
```

스크립트는 `.env`를 만들고 컨테이너를 빌드한 뒤 Qwen3:8b를 자동으로
준비한다. 첫 실행은 모델 다운로드 때문에 시간이 걸릴 수 있다.

서버가 Snap Docker를 사용하면 스크립트가 `compose.snap.yaml`을 자동으로
함께 적용한다. 이 파일은 일부 Snap Docker 환경에서 AppArmor가 컨테이너
종료 신호를 차단하는 문제를 우회한다. 일반 Docker 환경에서는 적용되지
않는다.

검증:

```bash
docker compose ps
python3 scripts/smoke_test.py
```

`docker compose ps`에서 상시 서비스 5개가 `healthy`이고 스모크 테스트가
`Smoke test passed.`를 출력하면 정상이다. `ollama-model-init`은 모델을
준비하고 정상 종료되는 일회성 컨테이너다.

## 원격 접속

Mac에서 SSH 터널을 열어 서버의 Orchestrator에 접속한다.

```bash
ssh -p <SSH_PORT> -L 8000:localhost:8000 <USER>@<SERVER>
```

브라우저에서 `http://localhost:8000/docs`를 연다. 터널을 연 터미널을
종료하면 브라우저 연결도 종료된다.

## 자주 쓰는 명령

```bash
docker compose ps
docker compose logs -f orchestrator
docker compose restart
docker compose down
```

`docker compose down`은 컨테이너만 내리고 데이터 볼륨은 보존한다.
`docker compose down -v`는 ChromaDB와 Ollama 모델 볼륨을 삭제하므로
초기화가 필요한 경우가 아니면 사용하지 않는다.

Snap Docker 서버에서 이미지를 다시 빌드할 때는 항상 다음 명령을 사용한다.

```bash
./scripts/bootstrap.sh
```

## 에이전트 장기 메모리

`POST /answer`는 `session_id`별로 이전 질문과 답변을 파일(JSONL)에 저장하고,
다음 질의에서 관련도가 높은 turn을 검색 결과와 함께 컨텍스트에 넣는다.

```bash
curl 'http://localhost:8000/memory?session_id=demo-session'
curl -X DELETE 'http://localhost:8000/memory?session_id=demo-session'
```

컨텍스트가 전부 `trusted`였던 turn만 `trusted` 메모리로 남는다. 오염된
문서로 만들어진 답변은 `untrusted` 메모리가 되어 `defended` 모드에서
제외되므로, 공격이 대화 turn을 넘어 지속되는지 그대로 측정할 수 있다.
실험용 엔드포인트는 기본값이 `use_memory: false`라 반복 실험에 영향이 없다.

## 개발 순서

1. 작업 전에 `main`을 최신 상태로 만든다.
2. 개인 기능 브랜치를 만든다.
3. 코드와 테스트를 함께 작성한다.
4. `python3 -m pytest -q`와 스모크 테스트를 실행한다.
5. Pull Request로 `main`에 병합한다.

비밀 키, OAuth 파일, `.env`, 실제 메일, 실제 Drive 문서는 Git에 올리지
않는다.
