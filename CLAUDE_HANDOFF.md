# Claude CLI 인수인계서

## 현재 상태

- 저장소: `agentic-rag-redblue-lab`
- 브랜치: `codex/agentpoison-lab`
- 원격 `main`/서버에는 반영하지 않았다.
- 기존 PoisonedRAG 기능과 사용자 변경은 보존해야 한다.
- 현재 미커밋 변경: `services/common/schemas.py`, `services/orchestrator/app.py`, `services/orchestrator/static/demo.html`, 신규 `services/orchestrator/agent_poison.py`, 신규 `tests/test_agent_poison.py`.

상위 디렉터리의 `AGENTS.md`를 반드시 먼저 읽어라. 임의 reset/clean/delete를 하지 말고, `.env`, `secrets/`, OAuth/API/SSH 비밀값을 출력하거나 커밋하지 말 것.

## 프로젝트 구조

- `services/orchestrator/app.py`: FastAPI 오케스트레이터. `/query`, `/answer`, `/experiments/*`, `/demo` 제공.
- `services/orchestrator/rag.py`: Ollama Chat 모델, context 조합, vulnerable/defended 모드.
- `services/orchestrator/evaluation.py`: expected/target phrase 기반 판정.
- `services/orchestrator/poisoned_rag.py`: 기존 PoisonedRAG. passage `I` 생성·검증 후 `P = Q + I`로 untrusted Chroma에 넣는다.
- `services/orchestrator/static/demo.html`: 기존 PoisonedRAG GUI와 새 AgentPoison 패널.
- `services/local_db_agent/app.py`: Local DB 검색 에이전트. Chroma와 untrusted 문서 API 제공.
- `services/common/chroma_store.py`: Chroma persistence, trust metadata, 검색 점수.
- `services/common/embeddings.py`: 기본 `DeterministicHashEmbeddings`, 선택적 Ollama embeddings.
- `services/common/schemas.py`: Pydantic API 스키마.
- `datasets/sample/`: trusted fixture.
- `datasets/generated/nq_100000.json`: 새 AgentPoison이 메모리로 읽는 NQ corpus.
- `datasets/experiments/nq_target_queries.json`: 기존 PoisonedRAG 시나리오 100개.
- `tests/`: API, 검색, 평가, PoisonedRAG, AgentPoison 테스트.

## 기존 PoisonedRAG

`POST /experiments/poisoned-rag`는 untrusted 문서를 삭제하고 baseline을 계산한다. LLM으로 passage를 생성·검증하고 `P = Q + I`로 저장한 뒤 검색 점수로 선별한다. attacked 답변, ASR, accuracy, poison-in-top-k, precision/recall/F1, generation/total time을 반환한다.

`POST /experiments/poisoned-rag/benchmark`는 N=0/1/3/5 반복, JSON/CSV 저장, 종료 후 정리를 수행한다. 이 기능을 깨뜨리지 말 것.

## AgentPoison 추가분

공식 AgentPoison은 공통 trigger가 포함된 query를 embedding 공간의 독특하고 조밀한 영역으로 보내 악성 key/value memory를 검색시키며, 공식 구현은 transformer white-box gradient-guided discrete beam search를 사용한다.

이 브랜치 구현은 GPU/외부 모델 의존성을 피하는 격리 surrogate다. 공식 결과와 동일하다고 주장하지 말고, 논문의 uniqueness/compactness 목적함수와 평가 프로토콜을 현재 embedding interface로 재현한다.

### 신규 코드

`services/orchestrator/agent_poison.py`

- `score_trigger`: benign center와의 거리(uniqueness), triggered center와의 거리(compactness)를 계산.
- `optimize_trigger`: 후보 token을 좌표별로 바꾸는 deterministic beam search.
- `rank_memory`: embedding distance로 memory top-k를 반환.

`POST /experiments/agent-poison`

- 입력: `train_queries`, `test_queries`, `target_action`, `seed_trigger`, `candidate_tokens`, `poison_count`, `top_k`, `iterations`, `benign_corpus_limit`.
- `datasets/generated/nq_100000.json`을 최대 corpus limit까지 메모리로 읽는다.
- Chroma/Local DB에는 쓰지 않는다. poison memory도 Python tuple로만 생성한다.
- clean, clean-under-poison, triggered 답변을 평가한다.
- 응답에 `optimizer=embedding_discrete_beam_surrogate`, `isolation=in_memory_no_database_writes`를 명시한다.
- JSON은 `EXPERIMENT_RESULTS_DIR/agentpoison-<id>.json`에 저장한다.

지표는 `ASR-r`(poison top-k 검색률), `ASR-a`(검색된 경우 target action률), `ASR-t`(전체 target action률), `benign_accuracy`(정규화 문자열 일치), `poison_rate`다.

GUI 하단 AgentPoison 패널은 줄바꿈 질의를 받아 trigger, 지표, objective, trial 답변을 표시한다.

## 검증 명령

```bash
cd '/Users/dongyeon/Documents/산학협력프로젝트 2/agentic-rag-redblue-lab'
.venv/bin/python -m pytest -q tests/test_agent_poison.py tests/test_poisoned_rag.py tests/test_agents.py
.venv/bin/python -m compileall -q services
git diff --check
git status --short --branch
```

현재 결과: 22 tests passed, compileall passed, diff check passed.

## 다음 작업

1. ~~로컬 Ollama 또는 TestClient에서 `/experiments/agent-poison`을 호출해 실제 응답과 JSON artifact를 확인한다.~~
   - 2026-08-14 완료: `.venv/bin/python`으로 `fastapi.testclient.TestClient` + 로컬 Ollama(`qwen3:8b`)를 사용해
     소규모 파라미터(train_queries=2, test_queries=1, iterations=2, benign_corpus_limit=10)로 실호출.
     `status_code=200`, 응답에 `optimizer=embedding_discrete_beam_surrogate`,
     `isolation=in_memory_no_database_writes` 확인. JSON artifact
     `/tmp/poisonedrag-results/agentpoison-006a199c6b0f.json` 정상 저장 확인.
     `asr_r=1.0`(트리거 쿼리가 poison memory를 top-k에서 검색), `asr_a=0.0`/`asr_t=0.0`
     (qwen3:8b가 target_action을 그대로 따르지 않음 — 공격 성공 여부는 모델/프롬프트에 따라 달라짐, 버그 아님).
2. `nq_100000.json` 존재/형식을 확인하고, 없을 때만 소규모 fixture fallback을 추가한다.
   - 2026-08-14 확인: `datasets/generated/nq_100000.json` 존재, 100,000개 레코드, 각 항목
     `id/source/trust/tags/text` 스키마. Fallback 불필요.
3. ~~`/demo`에서 새 패널과 기존 PoisonedRAG GUI를 모두 확인한다.~~
   - 2026-08-14 완료: orchestrator(8000) + local_db_agent(8001) 로컬 기동 후 `/demo` HTML과 JS 폼 바인딩,
     `/experiments/corpus-status`, `/experiments/scenarios`, `/experiments/agent-poison` 실호출로 확인.
     또한 `agentPoisonLab` 섹션이 `.workspace` 밖(2열 grid의 엉뚱한 3번째 child)에 있던 레이아웃 버그를 발견해
     `.workspace` 안, 기존 PoisonedRAG "N별 반복 평가" 섹션 바로 다음으로 이동시킴.
4. ~~필요 시 AgentPoison 반복 benchmark를 추가하되 반드시 in-memory 격리한다.~~
   - 2026-08-14 완료: `POST /experiments/agent-poison/benchmark` 추가 (poison_count 스윕 × repetitions,
     실패 격리, JSON/CSV 저장 — PoisonedRAG의 `/experiments/poisoned-rag/benchmark`와 동일 패턴,
     `run_agent_poison_experiment`를 내부 함수로 직접 재호출). 스키마: `AgentPoisonBenchmarkRequest/Point/Failure/Response`
     (`services/common/schemas.py`). GUI에 `#agentPoisonBenchmark` 섹션 신규 추가 — ASR-r/ASR-t 바 차트,
     JSON/CSV 다운로드, PoisonedRAG 벤치마크 UI와 동일한 시각 언어(`.benchmark`/`.chart`/`.bar-group` 재사용).
     AgentPoison 패널도 PoisonedRAG 사이드바 폼과 동일한 `.field`/`.triple`/`.double`/`.audit`/`.answer` 구조로
     리디자인하고, 논문 vs 이 랩 surrogate 하이퍼파라미터 비교 박스(`.paper`)를 추가함.
     엔드포인트 스모크 테스트: `POST /experiments/agent-poison/benchmark` (poison_counts=[1,3], repetitions=1)
     → 200, points 2개, failures 0, JSON/CSV 아티팩트 다운로드 200 확인.
5. ~~공식 gradient 구현과 surrogate 차이를 문서화한다.~~
   - 2026-08-14 완료: 공식 저장소(`AI-secure/AgentPoison`)의 `algo/trigger_optimization.py`,
     `algo/config.py`를 `gh api`로 직접 조회해 대조. 문서: [`docs/agent_poison.md`](docs/agent_poison.md).
     핵심 차이: 공식은 white-box BERT gradient + HotFlip + GMM 5-cluster objective + 실제 target LLM 기반
     ASR 측정(iterations=1000, candidates=100, grad batch=30)인 반면, 이 저장소는 black-box embedding
     인터페이스 + 좌표별 완전탐색 beam search + 단일 benign centroid 기반 objective(iterations~8, 후보 <10개).
     ASR 수치는 논문 수치와 직접 비교 불가.
6. 검증 결과를 사용자에게 먼저 보고한 뒤에만 커밋/원격 반영을 논의한다. (진행 중 — 이 세션에서 보고함, 커밋/원격 반영은 아직 미논의)

## 공식 참고자료

- 논문: https://papers.nips.cc/paper_files/paper/2024/file/eb113910e9c3f6242541c1652e30dfd6-Paper-Conference.pdf
- 저장소: https://github.com/AI-secure/AgentPoison.git
- 참고 코드: `algo/trigger_optimization.py`, `algo/config.py`, `ReAct/local_wikienv.py`, `ReAct/eval.py`.
