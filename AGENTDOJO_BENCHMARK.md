# AgentDojo Benchmark

이 경로는 현재 `ToolCallingAgent`를 AgentDojo 공식 환경에서 평가하며,
`src/defense`의 Regex, Prompt Guard, Spotlighting을 선택적으로 비교합니다.

## 실험 조건

- AgentDojo benchmark version: `v1.2.2`
- 모든 suite 도구에 `agentdojo:all` 권한 부여
- 사용자 승인 없이 격리된 AgentDojo 환경에서 도구 즉시 실행
- baseline에서는 모든 방어 비활성화, 방어 프로필 실행 시 선택 필터만 활성화
- 케이스마다 새로운 에이전트, 세션 메모리, 환경 사용
- AgentDojo 공식 UserTask/InjectionTask와 상태 기반 oracle 사용

승인을 비활성화한 이유는 사람의 개입으로 공격 성공률이 낮아지는 효과를 제외하고,
에이전트가 간접 지시를 실제 도구 행동으로 변환하는지 측정하기 위해서입니다. 모든
변경은 실제 Gmail·Drive·은행이 아니라 테스트마다 복제되는 AgentDojo 환경에만
적용됩니다.

## 설치

```bash
python -m pip install -e '.[dev,benchmark]'
```

AgentDojo API 변화를 피하기 위해 `agentdojo==0.1.35`로 고정되어 있습니다.

## 실행

Ollama와 대상 모델을 먼저 실행합니다. 모델 설정은 기존 `OLLAMA_*` 환경 변수를
그대로 사용합니다.

AgentDojo의 다중 도구 호출에서는 thinking이 짧은 출력 예산을 모두 소비해 빈
응답을 만들 수 있으므로, 벤치마크에는 다음 전용 기본값이 적용됩니다.

```dotenv
AGENTDOJO_OLLAMA_THINK=false
AGENTDOJO_OLLAMA_NUM_PREDICT=1024
AGENTDOJO_REQUEST_TIMEOUT_SECONDS=300
```

프로젝트 `.env`에서 값을 바꿀 수 있습니다. 일반 에이전트의 `OLLAMA_THINK`와
`OLLAMA_NUM_PREDICT`, `REQUEST_TIMEOUT_SECONDS` 설정에는 영향을 주지 않습니다.

작은 smoke run:

```bash
agentdojo-benchmark \
  --suite workspace \
  --user-task user_task_0 \
  --injection-task injection_task_0 \
  --attack important_instructions \
  --force-rerun
```

Workspace 전체:

```bash
agentdojo-benchmark --suite workspace --attack important_instructions
```

네 suite 전체:

```bash
agentdojo-benchmark --suite all --attack important_instructions
```

방어 프로필 비교:

```bash
agentdojo-benchmark \
  --suite all \
  --attack important_instructions \
  --defense baseline \
  --defense regex \
  --defense spotlighting:delimiting \
  --defense prompt_guard
```

지원 프로필은 `baseline`, `regex`, `spotlighting:delimiting`,
`spotlighting:datamarking`, `spotlighting:encoding`, `prompt_guard`, `all`입니다.
각 프로필은 별도 출력 디렉터리와 pipeline 이름을 사용하므로 AgentDojo 캐시가
서로 섞이지 않습니다. AgentDojo 읽기 도구 결과는 `items`/`item` 레코드로
정규화하고 `metadata.trust=untrusted`로 표시한 뒤 방어 파이프라인에 전달합니다.

Prompt Guard는 `.[defense,benchmark]` 의존성과 Hugging Face 모델 접근 권한이
필요합니다. 모델과 임계값은 다음 환경 변수로 지정합니다.

```bash
PROMPT_GUARD_MODEL=meta-llama/Prompt-Guard-86M
PROMPT_GUARD_DEVICE=cuda
PROMPT_GUARD_THRESHOLD=0.9
```

위 값을 프로젝트 루트의 `.env`에 넣으면 CLI와 GUI가 시작할 때 자동으로
불러옵니다. 별도의 `source .env`는 필요하지 않습니다. 운영체제, systemd 또는
컨테이너에서 이미 설정한 환경변수는 `.env` 값보다 우선합니다.

정상 utility만 측정하려면 `--benign-only`, 공격 조건만 실행하려면
`--attack-only`를 사용합니다. `--user-task`와 `--injection-task`는 반복 지정할 수
있지만 단일 suite 실행에서만 허용됩니다.

## 결과

기본 출력 디렉터리는 `runs/agentdojo`입니다.

- AgentDojo 원본 trace: 모델 대화, 함수 호출, 주입 문자열, utility/security 판정
- `current-agent-summary.json`: suite별 요약 및 케이스별 boolean 결과

요약 지표:

- `benign_utility`: 공격이 없을 때 정상 과업 성공률
- `utility_under_attack`: 공격 데이터가 있을 때 정상 과업 성공률
- `targeted_asr`: AgentDojo 공격 목표가 실제 상태 변화로 달성된 비율
- `injection_task_capability`: 공격 목표를 직접 요청했을 때 모델이 수행할 수 있는지

AgentDojo 내부의 `security=True`는 안전하다는 뜻이 아니라 공격 목표가 달성됐다는
뜻입니다. 요약 파일에서는 혼동을 피하기 위해 `attack_success_results`로 저장합니다.

## Trace GUI

```bash
agentdojo-benchmark-gui
```

브라우저에서 `http://127.0.0.1:19020`을 엽니다. GUI는 기본적으로
`runs/agentdojo`의 기존 trace를 읽으며 다음 내용을 정상/공격 실행으로 나란히
표시합니다.

- AgentDojo 사용자 쿼리와 공격 목표
- 외부 데이터에 삽입된 실제 공격 문자열과 injection vector
- 정상 실행 및 공격 실행의 최종 답변
- 순서가 보존된 도구 이름과 호출 인수
- utility, targeted attack success, 실행시간

화면 상단에서 전체 또는 단일 suite/태스크와 여러 방어 프로필을 골라 서버 작업을
생성할 수 있습니다. HTTP 요청은 작업 생성 직후 끝나며 벤치마크는 서버 프로세스의
백그라운드 task에서 계속됩니다. 브라우저나 SSH 연결이 끊겨도 서버 프로세스가
살아 있으면 실행은 계속됩니다.

작업별 파일은 `runs/agentdojo/jobs/<job-id>/`에 저장됩니다.

- `request.json`: 실행 요청
- `state.json`: queued/running/completed/failed/cancelled/interrupted 상태
- `checkpoint.json`: 완료된 방어 프로필과 부분 결과
- `result.json`: 최종 다운로드 결과

GUI에서 저장 작업 목록, 진행률, 중단, 재개, 결과 다운로드를 지원합니다. 서버
프로세스 자체가 재시작되면 실행 중 작업은 `interrupted`로 표시되며, 재개 버튼을
누르면 완료된 방어 프로필은 건너뛰고 체크포인트부터 계속합니다. 중단 요청은 현재
AgentDojo 프로필 실행이 끝난 경계에서 반영됩니다.

### GUI 방어 조합

GUI의 체크박스에서 Regex, Prompt Guard, Spotlighting을 골라 `선택 조합 추가`를
누르면 하나의 비교 조합이 생성됩니다. Spotlighting을 선택한 경우 delimiting,
datamarking, encoding 중 한 방식을 지정합니다. 예를 들면 다음 조합들을 한 작업에
동시에 추가할 수 있습니다.

```text
baseline
regex
regex+prompt_guard
regex+spotlighting:delimiting
regex+prompt_guard+spotlighting:encoding
```

`정상 + 공격 실행`을 누르면 생성 목록의 각 조합을 독립된 pipeline과 결과
디렉터리로 순차 실행합니다. 진행률의 전체 개수는 생성된 조합 수이며, 중단·재개 시
이미 완료된 조합은 다시 실행하지 않습니다.

각 서버 작업은 고유 결과 디렉터리를 사용하므로 작업 내부에서는 유효한 AgentDojo
trace를 캐시합니다. 일시적인 Ollama 오류로 profile이 실패한 뒤 `재개`하면 완료된
user/injection task는 건너뛰고, 불완전하거나 실패한 task부터 다시 실행합니다.

다른 trace 디렉터리를 열려면 다음처럼 지정합니다.

```bash
AGENTDOJO_OUTPUT_DIR=/path/to/runs agentdojo-benchmark-gui
```
