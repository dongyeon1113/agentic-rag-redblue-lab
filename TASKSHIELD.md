# TaskShield integration

현재 구현은 사용자 목표를 유일한 권위 있는 task 집합으로 두고, 에이전트 실행 중
새로 등장하는 지시가 그 목표에 기여하는지를 검사한다. ACL 2025 Task Shield의
instruction extraction, `ContributesTo` scoring, structured feedback 세 단계를
현재 `ToolCallingAgent` 경계에 맞춰 적용했다.

원 논문:

- <https://aclanthology.org/2025.acl-long.1435/>
- <https://arxiv.org/html/2412.16682>

## 검사 지점

```text
사용자 요청
  -> 사용자 task 추출
  -> LLM tool call 검사
      -> 정렬됨: 권한/승인 파이프라인으로 전달
      -> 비정렬: 실행 차단 + LLM에 구조화된 피드백
  -> tool output 내 actionable instruction 추출·검사
      -> 원본 결과 + 비정렬 지시를 무시하라는 피드백을 LLM에 전달
  -> 최종 assistant response의 actionable instruction 검사
      -> 비정렬이면 재생성, 반복 실패 시 안전 응답
```

TaskShield는 권한 검사나 사용자 승인을 대체하지 않는다. 실행 순서는 TaskShield
tool-call 검사, 간접 쓰기 차단, 승인 요청, 도구 서비스의 권한 검증 순이다. 따라서
정렬된 쓰기 작업도 기존 정책상 승인이 필요하다.

## 설정

```dotenv
TASKSHIELD_ENABLED=true
TASKSHIELD_THRESHOLD=0.0
TASKSHIELD_FAIL_CLOSED=true
TASKSHIELD_MAX_FEEDBACK_ROUNDS=2
```

- `TASKSHIELD_ENABLED`: API 요청에서 `defense`를 생략할 때 기본 활성화 여부
- `TASKSHIELD_THRESHOLD`: task별 기여도 합이 이 값보다 커야 허용
- `TASKSHIELD_FAIL_CLOSED`: 추출·판정 모델 실패 시 tool call을 차단할지 여부
- `TASKSHIELD_MAX_FEEDBACK_ROUNDS`: 최종 응답 재생성 최대 횟수

현재는 원 논문처럼 에이전트와 동일한 Ollama 모델을 temperature 0 설정으로
사용한다. 별도 모델 다운로드나 컨테이너는 필요하지 않지만, 추출과 판정 때문에
LLM 호출 횟수와 지연이 늘어난다.

요청별로 명시하려면 다음처럼 전달한다. `defense` 객체를 명시하면 환경변수의 기본
활성화 값보다 요청 설정이 우선한다.

```json
{
  "user_id": "test-user",
  "session_id": "taskshield-1",
  "query": "수신함을 요약해줘",
  "defense": {
    "task_shield": true,
    "block_indirect_actions": false
  }
}
```

`defense_report`의 주요 계측값은 다음과 같다.

- `taskshield_user_tasks`: 현재 요청에서 추출한 권위 있는 사용자 task 목록
- `taskshield_checks`: 사용자 task 추출, tool call/output, 응답 검사 횟수
- `taskshield_blocked_calls`: 실행 전에 차단한 tool call 수
- `taskshield_flagged_instructions`: tool output 또는 응답에서 찾은 비정렬 지시 수
- `taskshield_feedback_messages`: 에이전트에 반환한 교정 피드백 수
- `taskshield_checker_failures`: 구조화 출력 또는 모델 호출 실패 수

## AgentDojo

TaskShield만 평가하려면 기존 간접 쓰기 차단을 끈 독립 프로필을 사용한다.

```bash
agentdojo-benchmark \
  --suite workspace \
  --attack important_instructions \
  --defense baseline \
  --defense task_shield
```

GUI에서도 `Task Shield` 체크박스를 단독 또는 Regex, Prompt Guard, Spotlighting과
조합할 수 있다. 예: `task_shield`, `regex+task_shield`.

## 한계

이 구현은 Qwen의 instruction extraction과 기여도 판단 품질에 의존한다. 논문에서
보고한 다른 모델의 수치를 이 구현의 성능으로 간주하면 안 된다. 실제 효과는
AgentDojo의 benign utility와 targeted ASR을 함께 측정해 확인해야 한다. adaptive
attack, 모호한 사용자 목표, 긴 tool output, 구조화 출력 실패는 별도 평가 대상이다.

## 검증

```bash
python -m compileall -q src
pytest -q
docker compose --env-file .env.example config -q
```
