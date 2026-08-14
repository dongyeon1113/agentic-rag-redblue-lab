# AgentPoison: 공식 구현과 이 저장소의 surrogate 차이

공식 저장소(`https://github.com/AI-secure/AgentPoison.git`, commit 기준 `algo/trigger_optimization.py`, `algo/config.py`)와
이 저장소의 `services/orchestrator/agent_poison.py`를 대조한 기록. 2026-08-14 확인.

## 공식 구현 (`algo/trigger_optimization.py`)

- **모델 접근**: white-box BERT 계열 retriever(`BertModel`)의 토큰 임베딩 행렬에 직접 접근. CUDA 필요.
- **목적함수**: `compute_avg_cluster_distance` — benign DB 임베딩에 Gaussian Mixture(`GaussianMixture(n_components=5)`)를 적합해 얻은 5개 클러스터 중심까지의 평균 거리에서 `0.1 * variance(query_embedding)`를 뺀 값을 최대화. `variance`는 평균 임베딩으로부터의 L2 거리 평균(compactness 대리 지표).
  - 대안 objective `compute_avg_embedding_similarity`(cpa 알고리즘)도 지원.
- **트리거 탐색**: gradient-guided HotFlip.
  1. 매 iteration마다 학습 배치(`num_grad_iter=30`)에 대해 objective를 역전파해 트리거 토큰 임베딩의 gradient를 누적.
  2. 무작위로 고른 한 토큰 위치(`token_to_flip`)에 대해 `gradient_dot_embedding_matrix`로 상위 `num_cand=100` 후보 토큰을 선정(HotFlip, Ebrahimi et al. 방식).
  3. 후보 각각으로 치환한 뒤 실제로 재임베딩해 objective를 재계산, 기존보다 개선되면 채택.
  4. 선택: `ppl_filter`(GPT-2 perplexity로 부자연스러운 토큰 배제), `target_gradient_guidance`(실제 target LLM의 target-word 확률/ASR로 후보 재선별).
  5. 기본 `num_iter=1000`, 트리거 길이 `num_adv_passage_tokens=10`.
- **평가**: AgentDriver/StrategyQA/EHRAgent 실제 데이터셋, Llama-2-7b-chat 등 실제 target LLM에 대해 ASR 측정.

## 이 저장소의 surrogate (`services/orchestrator/agent_poison.py`)

- **모델 접근**: black-box `Embeddings` 인터페이스만 사용(`embed_query`/`embed_documents`). 기본은 `DeterministicHashEmbeddings`(순수 해시, gradient 없음), 선택적으로 Ollama `nomic-embed-text`. GPU/transformer 불필요.
- **목적함수**: `score_trigger` — benign 문서 전체의 **단일 평균 중심**(GMM 아님)까지 거리(uniqueness) − `0.1 * (트리거된 쿼리들 자체 중심까지의 평균 거리)`(compactness). 논문의 uniqueness/compactness 방향성은 유지하되, 클러스터 개수(5)나 GMM 적합은 재현하지 않음.
- **트리거 탐색**: `optimize_trigger` — gradient 없는 좌표별 완전탐색 beam search.
  1. 매 iteration마다 고정 위치(`iteration % len(seed)`, 무작위 아님)의 토큰만 후보 목록(`candidate_tokens`, 보통 10개 미만)으로 전수 치환.
  2. HotFlip 랭킹 없이 모든 (beam × candidate) 조합을 실제로 재점수화해 정렬.
  3. 상위 `beam_width=4`개를 다음 iteration의 beam으로 유지, deterministic tie-break(문자열 정렬)로 재현성 보장.
  4. perplexity/coherence 필터, target-model gradient guidance 없음.
  5. 기본 `iterations=8` 내외로 호출(엔드포인트 스모크 테스트 기준), 후보 토큰도 소규모.
- **평가**: 이 저장소의 로컬 corpus(`datasets/generated/nq_100000.json`)와 로컬 `OLLAMA_MODEL`(예: `qwen3:8b`)로 `ASR-r`/`ASR-a`/`ASR-t`/`benign_accuracy` 계산. Chroma/local_db_agent에는 쓰지 않는 in-memory 실행.

## 실질적 함의

- 두 구현 모두 "uniqueness 최대화 / compactness 최소화"라는 논문의 방향성은 공유하지만, 공식 구현은 **gradient 기반 discrete optimization + white-box 임베딩 공간**을, 이 저장소는 **gradient 없는 완전탐색 + black-box 임베딩 API**를 사용함.
- 트리거 탐색 규모(공식: iterations=1000 × candidates=100 × gradient batch=30 vs. 이 저장소: iterations~8 × candidates<10, 무배치)가 수 자릿수 차이 나므로, 이 저장소의 ASR 수치를 논문 수치와 직접 비교할 수 없음.
- 이 저장소는 논문이 사용한 실제 target LLM(Llama-2-7b-chat 등)이 아닌 로컬 `OLLAMA_MODEL`로 평가하므로, 절대적인 공격 성공률이 아니라 **이 lab 환경 안에서의 상대적 신호**로만 해석해야 함.
- `AgentPoisonResponse.optimizer="embedding_discrete_beam_surrogate"`, `isolation="in_memory_no_database_writes"` 필드로 이 차이를 API 응답에도 명시함.

## 참고

- 논문: https://papers.nips.cc/paper_files/paper/2024/file/eb113910e9c3f6242541c1652e30dfd6-Paper-Conference.pdf
- 공식 저장소: https://github.com/AI-secure/AgentPoison.git
- 대조한 파일: `algo/trigger_optimization.py`, `algo/config.py`
