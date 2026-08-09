# RAGPart & RAGMask 적용 계획

대상 논문: Pathmanathan et al., *RAGPart & RAGMask: Retrieval-Stage Defenses
Against Corpus Poisoning in Retrieval-Augmented Generation*
([arXiv:2512.24268](https://arxiv.org/abs/2512.24268))

## 논문 요약

두 방어 모두 **검색 단계(retrieval-stage)** 에서 동작한다. 생성 모델을 전혀
건드리지 않고, 리트리버 앞뒤에만 붙는다.

### RAGPart

핵심 관찰: dense 리트리버는 대조 학습 과정에서 문서의 일부 조각을 그 문서의
positive 예시로 학습한다(Contriever 등). 따라서 **문서 조각의 임베딩이 원본
문서의 의미를 거의 보존한다**는 귀납적 편향이 생긴다. RAGPart는 이 성질과
deep partition-and-aggregation(DPA) 방어를 결합한다.

1. 각 문서를 연속된 `N`개 조각으로 분할한다.
2. 조각을 **각각 따로** 임베딩한다.
3. 크기 `k`인 모든 부분집합 `C(N, k)`에 대해 조각 임베딩의 **평균**을 구한다.
   → 문서당 `C(N, k)`개의 표현이 생긴다.
4. `C(N, k)`개의 인덱스를 만들고, 질의마다 각 인덱스에서 top-p를 뽑는다.
5. `C(N, k)`개의 top-p 목록을 **다수결(majority vote)** 로 집계해 최종 top-p를
   결정한다.

오염 조각이 조합에 섞여도 평균 풀링이 그 영향을 `1/k`로 희석한다. 텍스트를
먼저 합친 뒤 임베딩하는 naive 방식(baseline)은 오염 토큰이 그대로 살아남아
방어가 되지 않는다는 것이 논문의 대조 실험이다.

집계 방식은 두 가지다. 교집합(intersection)은 더 안전하지만 유틸리티 손실이
크고, 논문도 대부분 실험에서 다수결을 쓴다.

논문 기본값은 `N=5, k=3`. 부록 하이퍼파라미터 표에서 `N`을 키우면 ASR과
유틸리티가 함께 내려간다(`N=5,k=3`: SR 60% / ASR 14%, `N=20,k=10`: SR 24% /
ASR 1%).

### RAGMask

상위 `αp`개 문서만 사후 정제(sanitize)한다. 문서를 길이 `m`짜리 구간으로
나눠 각 구간을 마스킹하고 질의와의 유사도를 다시 잰다. 마스킹했을 때 유사도가
크게 **떨어지는**(`v_masked + δ > v_orig`) 구간은 검색을 끌어올리기 위한
오염 토큰으로 보고 제거한 뒤, 정제된 문서로 재랭킹한다.

### 평가 지표 (논문 기준)

- **ASR**: 질의의 top-p 안에 오염 문서가 **하나라도** 들어간 비율 (낮을수록 좋음)
- **SR**: top-p 안에 정답 문서가 하나라도 들어간 비율 = 유틸리티 (높을수록 좋음)
- 방어는 "ASR 하락폭은 크게, SR 하락폭은 작게"로 평가한다.

논문의 ASR은 **검색 단계** 지표라서, 이 랩의 기존 답변 단계 ASR과 다른 축이다.
둘 다 측정해야 비교가 된다.

### 공격 및 기존 방어 baseline

논문이 다루는 공격은 HotFlip(그래디언트), HotFlip spread-out, **Query-as-Poison**,
AdvRAGgen이다. 이 중 Query-as-Poison은 문서에 질의문을 그대로 붙이는 방식으로,
이 랩이 이미 구현한 PoisonedRAG 블랙박스 공격 `P = Q || I`와 **같은 공격**이다.

논문은 paraphrase / perplexity 기반 방어가 이런 해석 가능한(interpretable)
공격에는 무력함을 보인다. Query-as-Poison의 perplexity는 119로 무오염 문서의
143과 구분되지 않는다.

## 이 랩에 적용할 때의 판단

### 왜 잘 맞는가

- 이 랩의 주력 공격이 논문의 Query-as-Poison과 동일하다. 논문이 "기존 방어가
  실패한다"고 지목한 바로 그 공격이라 방어 비교 실험의 대비가 뚜렷하다.
- 기존 `defended` 모드는 `trust` 메타데이터로 `untrusted` 문서를 걸러낸다.
  이는 "어떤 문서가 주입된 것인지 이미 안다"는 강한 가정이고, 실제 corpus
  poisoning에는 그런 라벨이 없다. **RAGPart는 trust 라벨을 전혀 쓰지 않는다.**
  따라서 이 랩에 처음으로 들어오는 "라벨 없는" 현실적 방어다.
- 생성 모델을 건드리지 않으므로 Ollama/Qwen 경로를 그대로 둔 채 붙는다.

### 주의할 점 (정직한 한계)

1. **임베딩 모델**: 이 랩의 `DeterministicHashEmbeddings`는 토큰 해시 기반
   bag-of-words이지 dense 리트리버가 아니다. 논문의 근거인 "학습 과정에서
   생긴 귀납적 편향"은 여기 없다.

   실제로 측정해보면(`N=5, k=3`, 질의 "What is the capital of France?"):

   | 문서 | 원본 문서 점수 | 점수>0인 조합 수 | 조합 평균 점수 |
   | --- | --- | --- | --- |
   | golden (`Paris is the capital…`) | 0.5345 | **9 / 10** | 0.4899 |
   | poison (`Q ‖ I`) | 0.2294 | **6 / 10** | 0.2397 |

   즉 **오염 문서의 평균 점수는 떨어지지 않는다.** 조각별 L2 정규화가 짧은
   조각을 오히려 증폭시키기 때문이다(질의 토큰이 37단어 전체에 정규화되는
   대신 7단어 조각에 정규화된다). 이 랩에서 RAGPart가 작동하는 실제 근거는
   점수 억제가 아니라 **조합 커버리지**다. 오염 문서는 질의 조각을 제외한
   4/10 조합에서 아예 검색되지 않으므로 다수결에서 밀린다.

   따라서 논문 수치를 그대로 재현하는 것은 아니고, 방어 효과는 다수결 집계에
   전적으로 의존한다. 실제 dense 리트리버(Ollama 임베딩, e5) 교체가 후속
   과제이며, 교체 후 이 표를 다시 측정해 비교해야 한다.
2. **문서 길이**: 현재 픽스처 문서는 평균 10단어다. `N=5`로 자르면 조각이
   2단어가 되어 유틸리티(SR) 손실이 논문보다 크게 나올 수 있다. 반면 오염
   문서는 `P = Q || I`로 약 37단어라 분할이 자연스럽게 동작한다. 이 비대칭
   자체가 실험에서 관찰할 대상이다. 구현은 문서가 짧으면 조각 수를 자동으로
   줄인다(단어 수보다 많은 조각을 만들지 않음).
3. **비용**: 문서당 `C(N,k)`개 벡터를 저장하고 질의당 `C(N,k)`번 검색한다.
   `N=5, k=3`이면 10배다. 랩 규모에서는 문제없지만 코퍼스를 키우면 고려해야
   한다.

## 적용 단계

### 1단계 — RAGPart

- [x] `services/common/ragpart.py`: 분할, 조합 임베딩 평균, 다수결 집계
- [x] `tests/test_ragpart.py`: 코어 알고리즘 단위 테스트 7개
- [ ] `ChromaDocumentStore`: 원본 컬렉션 옆에 `-ragpart` 보조 컬렉션 구축
- [ ] 검색 에이전트 `/search`에 `defense` 파라미터 추가 (`none` | `ragpart`)
- [ ] 오케스트레이터 `/query`, `/answer`, `/experiments/*`에
      `retrieval_defense` 전달
- [ ] 논문 기준 검색 단계 지표(ASR/SR)를 메트릭에 추가

#### 이어서 작업할 때 (인수인계)

브랜치: `feature/jhpark-ragpart` (main에서 분기)

코어 알고리즘 모듈과 계획 문서까지 끝났고, **검색 파이프라인 연결이 남았다.**
`services/common/ragpart.py`의 공개 함수 세 개만 쓰면 된다.

1. **`ChromaDocumentStore` 연결** (`services/common/chroma_store.py`)
   - `RagPartConfig`를 받아 보조 컬렉션 `{collection_name}-ragpart`를 만든다.
   - 색인: 문서마다 `partition_text` → 조각별 `embedding.embed_documents`
     → `combination_vectors` → `C(N,k)`개 벡터를 id `{doc_id}#c{j}`,
     메타데이터 `combo_index=j`로 저장한다.
   - 미리 계산한 벡터를 넣어야 하므로 langchain 래퍼가 아니라
     `vector_store._collection.add(embeddings=..., ids=..., metadatas=...)`를
     쓴다. `count()`가 이미 `_collection`을 쓰고 있으니 같은 방식이다.
   - 검색: 질의 벡터로 `combo_index=j`마다 `_collection.query(...,
     where={"combo_index": j}, n_results=limit)`를 `C(N,k)`번 호출해
     `(document_id, score)` 목록을 만든 뒤 `majority_vote(sets, limit)`.
     거리→점수 변환은 기존 `search()`와 동일하게 `1/(1+distance)`.
   - `add_document` / `delete_untrusted_document(s)`도 보조 컬렉션에 같이
     반영해야 한다. 실험 문서가 런타임에 주입·삭제되기 때문이다.
2. **에이전트**: `SearchRequest`에 `defense: Literal["none","ragpart"]`
   추가 → `create_search_agent`의 `/search`가 분기.
3. **오케스트레이터**: `OrchestratorQueryRequest`에 `retrieval_defense`를
   추가해 `_query_agents`의 payload로 전달.
4. **지표**: 논문 기준 검색 단계 ASR(top-k에 오염 문서 1개 이상) / SR(top-k에
   정답 문서 1개 이상)을 `AttackDashboardMetrics`에 추가.

주의: 오염 문서 방어 효과는 **다수결 집계에서만** 나온다(위 한계 1 참고).
따라서 `majority_vote`를 우회하고 조합 평균 점수로 랭킹하면 방어가 되지
않는다. 반드시 조합별 top-p를 따로 뽑아 집계해야 한다.

### 2단계 — 논문 정렬 평가

- `/experiments/poisoned-rag/benchmark`를 방어별로 sweep해서 `없음 / trust
  필터 / RAGPart` 의 ASR·SR 하락폭 비교표 생성
- `N`, `k` 하이퍼파라미터 스윕(논문 부록 Table 9/10 대응)

### 3단계 — RAGMask

- 상위 `αp` 문서에 대한 마스킹 기반 정제와 재랭킹
- `m`, `δ` 하이퍼파라미터 스윕
- RAGPart와의 유틸리티/비용 트레이드오프 비교

### 4단계 — 리트리버 교체 (선택)

- `DeterministicHashEmbeddings`를 실제 dense 리트리버로 교체해 논문의 귀납적
  편향 가정을 실제로 만족시킨 상태에서 재측정
