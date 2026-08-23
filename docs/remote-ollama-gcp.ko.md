# Google Cloud GPU의 Ollama 사용

이 구성은 API, RAG, Chroma와 장기 메모리는 기존 서버에서 실행하고,
생성 모델과 임베딩 계산만 Google Compute Engine GPU VM의 Ollama로
보낸다. 기본 `compose.yaml`의 로컬 GPU 실행 방식은 그대로 유지된다.

## 권장 구조

```text
애플리케이션 서버                         Google Cloud GPU VM
orchestrator / agents / Chroma  ───────▶  Ollama
                                           ├─ qwen3:8b
                                           ├─ llama3.2:3b
                                           ├─ llama3.2:1b
                                           └─ nomic-embed-text
```

두 서버는 같은 VPC 또는 Tailscale/WireGuard 같은 사설망으로 연결한다.
Ollama의 인증 없는 `11434` 포트를 공개 인터넷에 노출하지 않는다.

## 1. GPU VM 준비

Compute Engine에서 NVIDIA GPU가 연결된 Linux VM을 만든다. 8B 양자화
모델을 사용하는 현재 구성에는 VRAM 24GB의 L4 한 장부터 시작하는 것이
적절하다. NVIDIA 드라이버 설치와 GPU 연결 확인 후 Ollama를 설치한다.

```bash
nvidia-smi
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama가 사설망 인터페이스에서 요청을 받도록 systemd override를 만든다.

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

설정 반영 후 필요한 모델을 미리 받는다.

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
ollama pull qwen3:8b
ollama pull llama3.2:3b
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

GCP 방화벽은 애플리케이션 서버의 사설 IP 또는 VPN 대역에서 오는 TCP
11434만 허용한다.

## 2. 연결 확인

애플리케이션 서버에서 GPU VM의 사설 주소로 확인한다.

```bash
curl --fail --show-error http://GPU_VM_PRIVATE_IP:11434/api/tags
```

응답의 `models`에 위 모델들이 있어야 한다. 연결할 수 없다면 Ollama
서비스의 listen 주소, GCP 방화벽, OS 방화벽과 VPN 라우팅을 확인한다.

## 3. 원격 모드 실행

`.env.example`을 복사한 `.env`에 다음 값을 설정한다.

```dotenv
REMOTE_OLLAMA_BASE_URL=http://GPU_VM_PRIVATE_IP:11434
```

`GPU_VM_PRIVATE_IP`는 예시 문자열이므로 실제 VPC 또는 VPN IP로 바꾼다.
별도의 env 파일을 사용한다면(예: `.env.shkwon`) 그 파일에 같은 변수를
추가하고 Compose의 전역 옵션으로 지정한다.

그런 다음 원격 전용 override를 함께 지정한다.

```bash
docker compose \
  --env-file .env.shkwon \
  -f compose.yaml \
  -f compose.remote-ollama.yaml \
  up -d --build
```

Docker를 `sudo`로 실행할 때는 셸 앞에 붙인 환경변수가 기본적으로
보존되지 않는다. 위처럼 `REMOTE_OLLAMA_BASE_URL`을 env 파일에 넣는
방식을 권장한다. 일회성으로 전달하려면 다음과 같이 `sudo env`를 쓴다.

```bash
sudo env REMOTE_OLLAMA_BASE_URL="http://ACTUAL_PRIVATE_IP:11434" \
  docker compose \
  --env-file .env.shkwon \
  -f compose.yaml \
  -f compose.snap.yaml \
  -f compose.remote-ollama.yaml \
  -p shkwon-test \
  up -d --build
```

이 override는 다음을 수행한다.

- orchestrator와 local-db-agent를 원격 Ollama로 연결한다.
- 로컬 `ollama` 및 모델 초기화 서비스를 시작하지 않는다.
- orchestrator의 로컬 NVIDIA runtime/device 예약을 제거한다.
- 생성 모델과 임베딩 모델은 같은 원격 서버를 사용한다.

상태와 실제 연결 주소를 확인한다.

```bash
docker compose \
  -f compose.yaml \
  -f compose.remote-ollama.yaml \
  ps

curl --fail --show-error http://localhost:8000/health
curl --fail --show-error http://localhost:8000/models
```

## 4. 종료와 비용 관리

```bash
docker compose \
  -f compose.yaml \
  -f compose.remote-ollama.yaml \
  down
```

실험하지 않을 때는 GPU VM도 중지한다. 모델 디렉터리를 영구 디스크에
두면 VM 재시작 후 다시 다운로드할 필요가 없다. VM을 중지해도 영구
디스크와 고정 외부 IP 등의 비용은 남을 수 있다.

## 운영 시 주의사항

- `REMOTE_OLLAMA_BASE_URL`에는 애플리케이션 컨테이너에서 접근 가능한
  주소를 사용한다. `localhost`는 애플리케이션 컨테이너 자신을 뜻한다.
- 모델 변경 후에는 GPU VM에서 먼저 `ollama pull`을 실행한다.
- 원격 통신 지연을 고려해 필요하면 `REQUEST_TIMEOUT_SECONDS`를 높인다.
- 실험 재현성을 위해 Ollama 버전, 모델 tag와 digest를 실험 결과에 함께
  기록하는 것이 좋다.
- 공개망 연결이 불가피하면 인증과 TLS가 있는 reverse proxy를 앞에 둔다.
