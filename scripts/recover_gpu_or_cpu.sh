#!/usr/bin/env bash
set -uo pipefail

GPU_DEVICE="0000:01:00.0"
AUDIO_DEVICE="0000:01:00.1"
UPSTREAM_BRIDGE="0000:00:01.0"
EXPECTED_GPU_CONFIG="de 10 0a 22"
CPU_OLLAMA_PORT="${OLLAMA_CPU_PORT:-19005}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_DIR}/.env.shkwon"
CPU_OVERRIDE="${PROJECT_DIR}/compose.cpu-recovery.yaml"

COMPOSE=(
  docker compose
  -p shkwon-test
  --file "${PROJECT_DIR}/compose.yaml"
  --file "${PROJECT_DIR}/compose.snap.yaml"
  --file "${CPU_OVERRIDE}"
  --env-file "${ENV_FILE}"
)

log() {
  printf '[recovery] %s\n' "$*"
}

warn() {
  printf '[recovery] WARNING: %s\n' "$*" >&2
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    printf 'Run this script with sudo: sudo %s\n' "$0" >&2
    exit 2
  fi
}

validate_target() {
  local vendor device subordinate
  vendor="$(<"/sys/bus/pci/devices/${GPU_DEVICE}/vendor")"
  device="$(<"/sys/bus/pci/devices/${GPU_DEVICE}/device")"
  if [[ "${vendor}" != "0x10de" || "${device}" != "0x220a" ]]; then
    warn "Refusing reset: ${GPU_DEVICE} is not the expected NVIDIA RTX 3080."
    return 1
  fi
  if [[ ! -e "/sys/bus/pci/devices/${AUDIO_DEVICE}" ]]; then
    warn "Refusing reset: paired HDMI audio function ${AUDIO_DEVICE} is absent."
    return 1
  fi
  subordinate="/sys/bus/pci/devices/${UPSTREAM_BRIDGE}/reset_subordinate"
  if [[ ! -w "${subordinate}" ]]; then
    warn "Refusing reset: ${subordinate} is unavailable."
    return 1
  fi
}

gpu_config_bytes() {
  od -An -tx1 -N4 "/sys/bus/pci/devices/${GPU_DEVICE}/config" \
    2>/dev/null | xargs
}

unbind_if_bound() {
  local pci_device="$1"
  local expected_driver="$2"
  local driver_link="/sys/bus/pci/devices/${pci_device}/driver"
  local current_driver
  if [[ ! -L "${driver_link}" ]]; then
    return 0
  fi
  current_driver="$(basename -- "$(readlink -f -- "${driver_link}")")"
  if [[ "${current_driver}" != "${expected_driver}" ]]; then
    warn "${pci_device} is bound to unexpected driver ${current_driver}."
    return 1
  fi
  log "Unbinding ${pci_device} from ${current_driver}."
  printf '%s' "${pci_device}" > "/sys/bus/pci/drivers/${current_driver}/unbind"
}

probe_device() {
  local pci_device="$1"
  if [[ -e "/sys/bus/pci/devices/${pci_device}" ]]; then
    printf '%s' "${pci_device}" > /sys/bus/pci/drivers_probe 2>/dev/null || true
  fi
}

attempt_gpu_recovery() {
  local config_after

  systemctl stop nvidia-persistenced 2>/dev/null || true
  log "Checking for active NVIDIA device handles."
  if fuser /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm >/dev/null 2>&1; then
    warn "An active process still owns an NVIDIA device; skipping PCI reset."
    return 1
  fi

  log "Disabling automatic restart for the two wedged Ollama containers."
  docker update --restart=no \
    shkwon-test-ollama-1 team-project-ollama-1 >/dev/null 2>&1 || true

  if ! unbind_if_bound "${AUDIO_DEVICE}" snd_hda_intel; then
    warn "Could not detach the HDMI audio function safely."
    systemctl start nvidia-persistenced 2>/dev/null || true
    return 1
  fi
  if ! unbind_if_bound "${GPU_DEVICE}" nvidia; then
    warn "Could not detach the NVIDIA function safely."
    systemctl start nvidia-persistenced 2>/dev/null || true
    return 1
  fi

  if ! modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null; then
    warn "NVIDIA modules still have references; continuing with the driver-unbound bus reset only."
  fi

  log "Issuing one Linux-managed secondary-bus reset on ${UPSTREAM_BRIDGE}."
  if ! printf '1' > "/sys/bus/pci/devices/${UPSTREAM_BRIDGE}/reset_subordinate"; then
    warn "The kernel rejected reset_subordinate."
    return 1
  fi
  sleep 2

  config_after="$(gpu_config_bytes)"
  log "GPU config after reset: ${config_after:-unreadable}"
  if [[ "${config_after}" != "${EXPECTED_GPU_CONFIG}" ]]; then
    warn "PCI config did not recover; the GPU remains off the bus."
    return 1
  fi

  modprobe nvidia || return 1
  modprobe nvidia_modeset 2>/dev/null || true
  modprobe nvidia_drm 2>/dev/null || true
  modprobe nvidia_uvm || return 1
  probe_device "${GPU_DEVICE}"
  probe_device "${AUDIO_DEVICE}"
  systemctl start nvidia-persistenced 2>/dev/null || true

  if timeout 20 nvidia-smi >/dev/null 2>&1; then
    log "GPU recovery succeeded; nvidia-smi is healthy."
    nvidia-smi
    return 0
  fi
  warn "PCI config returned, but the NVIDIA driver did not become healthy."
  return 1
}

wait_for_url() {
  local url="$1"
  local attempts="${2:-60}"
  local index
  for ((index = 1; index <= attempts; index++)); do
    if curl --max-time 2 --fail --silent --show-error "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_cpu_fallback() {
  if [[ ! -f "${ENV_FILE}" || ! -f "${CPU_OVERRIDE}" ]]; then
    warn "CPU recovery files are missing."
    return 1
  fi

  log "Starting isolated CPU-only Ollama on host port ${CPU_OLLAMA_PORT}."
  if ! "${COMPOSE[@]}" config >/dev/null; then
    warn "CPU recovery Compose configuration is invalid."
    return 1
  fi
  if ! "${COMPOSE[@]}" up --detach ollama-cpu; then
    warn "CPU-only Ollama container failed to start."
    return 1
  fi
  if ! wait_for_url "http://127.0.0.1:${CPU_OLLAMA_PORT}/api/tags" 90; then
    warn "CPU-only Ollama API did not become ready."
    return 1
  fi

  log "Recreating local-db-agent against CPU embeddings."
  "${COMPOSE[@]}" up \
    --detach --no-deps --build --force-recreate local-db-agent || return 1
  if ! wait_for_url "http://127.0.0.1:19001/health" 180; then
    warn "local-db-agent did not become healthy against CPU Ollama."
    return 1
  fi

  log "Recreating orchestrator against CPU Ollama."
  "${COMPOSE[@]}" up \
    --detach --no-deps --build --force-recreate orchestrator || return 1
  if ! wait_for_url "http://127.0.0.1:19000/health" 60; then
    warn "orchestrator did not become healthy."
    return 1
  fi

  log "CPU fallback is ready."
  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"query":"hello","use_memory":false}' \
    http://127.0.0.1:19000/answer
  printf '\n'
}

main() {
  require_root
  cd -- "${PROJECT_DIR}"
  if [[ "${1:-}" == "--cpu-only" ]]; then
    log "Skipping the wedged GPU and starting CPU fallback directly."
    start_cpu_fallback
    exit $?
  fi

  validate_target || exit 1

  log "Initial GPU config: $(gpu_config_bytes)"
  if attempt_gpu_recovery; then
    log "GPU reset path completed successfully. Existing zombie Ollama containers remain disabled."
    exit 0
  fi

  warn "GPU recovery failed; switching the agent stack to CPU-only Ollama."
  start_cpu_fallback
}

main "$@"
