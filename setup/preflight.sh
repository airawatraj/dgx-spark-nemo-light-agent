#!/usr/bin/env bash
# setup/preflight.sh
# Pre-benchmark preflight for DGX Spark · Nemotron-3.5-Lightning-30B
# Run this on the DGX Spark before docker/start.sh and benchmarking.
# Read-only — makes no changes to the system.
# Exit code: 0 = all checks passed, 1 = one or more FAIL
set -uo pipefail

PASS=0; WARN=0; FAIL=0
IMAGE="vllm/vllm-openai:latest"
MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
SPECULATIVE_MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_CACHE_DIR="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
SPEC_CACHE_DIR="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"
PORT=8000

GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"
BOLD="\033[1m"; DIM="\033[2m"; RESET="\033[0m"

pass()  { echo -e "  ${GREEN}✓ PASS${RESET}  $1"; ((PASS++));  }
warn()  { echo -e "  ${YELLOW}⚠ WARN${RESET}  $1"; ((WARN++)); }
fail()  { echo -e "  ${RED}✗ FAIL${RESET}  $1"; ((FAIL++));  }
header(){ echo -e "\n${BOLD}$1${RESET}"; echo "  $(printf '─%.0s' {1..55})"; }

echo -e "\n${BOLD}DGX Spark · Nemotron-3.5-Lightning-30B Preflight Check${RESET}"
echo -e "${DIM}  $(date)${RESET}"

# ── 1. GPU ────────────────────────────────────────────────────────────────────
header "1 / GPU & Driver"

if command -v nvidia-smi &>/dev/null; then
  pass "nvidia-smi found"
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  echo -e "     GPU:    ${GPU_NAME:-unknown}"
  echo -e "     Memory: ${GPU_MEM:-?} MiB"
  echo -e "     Driver: ${DRIVER:-unknown}"

  # GB10 / Blackwell shows as compute capability 12.x
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  echo -e "     Compute capability: ${CC:-unknown}"
  if [[ "${CC%%.*}" -ge 12 ]] 2>/dev/null; then
    pass "Blackwell (sm_12x) detected — FP8/FP4 operations natively supported"
  elif [[ "${CC%%.*}" -ge 9 ]] 2>/dev/null; then
    warn "Hopper (sm_9x) — check FP8/FP4 compatibility with your vLLM build"
  else
    warn "Compute cap ${CC:-unknown} — --kv-cache-dtype fp8 may have limitations; consider checking compatibility"
  fi
else
  fail "nvidia-smi not found — NVIDIA driver not installed or not on PATH"
fi

# ── 2. Swap ───────────────────────────────────────────────────────────────────
header "2 / Swap"

SWAP=$(free 2>/dev/null | awk '/^Swap:/ {print $2}' || echo "0")
if [[ "${SWAP:-0}" -eq 0 ]]; then
  pass "Swap is off — unified memory thrashing risk eliminated"
else
  fail "Swap is ON (${SWAP} kB) — run setup/install.sh or 'sudo swapoff -a' before benchmarking"
fi

# VmSwap for any existing vLLM container
EXISTING_PID=$(docker inspect --format '{{.State.Pid}}' spark-brain 2>/dev/null || echo "")
if [[ -n "$EXISTING_PID" && "$EXISTING_PID" != "0" ]]; then
  VMSWAP=$(grep VmSwap "/proc/$EXISTING_PID/status" 2>/dev/null || echo "VmSwap: 0 kB")
  if echo "$VMSWAP" | grep -q "0 kB"; then
    pass "Existing spark-brain container: VmSwap = 0"
  else
    warn "Existing spark-brain container: $VMSWAP (non-zero swap in use)"
  fi
fi

# ── 3. Memory ─────────────────────────────────────────────────────────────────
header "3 / System Memory"

FREE_GB=$(free -g 2>/dev/null | awk '/^Mem:/ {print $7}' || echo "?")
TOTAL_GB=$(free -g 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "?")
echo -e "     Total: ${TOTAL_GB} GB   Available: ${FREE_GB} GB"

if [[ "$TOTAL_GB" -ge 120 ]] 2>/dev/null; then
  pass "≥120 GB unified memory confirmed"
else
  warn "Total memory ${TOTAL_GB} GB — expected ~128 GB on DGX Spark"
fi

if [[ "$FREE_GB" -ge 80 ]] 2>/dev/null; then
  pass "${FREE_GB} GB free — enough headroom for model + KV cache"
elif [[ "$FREE_GB" -ge 50 ]] 2>/dev/null; then
  warn "${FREE_GB} GB free — may be tight; stop other processes before starting"
else
  fail "${FREE_GB} GB free — insufficient; need ≥80 GB free before launching vLLM"
fi

# ── 4. Disk ───────────────────────────────────────────────────────────────────
header "4 / Disk Space"

CACHE_DISK=$(df -BG "$HF_HOME" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4}' || echo "?")
echo -e "     HF cache location: $HF_HOME"
echo -e "     Available: ${CACHE_DISK} GB"
if [[ "$CACHE_DISK" -ge 80 ]] 2>/dev/null; then
  pass "≥80 GB free on model cache disk"
elif [[ "$CACHE_DISK" -ge 40 ]] 2>/dev/null; then
  warn "${CACHE_DISK} GB free — model weights are large; may be tight"
else
  fail "${CACHE_DISK} GB free — model weights require more space; free space first"
fi

# ── 5. Model weights ──────────────────────────────────────────────────────────
header "5 / Model Weights"

WEIGHT_COUNT=$(find "$MODEL_CACHE_DIR" -maxdepth 3 \
  \( -name "*.safetensors" -o -name "config.json" \) 2>/dev/null | wc -l)
if [[ "$WEIGHT_COUNT" -gt 0 ]]; then
  WEIGHT_SIZE=$(du -sh "$MODEL_CACHE_DIR" 2>/dev/null | cut -f1 || echo "?")
  pass "Base model found at $MODEL_CACHE_DIR ($WEIGHT_SIZE on disk, $WEIGHT_COUNT files)"
else
  fail "Base model weights not found at $MODEL_CACHE_DIR — run setup/download_model.sh first"
fi

SPEC_COUNT=$(find "$SPEC_CACHE_DIR" -maxdepth 3 \
  \( -name "*.safetensors" -o -name "config.json" \) 2>/dev/null | wc -l)
if [[ "$SPEC_COUNT" -gt 0 ]]; then
  SPEC_SIZE=$(du -sh "$SPEC_CACHE_DIR" 2>/dev/null | cut -f1 || echo "?")
  pass "Speculative model found at $SPEC_CACHE_DIR ($SPEC_SIZE on disk, $SPEC_COUNT files)"
else
  fail "Speculative model weights not found at $SPEC_CACHE_DIR — run setup/download_model.sh first"
fi

# ── 6. Docker & NVIDIA Container Toolkit ─────────────────────────────────────
header "6 / Docker & NVIDIA Container Toolkit"

if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
  DOCKER_VER=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
  pass "Docker daemon running (v$DOCKER_VER)"
else
  fail "Docker daemon not running — start Docker first"
fi

# Check NVIDIA container runtime is registered
if docker info 2>/dev/null | grep -q "nvidia"; then
  pass "NVIDIA container runtime registered in Docker"
else
  fail "NVIDIA container runtime not found — install nvidia-container-toolkit and restart Docker"
fi

# Check --gpus all works with a smoke test
if docker run --rm --gpus all --entrypoint nvidia-smi \
    "$IMAGE" --query-gpu=name --format=csv,noheader &>/dev/null 2>&1; then
  pass "--gpus all works inside $IMAGE"
elif docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi \
    --query-gpu=name --format=csv,noheader &>/dev/null 2>&1; then
  pass "--gpus all works (tested with nvidia/cuda image; vllm image may not have nvidia-smi)"
else
  warn "Could not verify --gpus all inside container — check nvidia-container-toolkit"
fi

# ── 7. vLLM Docker image ──────────────────────────────────────────────────────
header "7 / vLLM Image ($IMAGE)"

if docker image inspect "$IMAGE" &>/dev/null 2>&1; then
  IMG_SIZE=$(docker image inspect "$IMAGE" --format '{{.Size}}' 2>/dev/null || echo "?")
  IMG_GB=$(awk "BEGIN {printf \"%.1f\", $IMG_SIZE/1073741824}" 2>/dev/null || echo "?")
  pass "Image present locally (${IMG_GB} GB)"
else
  fail "Image $IMAGE not pulled — run: docker pull $IMAGE"
  echo -e "     ${DIM}This can take 10–30 min on first pull${RESET}"
fi

# ── 8. Port availability ──────────────────────────────────────────────────────
header "8 / Port $PORT"

if ss -tlnp 2>/dev/null | grep -q ":${PORT} " || \
   netstat -tlnp 2>/dev/null | grep -q ":${PORT} "; then
  # Check if it's our own container
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^spark-brain$"; then
    warn "Port $PORT in use — but spark-brain container is running (will be replaced by start.sh)"
  else
    fail "Port $PORT in use by another process — identify with: ss -tlnp | grep $PORT"
  fi
else
  pass "Port $PORT is free"
fi

# ── 9. HF_TOKEN ───────────────────────────────────────────────────────────────
header "9 / HF_TOKEN"

if [[ -n "${HF_TOKEN:-}" ]]; then
  MASKED="${HF_TOKEN:0:4}****${HF_TOKEN: -4}"
  pass "HF_TOKEN is set ($MASKED)"
else
  warn "HF_TOKEN not set — model weights may already be downloaded; set before running start.sh"
fi

# ── 10. vLLM flag compatibility ───────────────────────────────────────────────
header "10 / vLLM Flag Compatibility (dry-run)"

VLLM_HELP=$(docker run --rm "$IMAGE" --help 2>/dev/null || echo "")
if echo "$VLLM_HELP" | grep -q "kv-cache-dtype"; then
  pass "--kv-cache-dtype supported by this image"
else
  warn "--kv-cache-dtype not found in vllm --help — may be unsupported"
fi

if echo "$VLLM_HELP" | grep -q "reasoning-parser"; then
  pass "--reasoning-parser supported by this image"
else
  warn "--reasoning-parser not found in vllm --help — may be unsupported; check vLLM version"
fi

if echo "$VLLM_HELP" | grep -q "tool-call-parser"; then
  pass "--tool-call-parser supported by this image"
else
  warn "--tool-call-parser not found in vllm --help — may be unsupported; check vLLM version"
fi

# ── 11. Existing spark-brain container ────────────────────────────────────────
header "11 / Existing spark-brain Container"

if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^spark-brain$"; then
  STATUS=$(docker inspect spark-brain --format '{{.State.Status}}' 2>/dev/null || echo "unknown")
  if [[ "$STATUS" == "running" ]]; then
    warn "spark-brain already running — start.sh will stop and replace it"
  else
    warn "spark-brain container exists (status: $STATUS) — start.sh will remove it"
  fi
else
  pass "No existing spark-brain container — clean start"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}$(printf '═%.0s' {1..58})${RESET}"
echo -e "${BOLD}  Preflight Summary${RESET}"
echo -e "  ${GREEN}PASS: $PASS${RESET}   ${YELLOW}WARN: $WARN${RESET}   ${RED}FAIL: $FAIL${RESET}"
echo -e "${BOLD}$(printf '═%.0s' {1..58})${RESET}"
echo ""

if [[ "$FAIL" -gt 0 ]]; then
  echo -e "  ${RED}${BOLD}✗ Fix the FAIL items above before running start.sh${RESET}"
  echo ""
  exit 1
elif [[ "$WARN" -gt 0 ]]; then
  echo -e "  ${YELLOW}△ Warnings present — review before benchmarking${RESET}"
  echo -e "  ${GREEN}→ Proceed with: bash docker/start.sh${RESET}"
  echo ""
  exit 0
else
  echo -e "  ${GREEN}${BOLD}★ All checks passed — ready to benchmark${RESET}"
  echo -e "  ${GREEN}→ Proceed with: bash docker/start.sh${RESET}"
  echo ""
  exit 0
fi
