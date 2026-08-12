#!/usr/bin/env bash
# docker/start.sh
# Starts the spark-brain container with Nemotron 3.5 Lightning 30B and DSpark.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
SPECULATIVE_MODEL_ID="${SPECULATIVE_MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}" # 1M context

echo "=== vLLM Nemotron 3.5 Lightning preflight ==="
echo "  Main Model:        $MODEL_ID"
echo "  Speculative Model: $SPECULATIVE_MODEL_ID"
echo "  Served name:       $SERVED_MODEL_NAME"
echo "  Container:         $CONTAINER_NAME"
echo "  Image:             $VLLM_IMAGE"
echo "  Port:              $PORT"
echo "  Max model len:     $MAX_MODEL_LEN"
echo

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Gated models may fail to load."
  echo
fi

HF_ENV=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ENV=(-e "HF_TOKEN=${HF_TOKEN}")
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Cleaning up existing container..."
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

mkdir -p "$HOME/.cache/huggingface" "$HOME/.cache/triton" "$HOME/.cache/vllm"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Starting vLLM container..."
docker run -d --name "$CONTAINER_NAME" \
  --gpus all \
  --restart=unless-stopped \
  --ipc=host \
  --network host \
  --shm-size=32gb \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -v "$HOME/.cache/triton:/root/.cache/triton" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$SCRIPT_DIR/qwen3_dspark.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dspark.py" \
  "${HF_ENV[@]}" \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e HF_HUB_OFFLINE=1 \
  "$VLLM_IMAGE" \
    --model "$MODEL_ID" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --moe-backend marlin \
    --kv-cache-dtype fp8 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 4240 \
    --enforce-eager \
    --enable-prefix-caching \
    --speculative_config.method dspark \
    --speculative_config.model "$SPECULATIVE_MODEL_ID" \
    --speculative_config.num_speculative_tokens 3 \
    --mamba-backend flashinfer \
    --mamba-cache-mode align \
    --reasoning-parser nemotron_v3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"

echo
echo "Container started."
echo "Next: docker logs -f $CONTAINER_NAME"
echo "Ready check: curl -sf http://localhost:$PORT/health && echo OK"
