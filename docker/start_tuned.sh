#!/usr/bin/env bash
# docker/start_tuned.sh
# High-throughput vLLM runner for Nemotron 3.5 Lightning NVFP4 + DSpark
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.27.1}"
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
SPECULATIVE_MODEL_ID="${SPECULATIVE_MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"

echo "=== Starting Tuned vLLM Server ==="
echo "  Main Model:               $MODEL_ID"
echo "  Speculative Model:        $SPECULATIVE_MODEL_ID"
echo "  Num Speculative Tokens:   $NUM_SPECULATIVE_TOKENS"
echo "  Served Model Name:        $SERVED_MODEL_NAME"
echo "  Container:                $CONTAINER_NAME"
echo "  Image:                    $VLLM_IMAGE"
echo "  Port:                     $PORT"
echo "  Max Model Len:            $MAX_MODEL_LEN"
echo "  GPU Memory Utilization:   $GPU_MEMORY_UTILIZATION"
echo

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Cleaning up existing container..."
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

mkdir -p "$HOME/.cache/huggingface" "$HOME/.cache/triton" "$HOME/.cache/vllm"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

HF_ENV=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ENV=(-e "HF_TOKEN=${HF_TOKEN}")
fi

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
  "${HF_ENV[@]}" \
  "$VLLM_IMAGE" \
    --model "$MODEL_ID" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 1 \
    --moe-backend marlin \
    --kv-cache-dtype fp8 \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-prefix-caching \
    --mamba-backend flashinfer \
    --mamba-cache-mode align \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --speculative-config "{\"method\":\"dspark\",\"model\":\"$SPECULATIVE_MODEL_ID\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}" \
    --reasoning-parser nemotron_v3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice

echo
echo "Container started."
echo "Next: docker logs -f $CONTAINER_NAME"
echo "Ready check: curl -sf http://localhost:$PORT/health && echo OK"
