#!/usr/bin/env bash
# docker/start.sh
# Starts the spark-brain container with Nemotron 3.5 Lightning 30B and DSpark.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
SPECULATIVE_MODEL_ID="${SPECULATIVE_MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
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

# ── Patch DSpark model config if needed ─────────────────────────────────────────
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SPEC_CACHE="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"

if [ -d "$SPEC_CACHE" ]; then
  echo "Checking DSpark model configs for rank mismatches..."
  SPEC_CACHE="$SPEC_CACHE" python3 << 'EOF'
import glob, json, os
spec_cache = os.environ.get('SPEC_CACHE', '')
for config_path in glob.glob(os.path.join(spec_cache, '**/config.json'), recursive=True):
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        updated = False
        if cfg.get('markov_rank') != 256:
            print(f"Setting markov_rank from {cfg.get('markov_rank')} to 256 in {config_path}")
            cfg['markov_rank'] = 256
            updated = True
        if cfg.get('dspark_markov_rank') != 512:
            print(f"Setting dspark_markov_rank from {cfg.get('dspark_markov_rank')} to 512 in {config_path}")
            cfg['dspark_markov_rank'] = 512
            updated = True
        if updated:
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            print(f"Successfully patched {config_path}")
    except Exception as e:
        print(f"Error patching {config_path}: {e}")
EOF
fi

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
