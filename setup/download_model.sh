#!/usr/bin/env bash
# setup/download_model.sh
# Downloads the Nemotron-3.5-Lightning-30B weights to local HF cache.
# Run this before docker/start.sh
set -euo pipefail

MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
SPECULATIVE_MODEL_ID="${SPECULATIVE_MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark}"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_CACHE_DIR="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
SPEC_CACHE_DIR="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"

echo "=== Downloading Nemotron 3.5 Lightning 30B Weights ==="
echo "  Main Model:        $MODEL_ID"
echo "  Speculative Model: $SPECULATIVE_MODEL_ID"
echo

# ── Check Base Model Cache ───────────────────────────────────────────────────
BASE_CACHED=false
if [[ -d "$MODEL_CACHE_DIR/snapshots" ]]; then
  SNAP_COUNT=$(find "$MODEL_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  if [[ "$SNAP_COUNT" -gt 0 ]]; then
    echo "✓ Base model already present at $MODEL_CACHE_DIR"
    BASE_CACHED=true
  fi
fi

# ── Check Speculative Model Cache ────────────────────────────────────────────
SPEC_CACHED=false
if [[ -d "$SPEC_CACHE_DIR/snapshots" ]]; then
  SPEC_SNAP_COUNT=$(find "$SPEC_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  if [[ "$SPEC_SNAP_COUNT" -gt 0 ]]; then
    echo "✓ Speculative draft model already present at $SPEC_CACHE_DIR"
    SPEC_CACHED=true
  fi
fi

if [[ "$BASE_CACHED" == "true" && "$SPEC_CACHED" == "true" ]]; then
  echo "  All weights present. Skipping download."
  echo ""
  echo "Next: bash docker/start.sh"
  exit 0
fi

if ! command -v uv &>/dev/null; then
  echo "ERROR: uv is not installed. Please run: bash setup/install.sh"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. If the model requires licensing acceptance,"
  echo "         the download will fail. Run: export HF_TOKEN='your_api_token'"
  echo
fi

if [[ "$BASE_CACHED" != "true" ]]; then
  echo "Downloading main model..."
  uvx hf download "$MODEL_ID"
fi

if [[ "$SPEC_CACHED" != "true" ]]; then
  echo "Downloading speculative model..."
  uvx hf download "$SPECULATIVE_MODEL_ID"
fi

echo ""
echo "✓ Download complete. Models cached in Hugging Face directory."
echo "Next: bash docker/start.sh"
