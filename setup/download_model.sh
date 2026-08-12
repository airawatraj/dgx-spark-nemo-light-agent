#!/usr/bin/env bash
# setup/download_model.sh
# Downloads the Nemotron-3.5-Lightning-30B weights to local HF cache.
# Run this before docker/start.sh
set -euo pipefail

MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
SPECULATIVE_MODEL_ID="${SPECULATIVE_MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark}"

echo "=== Downloading Nemotron 3.5 Lightning 30B Weights ==="
echo "  Main Model:        $MODEL_ID"
echo "  Speculative Model: $SPECULATIVE_MODEL_ID"
echo

if ! command -v uv &>/dev/null; then
  echo "ERROR: uv is not installed. Please run: bash setup/install.sh"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. If the model requires licensing acceptance,"
  echo "         the download will fail. Run: export HF_TOKEN='your_api_token'"
  echo
fi

echo "Downloading main model..."
uvx hf download "$MODEL_ID"

echo "Downloading speculative model..."
uvx hf download "$SPECULATIVE_MODEL_ID"

echo ""
echo "✓ Download complete. Models cached in Hugging Face directory."
echo "Next: bash docker/start.sh"
