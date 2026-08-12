#!/usr/bin/env bash
# setup/install.sh
# Prerequisites for DGX Spark / Nemotron 3.5 Lightning production setup.
# Run once on a fresh DGX Spark before starting vLLM.
set -euo pipefail

echo "=== DGX Spark Setup ==="

# ── 1. Disable swap permanently ───────────────────────────────────────────────
echo "[1/3] Disabling swap..."
sudo swapoff -a 2>/dev/null || true

if [[ -f /etc/fstab ]]; then
  # Comment out active swap entries in fstab to survive reboots.
  sudo sed -ri '/^[[:space:]]*#/! s@^([[:space:]]*[^[:space:]#]+[[:space:]]+[^[:space:]]+[[:space:]]+swap[[:space:]].*)$@#\1@' /etc/fstab 2>/dev/null || true
  echo "    Swap disabled in /etc/fstab."
else
  echo "    No /etc/fstab found. Skipping swap persistence config."
fi

# ── 2. Verify Docker is running ───────────────────────────────────────────────
echo "[2/3] Checking Docker..."
if command -v docker &>/dev/null; then
  docker version --format 'Docker {{.Server.Version}}' || {
    echo "ERROR: Docker is installed but not running. Start Docker first."
    exit 1
  }
else
  echo "ERROR: Docker is not installed."
  exit 1
fi

# ── 3. Install uv ─────────────────────────────────────────────────────────────
echo "[3/3] Installing uv..."
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [[ -f "$HOME/.cargo/env" ]]; then
    source "$HOME/.cargo/env"
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo ""
echo "=== Setup complete ==="
echo "Next: export HF_TOKEN='your_token' (if gated) && bash setup/download_model.sh"
