#!/usr/bin/env bash
# docker/stop.sh
# Stops and removes the spark-brain container.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping $CONTAINER_NAME..."
  docker stop "$CONTAINER_NAME"
  docker rm "$CONTAINER_NAME"
  echo "✓ $CONTAINER_NAME stopped and removed."
else
  echo "$CONTAINER_NAME is not running."
  docker rm "$CONTAINER_NAME" 2>/dev/null && echo "Removed stopped container." || true
fi
