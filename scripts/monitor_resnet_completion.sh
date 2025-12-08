#!/bin/bash
# Monitor a ResNet training PID and notify when training completes by
# writing a small notification file with the latest checkpoint path.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <PID> [poll_seconds]"
  exit 1
fi

PID="$1"
POLL_SECONDS="${2:-60}"
WORK_DIR="/zfs/ai4good/student/hgupta"
LOG_FILE="$WORK_DIR/monitor_resnet.log"
NOTIFY_FILE="$WORK_DIR/RESNET_TRAINING_DONE.txt"
CHECK_RUNS_DIR="$WORK_DIR/logs/mushroom/train/runs"

echo "$(date -u --iso-8601=seconds) Starting monitor for PID=$PID (poll=${POLL_SECONDS}s)" >> "$LOG_FILE"

while kill -0 "$PID" 2>/dev/null; do
  echo "$(date -u --iso-8601=seconds) PID $PID still running" >> "$LOG_FILE"
  sleep "$POLL_SECONDS"
done

echo "$(date -u --iso-8601=seconds) PID $PID no longer running" >> "$LOG_FILE"

# Find latest run folder and checkpoint
LATEST_RUN=""
if compgen -G "$CHECK_RUNS_DIR/*" > /dev/null; then
  LATEST_RUN=$(ls -td "$CHECK_RUNS_DIR"/* 2>/dev/null | head -n 1 || true)
fi

if [ -n "$LATEST_RUN" ] && compgen -G "$LATEST_RUN/checkpoints/*.ckpt" > /dev/null; then
  LATEST_CKPT=$(ls -t "$LATEST_RUN"/checkpoints/*.ckpt 2>/dev/null | head -n 1 || true)
  if [ -n "$LATEST_CKPT" ]; then
    echo "$(date -u --iso-8601=seconds) Found checkpoint: $LATEST_CKPT" >> "$LOG_FILE"
    echo "DONE: $LATEST_CKPT" > "$NOTIFY_FILE"
    echo "Checkpoint written to $NOTIFY_FILE"
    exit 0
  fi
fi

echo "$(date -u --iso-8601=seconds) Training ended but no checkpoint found under $CHECK_RUNS_DIR" >> "$LOG_FILE"
echo "ENDED_NO_CKPT" > "$NOTIFY_FILE"
echo "Wrote $NOTIFY_FILE (no checkpoint found)"
exit 0
