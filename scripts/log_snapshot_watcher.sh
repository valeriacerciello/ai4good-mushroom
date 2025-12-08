#!/usr/bin/env bash
# Save 200-line snapshots of the fusion log every 60 seconds
LOG_FILE="logs/fusion_shots100_pecore_meta.log"
SNAP_DIR="logs/snapshots"
INTERVAL=60
mkdir -p "$SNAP_DIR"
while true; do
  if [ -f "$LOG_FILE" ]; then
    ts=$(date +"%Y%m%d_%H%M%S")
    tail -n 200 "$LOG_FILE" > "$SNAP_DIR/fusion_snapshot_${ts}.log"
  else
    ts=$(date +"%Y%m%d_%H%M%S")
    echo "No log file at $ts" > "$SNAP_DIR/fusion_snapshot_${ts}.log"
  fi
  sleep $INTERVAL
done
