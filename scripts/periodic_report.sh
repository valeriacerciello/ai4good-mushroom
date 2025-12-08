#!/usr/bin/env bash
# Periodically append brief status and latest log snapshot to logs/periodic_reports.log
LOG_FILE="logs/fusion_shots100_pecore_meta.log"
SNAP_DIR="logs/snapshots"
OUT_LOG="logs/periodic_reports.log"
PID_FILE="logs/fusion_shots100_pecore_meta.pid"
INTERVAL=${1:-60}
mkdir -p $(dirname "$OUT_LOG")
while true; do
  ts=$(date +"%Y-%m-%d %H:%M:%S")
  echo "== Report: $ts ==" >> "$OUT_LOG"
  if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    if ps -p $pid > /dev/null; then
      echo "Process PID:$pid running" >> "$OUT_LOG"
    else
      echo "Process PID:$pid not-running" >> "$OUT_LOG"
    fi
  else
    echo "No PID file" >> "$OUT_LOG"
  fi
  echo "-- last modified log info --" >> "$OUT_LOG"
  if [ -f "$LOG_FILE" ]; then
    ls -lh "$LOG_FILE" | sed -n '1p' >> "$OUT_LOG"
  else
    echo "No main log file yet" >> "$OUT_LOG"
  fi
  echo "-- latest snapshot (last 40 lines) --" >> "$OUT_LOG"
  latest=$(ls -1t "$SNAP_DIR" 2>/dev/null | head -n1 || true)
  if [ -n "$latest" ]; then
    tail -n 40 "$SNAP_DIR/$latest" >> "$OUT_LOG" 2>/dev/null || true
  else
    tail -n 40 "$LOG_FILE" >> "$OUT_LOG" 2>/dev/null || true
  fi
  echo "" >> "$OUT_LOG"
  sleep $INTERVAL
done
