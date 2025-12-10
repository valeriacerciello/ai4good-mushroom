#!/bin/bash
# Periodically append concise progress snapshots from the fusion log.
LOG="/zfs/ai4good/student/hgupta/logs/fusion_shots100_pecore_meta_fresh.log"
OUT="/zfs/ai4good/student/hgupta/logs/periodic_reports_short.log"

# Ensure output dir exists
mkdir -p "$(dirname "$OUT")"

while true; do
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  # Last LinearProbe progress line
  last_epoch_line=$(grep "LinearProbe Epoch" "$LOG" 2>/dev/null | tail -n 1)
  # Last Best alpha/printer line
  last_best_line=$(grep "Best α on validation" "$LOG" 2>/dev/null | tail -n 1)
  # Last 30 lines snapshot
  snapshot=$(tail -n 30 "$LOG" 2>/dev/null)

  echo "[$ts] Snapshot from fusion run:" >> "$OUT"
  if [ -n "$last_epoch_line" ]; then
    echo "  Progress: $last_epoch_line" >> "$OUT"
  else
    echo "  Progress: (no LinearProbe progress found yet)" >> "$OUT"
  fi
  if [ -n "$last_best_line" ]; then
    echo "  $last_best_line" >> "$OUT"
  fi
  echo "--- last 30 lines ---" >> "$OUT"
  echo "$snapshot" >> "$OUT"
  echo "--- end ---" >> "$OUT"
  echo "" >> "$OUT"

  sleep 60
done
