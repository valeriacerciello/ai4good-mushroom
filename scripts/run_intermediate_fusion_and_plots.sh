#!/usr/bin/env bash
set -euo pipefail
cd /zfs/ai4good/student/hgupta
# Use the project's virtualenv python so torch and other deps are available
PYTHON="/zfs/ai4good/student/hgupta/mushroom_env/bin/python3"
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: python not found at $PYTHON" >&2
  exit 1
fi

CHECKPOINT="./logs/mushroom/train/runs/2025-10-22_11-03-32/checkpoints/last.ckpt"
PROMPTS="delta_prompts.json"
CLIP_TEMP=0.05
OUT_JSON="fusion_results_intermediate.json"
PLOTS_DIR="plots"
mkdir -p "$PLOTS_DIR"

# Run fusion (this script accepts --ckpt and --prompts)
"$PYTHON" fusion_clip_resnet.py \
  --ckpt "$CHECKPOINT" \
  --prompts "$PROMPTS" \
  --clip_temp $CLIP_TEMP \
  --out "$OUT_JSON"

# Generate plots (create_fusion_graphs.py expects fusion_results.json; copy)
cp "$OUT_JSON" fusion_results.json
"$PYTHON" create_fusion_graphs.py --input_json fusion_results.json --out_dir "$PLOTS_DIR"

# copy plots to plots/ (they should already be generated there but ensure)
cp -v fusion_*png "$PLOTS_DIR"/ || true

echo "Intermediate fusion + plots complete. Results: $OUT_JSON, plots in $PLOTS_DIR" > /zfs/ai4good/student/hgupta/INTERMEDIATE_FUSION_DONE.txt
