#!/bin/bash
# Monitor all training and evaluation tasks

WORK_DIR="/zfs/ai4good/student/hgupta"

echo "======================================================================"
echo "MUSHROOM CLASSIFICATION - TRAINING STATUS MONITOR"
echo "======================================================================"
echo "Current time: $(date)"
echo ""

# Check tmux sessions
echo "=== ACTIVE TMUX SESSIONS ==="
tmux list-sessions 2>/dev/null || echo "No active tmux sessions"
echo ""

# Check ResNet training
echo "=== RESNET TRAINING STATUS ==="
if tmux has-session -t resnet_training 2>/dev/null; then
    echo "✓ ResNet training session is ACTIVE"
    echo ""
    echo "Latest log output (last 20 lines):"
    echo "-------------------------------------------------------------------"
    tail -n 20 "$WORK_DIR/resnet18_20epochs.log" 2>/dev/null || echo "Log file not yet created"
    echo "-------------------------------------------------------------------"
    echo ""
    echo "To attach to this session: tmux attach-session -t resnet_training"
    echo "To view full log: tail -f $WORK_DIR/resnet18_20epochs.log"
else
    echo "✗ ResNet training session is NOT running"
    if [ -f "$WORK_DIR/resnet18_20epochs.log" ]; then
        echo "Training may have completed or crashed. Check log:"
        echo "tail -n 50 $WORK_DIR/resnet18_20epochs.log"
    fi
fi
echo ""

# Check CLIP evaluation
echo "=== CLIP EVALUATION STATUS ==="
if tmux has-session -t clip_eval 2>/dev/null; then
    echo "✓ CLIP evaluation session is ACTIVE"
    echo ""
    echo "Latest log output (last 20 lines):"
    echo "-------------------------------------------------------------------"
    tail -n 20 "$WORK_DIR/clip_all_models_evaluation.log" 2>/dev/null || echo "Log file not yet created"
    echo "-------------------------------------------------------------------"
    echo ""
    echo "To attach to this session: tmux attach-session -t clip_eval"
    echo "To view full log: tail -f $WORK_DIR/clip_all_models_evaluation.log"
else
    echo "✗ CLIP evaluation session is NOT running"
    if [ -f "$WORK_DIR/clip_all_models_evaluation.log" ]; then
        echo "Evaluation may have completed or crashed. Check log:"
        echo "tail -n 50 $WORK_DIR/clip_all_models_evaluation.log"
    fi
fi
echo ""

# Check for completed results
echo "=== COMPLETED RESULTS ==="
if [ -f "$WORK_DIR/clip_ensemble_results.json" ]; then
    echo "✓ CLIP ensemble results available"
    python3 -c "
import json
with open('$WORK_DIR/clip_ensemble_results.json') as f:
    data = json.load(f)
    print(f\"  Top-1 Accuracy: {data['top1_accuracy']*100:.2f}%\")
    print(f\"  Models: {', '.join(data['models'])}\")
" 2>/dev/null || echo "  (Could not parse results)"
else
    echo "⏳ CLIP ensemble results not yet available"
fi
echo ""

# Check for individual CLIP results
for model_file in "$WORK_DIR"/clip_ViT-*.json; do
    if [ -f "$model_file" ]; then
        basename_file=$(basename "$model_file")
        echo "✓ $(basename "$model_file")"
        python3 -c "
import json
with open('$model_file') as f:
    data = json.load(f)
    print(f\"  Top-1: {data['top1_accuracy']*100:.2f}%  Top-5: {data['top5_accuracy']*100:.2f}%\")
" 2>/dev/null || echo "  (Could not parse)"
    fi
done
echo ""

# Check for ResNet checkpoint
echo "=== RESNET CHECKPOINTS ==="
latest_run=$(ls -td "$WORK_DIR"/logs/mushroom/train/runs/* 2>/dev/null | head -n 1)
if [ -n "$latest_run" ]; then
    echo "Latest training run: $latest_run"
    checkpoints=$(ls "$latest_run"/checkpoints/*.ckpt 2>/dev/null | wc -l)
    echo "Checkpoints found: $checkpoints"
    if [ $checkpoints -gt 0 ]; then
        echo "Latest checkpoint:"
        ls -lh "$latest_run"/checkpoints/*.ckpt | tail -n 1
    fi
else
    echo "No training runs found yet"
fi
echo ""

# System resource usage
echo "=== SYSTEM RESOURCES ==="
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "  Usage: " 100 - $1 "%"}'
echo "Memory:"
free -h | grep Mem | awk '{print "  Used: " $3 " / Total: " $2 " (" $3/$2*100 "%)"}'
echo ""

# Process check
echo "=== PYTHON PROCESSES ==="
ps aux | grep "[p]ython3.*train.py\|[p]ython3.*eval_all" | awk '{print $2, $11, $12, $13, $14, $15}' | head -n 5
echo ""

echo "======================================================================"
echo "QUICK COMMANDS"
echo "======================================================================"
echo "Attach to ResNet training:  tmux attach-session -t resnet_training"
echo "Attach to CLIP evaluation:  tmux attach-session -t clip_eval"
echo "View ResNet log:            tail -f $WORK_DIR/resnet18_20epochs.log"
echo "View CLIP log:              tail -f $WORK_DIR/clip_all_models_evaluation.log"
echo "Refresh this monitor:       $0"
echo "List all sessions:          tmux list-sessions"
echo "Kill a session:             tmux kill-session -t SESSION_NAME"
echo "======================================================================"
