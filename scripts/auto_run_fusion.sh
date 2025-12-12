#!/bin/bash
# Auto-run fusion when both ResNet training and CLIP evaluation complete

WORK_DIR="/zfs/ai4good/student/hgupta"
VENV="$WORK_DIR/mushroom_env/bin/python3"

echo "======================================================================"
echo "WAITING FOR TRAINING AND EVALUATION TO COMPLETE"
echo "======================================================================"
echo ""

# Function to check if CLIP eval is done
check_clip_done() {
    if [ -f "$WORK_DIR/clip_ensemble_results.json" ]; then
        return 0  # Success
    fi
    return 1  # Not done
}

# Function to check if ResNet training is done
check_resnet_done() {
    # Check if at least one checkpoint exists from the latest run
    latest_run=$(ls -td "$WORK_DIR"/logs/mushroom/train/runs/* 2>/dev/null | head -n 1)
    if [ -n "$latest_run" ]; then
        checkpoints=$(ls "$latest_run"/checkpoints/*.ckpt 2>/dev/null | wc -l)
        if [ "$checkpoints" -gt 0 ]; then
            return 0  # Success
        fi
    fi
    return 1  # Not done
}

# Check current status
echo "Checking current status..."
echo ""

clip_done=false
resnet_done=false

if check_clip_done; then
    echo "✓ CLIP evaluation: COMPLETE"
    clip_done=true
else
    echo "⏳ CLIP evaluation: IN PROGRESS"
fi

if check_resnet_done; then
    echo "✓ ResNet training: COMPLETE"
    resnet_done=true
else
    echo "⏳ ResNet training: IN PROGRESS"
fi

echo ""

# If both are done, run fusion immediately
if $clip_done && $resnet_done; then
    echo "Both tasks complete! Running final fusion now..."
    cd "$WORK_DIR"
    $VENV final_fusion.py
    exit 0
fi

# Otherwise, wait for completion
echo "Waiting for tasks to complete..."
echo "This script will check every 5 minutes"
echo "Press Ctrl+C to cancel and run fusion manually later"
echo ""

wait_time=0
max_wait=720  # 12 hours maximum

while [ $wait_time -lt $max_wait ]; do
    sleep 300  # Wait 5 minutes
    wait_time=$((wait_time + 5))
    
    # Re-check status
    clip_done=false
    resnet_done=false
    
    if check_clip_done; then
        clip_done=true
    fi
    
    if check_resnet_done; then
        resnet_done=true
    fi
    
    echo "[$(date +%H:%M)] Status: CLIP=$(if $clip_done; then echo "✓"; else echo "⏳"; fi) ResNet=$(if $resnet_done; then echo "✓"; else echo "⏳"; fi) (waited ${wait_time} min)"
    
    # If both done, run fusion
    if $clip_done && $resnet_done; then
        echo ""
        echo "======================================================================"
        echo "BOTH TASKS COMPLETE! STARTING FINAL FUSION"
        echo "======================================================================"
        echo ""
        
        cd "$WORK_DIR"
        $VENV final_fusion.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "======================================================================"
            echo "FUSION COMPLETE!"
            echo "======================================================================"
            echo ""
            echo "Results saved to: final_fusion_results.json"
            echo ""
            echo "To view results:"
            echo "  cat final_fusion_results.json | python3 -m json.tool"
            echo ""
            
            # Generate updated graphs
            if [ -f "$WORK_DIR/create_fusion_graphs.py" ]; then
                echo "Generating updated visualizations..."
                $VENV create_fusion_graphs.py
                echo "Graphs updated!"
            fi
        else
            echo "ERROR: Fusion script failed. Check logs above."
            exit 1
        fi
        
        exit 0
    fi
done

echo ""
echo "Maximum wait time reached (12 hours)."
echo "Please check the status and run fusion manually:"
echo "  cd $WORK_DIR"
echo "  mushroom_env/bin/python3 final_fusion.py"
