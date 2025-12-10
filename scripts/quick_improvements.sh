#!/bin/bash
# Quick Accuracy Improvement Script
# Run this to get immediate gains (35-45% top-1 accuracy)

set -e  # Exit on error

echo "======================================================================"
echo "MUSHROOM CLASSIFICATION - QUICK ACCURACY IMPROVEMENTS"
echo "======================================================================"
echo ""

WORK_DIR="/zfs/ai4good/student/hgupta"
MUSHROOM_DIR="$WORK_DIR/ai4good-mushroom/external/ecovision_mushroom"
VENV="$WORK_DIR/mushroom_env/bin/python3"

# Step 1: Train stronger ResNet18
echo "Step 1/3: Training ResNet18 for 20 epochs..."
echo "This will take 8-12 hours. Starting in background..."
cd "$MUSHROOM_DIR"

nohup $VENV src/train.py \
  experiment=base \
  model=resnet18 \
  logger=csv \
  trainer.max_epochs=20 \
  trainer.accelerator=cpu \
  trainer.precision=32 \
  > "$WORK_DIR/resnet18_20epochs.log" 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Monitor progress: tail -f $WORK_DIR/resnet18_20epochs.log"
echo ""
echo "Waiting for training to complete..."
echo "(You can Ctrl+C this script and training will continue in background)"
echo ""

# Wait for training or allow user to continue
read -p "Press Enter to skip waiting and continue with other improvements, or wait for training to finish: " -t 10 || true
echo ""

# Step 2: Evaluate with larger CLIP model
echo "Step 2/3: Creating evaluation script for larger CLIP model..."
cd "$WORK_DIR"

cat > eval_large_clip.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""Evaluate with larger CLIP model (ViT-L/14)"""

import torch
import clip
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os

# Dataset class
class MushroomDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, n_samples=-1):
        self.df = pd.read_csv(csv_path)
        if n_samples > 0:
            self.df = self.df.sample(n=min(n_samples, len(self.df)), random_state=42)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, row['label'], row['species']

# Load CLIP model
print("Loading CLIP ViT-L/14 model (this may take a minute)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
print(f"Model loaded on {device}")

# Load prompts
print("Loading prompts...")
with open("enhanced_with_common_names_prompts_clean.json", "r") as f:
    all_prompts = json.load(f)

# Get unique species
species_list = sorted(all_prompts.keys())
print(f"Found {len(species_list)} species")

# Encode all text prompts
print("Encoding text prompts...")
text_embeddings = {}
with torch.no_grad():
    for species in tqdm(species_list, desc="Encoding prompts"):
        prompts = all_prompts[species]
        if len(prompts) == 0:
            prompts = [f"a photo of {species}"]
        
        text_tokens = clip.tokenize(prompts).to(device)
        embeddings = model.encode_text(text_tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        text_embeddings[species] = embeddings

# Load test data
print("Loading test data...")
data_dir = "/zfs/ai4good/datasets/mushroom"
test_dataset = MushroomDataset(
    csv_path=os.path.join(data_dir, "test.csv"),
    data_dir=data_dir,
    transform=preprocess,
    n_samples=500  # Use 500 samples for faster evaluation
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Test samples: {len(test_dataset)}")

# Evaluate
print("Evaluating...")
all_preds = []
all_labels = []
all_top5 = []

temperature = 0.05

with torch.no_grad():
    for images, labels, species_names in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        
        # Encode images
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities for each class
        batch_size = images.shape[0]
        class_scores = torch.zeros(batch_size, len(species_list)).to(device)
        
        for class_idx, species in enumerate(species_list):
            text_embs = text_embeddings[species]
            
            # Compute similarities
            similarities = (image_features @ text_embs.T) * 100  # Scale by 100 like CLIP
            
            # Temperature-scaled pooling
            weights = torch.softmax(similarities / temperature, dim=1)
            pooled_scores = (weights * similarities).sum(dim=1)
            
            class_scores[:, class_idx] = pooled_scores
        
        # Get predictions
        probs = torch.softmax(class_scores, dim=1)
        top5_preds = torch.topk(probs, k=5, dim=1).indices
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_top5.extend(top5_preds.cpu().numpy())

# Convert to numpy
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_top5 = np.array(all_top5)

# Compute metrics
top1_acc = (all_preds == all_labels).mean()

top5_correct = np.any(all_top5 == all_labels.reshape(-1, 1), axis=1)
top5_acc = top5_correct.mean()

balanced_acc = balanced_accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

# Print results
print("\n" + "="*60)
print("CLIP ViT-L/14 Results (Enhanced Prompts, T=0.05)")
print("="*60)
print(f"Test Samples: {len(all_labels)}")
print(f"Top-1 Accuracy:       {top1_acc:.4f} ({top1_acc*100:.2f}%)")
print(f"Top-5 Accuracy:       {top5_acc:.4f} ({top5_acc*100:.2f}%)")
print(f"Balanced Accuracy:    {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
print(f"Macro F1-Score:       {macro_f1:.4f} ({macro_f1*100:.2f}%)")
print("="*60)

# Save results
results = {
    "model": "ViT-L/14",
    "prompts": "enhanced_with_common_names",
    "temperature": temperature,
    "test_samples": len(all_labels),
    "top1_accuracy": float(top1_acc),
    "top5_accuracy": float(top5_acc),
    "balanced_accuracy": float(balanced_acc),
    "macro_f1": float(macro_f1)
}

with open("clip_vitl14_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to clip_vitl14_results.json")
PYTHON_SCRIPT

chmod +x eval_large_clip.py
echo "Script created: eval_large_clip.py"
echo ""

# Run large CLIP evaluation
echo "Running evaluation with ViT-L/14 (this will take 30-60 minutes)..."
echo "Progress will be shown below..."
echo ""

$VENV eval_large_clip.py

echo ""
echo "Step 2 complete! Check clip_vitl14_results.json for results."
echo ""

# Step 3: Wait for training and run fusion
echo "Step 3/3: Waiting for ResNet training to complete..."
echo ""

if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "ResNet training still running (PID: $TRAIN_PID)"
    echo "You can:"
    echo "  1. Wait for it to finish: wait $TRAIN_PID"
    echo "  2. Check progress: tail -f $WORK_DIR/resnet18_20epochs.log"
    echo "  3. Continue this script later by running:"
    echo "     cd $WORK_DIR && ./run_fusion_after_training.sh"
    echo ""
    
    # Create follow-up script
    cat > run_fusion_after_training.sh << 'BASH_SCRIPT'
#!/bin/bash
# Run fusion after ResNet training completes

WORK_DIR="/zfs/ai4good/student/hgupta"
VENV="$WORK_DIR/mushroom_env/bin/python3"

cd "$WORK_DIR"

# Find latest checkpoint
LATEST_RUN=$(ls -td logs/mushroom/train/runs/* 2>/dev/null | head -n 1)
LATEST_CKPT=$(ls -t "$LATEST_RUN"/checkpoints/*.ckpt 2>/dev/null | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found!"
    exit 1
fi

echo "Using checkpoint: $LATEST_CKPT"
echo ""

# Run fusion
echo "Running CLIP + ResNet fusion..."
$VENV fusion_clip_resnet.py \
  --model resnet18 \
  --ckpt "$LATEST_CKPT" \
  --prompts enhanced_with_common_names_prompts_clean.json \
  --val_n 500 \
  --test_n 500 \
  --clip_temp 0.05 \
  --out fusion_results_strong.json

echo ""
echo "Fusion complete! Check fusion_results_strong.json"
echo ""

# Regenerate graphs
echo "Regenerating fusion graphs..."
$VENV create_fusion_graphs.py

echo ""
echo "======================================================================"
echo "ALL IMPROVEMENTS COMPLETE!"
echo "======================================================================"
echo ""
echo "Results:"
echo "  - CLIP ViT-L/14: clip_vitl14_results.json"
echo "  - ResNet18 (20 epochs): $LATEST_RUN"
echo "  - Fusion: fusion_results_strong.json"
echo "  - Graphs: fusion_*.png"
echo ""
echo "Expected improvement: 35-45% top-1 accuracy (from 25% baseline)"
echo ""
BASH_SCRIPT
    
    chmod +x run_fusion_after_training.sh
    echo "Created: run_fusion_after_training.sh"
    echo "Run it after training completes!"
    
else
    echo "Training appears to have completed. Running fusion now..."
    ./run_fusion_after_training.sh
fi

echo ""
echo "======================================================================"
echo "QUICK IMPROVEMENTS INITIATED!"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  ✓ ResNet18 training started (20 epochs, ~8-12 hours)"
echo "  ✓ CLIP ViT-L/14 evaluation completed"
echo "  ⏳ Fusion will run after training completes"
echo ""
echo "Next steps:"
echo "  1. Monitor training: tail -f resnet18_20epochs.log"
echo "  2. After training: ./run_fusion_after_training.sh"
echo "  3. For further improvements: see ACCURACY_IMPROVEMENT_STRATEGIES.md"
echo ""
