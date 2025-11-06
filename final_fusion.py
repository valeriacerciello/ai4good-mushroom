#!/usr/bin/env python3
"""
Final fusion script combining:
- Best CLIP model (or ensemble)
- Trained ResNet18 (20 epochs)
- All prompts (19,141)
"""

import torch
import torch.nn.functional as F
import clip
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
import sys

# Add clip_utils to path
sys.path.append('/zfs/ai4good/student/hgupta')
from clip_utils import get_text_embeddings

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

def compute_clip_ensemble_probs(images, clip_models, text_embeddings, temperature=0.05):
    """Compute ensemble probabilities from multiple CLIP models"""
    ensemble_probs = None
    
    for model_name, (model, _) in clip_models.items():
        # Encode images
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute class scores
        batch_size = images.shape[0]
        num_classes = len(text_embeddings[model_name])
        class_scores = torch.zeros(batch_size, num_classes).to(images.device)
        
        for class_idx, species in enumerate(sorted(text_embeddings[model_name].keys())):
            text_embs = text_embeddings[model_name][species]
            
            # Compute similarities
            similarities = (image_features @ text_embs.T) * 100
            
            # Temperature-scaled pooling
            weights = torch.softmax(similarities / temperature, dim=1)
            pooled_scores = (weights * similarities).sum(dim=1)
            
            class_scores[:, class_idx] = pooled_scores
        
        # Get probabilities
        probs = torch.softmax(class_scores, dim=1)
        
        # Accumulate ensemble
        if ensemble_probs is None:
            ensemble_probs = probs
        else:
            ensemble_probs += probs
    
    # Average ensemble
    ensemble_probs = ensemble_probs / len(clip_models)
    return ensemble_probs

def load_resnet_model(checkpoint_path, num_classes=169):
    """Load trained ResNet model"""
    from torchvision import models
    import torch
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Clean keys (remove 'model.' prefix if present)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('model.', '').replace('net.', '')
        cleaned_state_dict[new_key] = value
    
    # Create model
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    
    return model

def compute_resnet_probs(images, resnet_model):
    """Compute probabilities from ResNet"""
    with torch.no_grad():
        logits = resnet_model(images)
        probs = F.softmax(logits, dim=1)
    return probs

def evaluate_fusion(val_loader, test_loader, clip_models, text_embeddings, 
                   resnet_model, temperature=0.05, device='cpu'):
    """Evaluate fusion with alpha tuning"""
    
    print("="*70)
    print("FINAL FUSION: CLIP Ensemble + ResNet18")
    print("="*70)
    print(f"CLIP models: {list(clip_models.keys())}")
    print(f"ResNet: ResNet18 (20 epochs)")
    print(f"Temperature: {temperature}")
    print("="*70)
    
    # Collect predictions on validation set
    print("\nCollecting validation predictions...")
    val_clip_probs = []
    val_resnet_probs = []
    val_labels = []
    
    for images, labels, _ in tqdm(val_loader, desc="Validation"):
        images = images.to(device)
        
        # CLIP ensemble probabilities
        clip_probs = compute_clip_ensemble_probs(images, clip_models, text_embeddings, temperature)
        
        # ResNet probabilities
        resnet_probs = compute_resnet_probs(images, resnet_model)
        
        val_clip_probs.append(clip_probs.cpu())
        val_resnet_probs.append(resnet_probs.cpu())
        val_labels.append(labels)
    
    val_clip_probs = torch.cat(val_clip_probs, dim=0)
    val_resnet_probs = torch.cat(val_resnet_probs, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    
    # Alpha sweep on validation
    print("\nPerforming alpha sweep on validation set...")
    alphas = [i/10 for i in range(11)]
    best_alpha = 0.0
    best_top1 = 0.0
    alpha_results = {}
    
    for alpha in alphas:
        # Fused probabilities
        fused_probs = alpha * val_resnet_probs + (1 - alpha) * val_clip_probs
        preds = torch.argmax(fused_probs, dim=1).numpy()
        labels_np = val_labels.numpy()
        
        # Compute metrics
        top1 = (preds == labels_np).mean()
        balanced = balanced_accuracy_score(labels_np, preds)
        macro_f1 = f1_score(labels_np, preds, average='macro', zero_division=0)
        
        alpha_results[alpha] = {
            'top1': float(top1),
            'balanced': float(balanced),
            'macro_f1': float(macro_f1)
        }
        
        print(f"  α={alpha:.1f}: Top-1={top1:.4f}, Balanced={balanced:.4f}, F1={macro_f1:.4f}")
        
        if top1 > best_top1:
            best_top1 = top1
            best_alpha = alpha
    
    print(f"\nBest α on validation: {best_alpha:.1f} (Top-1={best_top1:.4f})")
    
    # Evaluate on test set with best alpha
    print(f"\nEvaluating on test set with α={best_alpha:.1f}...")
    test_clip_probs = []
    test_resnet_probs = []
    test_labels = []
    test_top5_preds = []
    
    for images, labels, _ in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        
        # CLIP ensemble probabilities
        clip_probs = compute_clip_ensemble_probs(images, clip_models, text_embeddings, temperature)
        
        # ResNet probabilities
        resnet_probs = compute_resnet_probs(images, resnet_model)
        
        # Fused probabilities
        fused_probs = best_alpha * resnet_probs + (1 - best_alpha) * clip_probs
        
        # Top-5 predictions
        top5 = torch.topk(fused_probs, k=5, dim=1).indices
        
        test_clip_probs.append(clip_probs.cpu())
        test_resnet_probs.append(resnet_probs.cpu())
        test_labels.append(labels)
        test_top5_preds.append(top5.cpu())
    
    test_clip_probs = torch.cat(test_clip_probs, dim=0)
    test_resnet_probs = torch.cat(test_resnet_probs, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    test_top5_preds = torch.cat(test_top5_preds, dim=0)
    
    # Compute final test metrics
    test_fused_probs = best_alpha * test_resnet_probs + (1 - best_alpha) * test_clip_probs
    test_preds = torch.argmax(test_fused_probs, dim=1).numpy()
    test_labels_np = test_labels.numpy()
    
    test_top1 = (test_preds == test_labels_np).mean()
    test_top5_correct = torch.any(test_top5_preds == test_labels.unsqueeze(1), dim=1).numpy()
    test_top5 = test_top5_correct.mean()
    test_balanced = balanced_accuracy_score(test_labels_np, test_preds)
    test_macro_f1 = f1_score(test_labels_np, test_preds, average='macro', zero_division=0)
    
    # Also get CLIP-only and ResNet-only metrics for comparison
    clip_preds = torch.argmax(test_clip_probs, dim=1).numpy()
    resnet_preds = torch.argmax(test_resnet_probs, dim=1).numpy()
    
    clip_top1 = (clip_preds == test_labels_np).mean()
    resnet_top1 = (resnet_preds == test_labels_np).mean()
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Test Samples: {len(test_labels_np)}")
    print()
    print(f"CLIP Ensemble (α=0.0):")
    print(f"  Top-1 Accuracy:    {clip_top1:.4f} ({clip_top1*100:.2f}%)")
    print()
    print(f"ResNet18 (α=1.0):")
    print(f"  Top-1 Accuracy:    {resnet_top1:.4f} ({resnet_top1*100:.2f}%)")
    print()
    print(f"Fusion (α={best_alpha:.1f}):")
    print(f"  Top-1 Accuracy:    {test_top1:.4f} ({test_top1*100:.2f}%)")
    print(f"  Top-5 Accuracy:    {test_top5:.4f} ({test_top5*100:.2f}%)")
    print(f"  Balanced Accuracy: {test_balanced:.4f} ({test_balanced*100:.2f}%)")
    print(f"  Macro F1-Score:    {test_macro_f1:.4f} ({test_macro_f1*100:.2f}%)")
    print()
    print(f"Improvement over CLIP: {(test_top1 - clip_top1)*100:+.2f}%")
    print(f"Improvement over ResNet: {(test_top1 - resnet_top1)*100:+.2f}%")
    print("="*70)
    
    return {
        'best_alpha': float(best_alpha),
        'alpha_sweep': alpha_results,
        'test': {
            'clip_only': {'top1': float(clip_top1)},
            'resnet_only': {'top1': float(resnet_top1)},
            'fusion': {
                'top1': float(test_top1),
                'top5': float(test_top5),
                'balanced': float(test_balanced),
                'macro_f1': float(test_macro_f1)
            }
        }
    }

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    prompts_file = "enhanced_with_common_names_prompts_clean.json"
    temperature = 0.05
    val_samples = 1000
    test_samples = 1000
    
    # Find latest ResNet checkpoint
    import glob
    checkpoint_pattern = "/zfs/ai4good/student/hgupta/logs/mushroom/train/runs/*/checkpoints/*.ckpt"
    checkpoints = sorted(glob.glob(checkpoint_pattern), key=os.path.getmtime, reverse=True)
    
    if not checkpoints:
        print("ERROR: No ResNet checkpoint found!")
        print(f"Searched in: {checkpoint_pattern}")
        sys.exit(1)
    
    resnet_checkpoint = checkpoints[0]
    print(f"Using ResNet checkpoint: {resnet_checkpoint}")
    
    # Load CLIP models
    print("\nLoading CLIP models...")
    clip_models = {}
    for model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14"]:
        result_file = f"clip_{model_name.replace('/', '_')}_all_prompts.json"
        if os.path.exists(result_file):
            print(f"  Loading {model_name}...")
            model, preprocess = clip.load(model_name, device=device)
            clip_models[model_name] = (model, preprocess)
    
    if not clip_models:
        print("ERROR: No CLIP model results found!")
        print("Please run eval_all_clip_models.py first")
        sys.exit(1)
    
    print(f"Loaded {len(clip_models)} CLIP models: {list(clip_models.keys())}")
    
    # Load prompts and encode text
    print(f"\nLoading prompts from {prompts_file}...")
    with open(prompts_file, 'r') as f:
        all_prompts = json.load(f)
    
    species_list = sorted(all_prompts.keys())
    print(f"Species: {len(species_list)}")
    
    # Encode text for each CLIP model
    print("\nEncoding text prompts for each CLIP model...")
    text_embeddings = {}
    
    for model_name, (model, _) in clip_models.items():
        print(f"  Encoding for {model_name}...")
        text_embeddings[model_name] = {}
        
        for species in tqdm(species_list, desc=f"  {model_name}"):
            prompts = all_prompts[species]
            if len(prompts) == 0:
                prompts = [f"a photo of {species}"]
            
            # Batch encode
            batch_size = 64
            embeddings_list = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                text_tokens = clip.tokenize(batch_prompts, truncate=True).to(device)
                with torch.no_grad():
                    batch_embeddings = model.encode_text(text_tokens)
                    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                embeddings_list.append(batch_embeddings)
            
            embeddings = torch.cat(embeddings_list, dim=0)
            text_embeddings[model_name][species] = embeddings
    
    # Load ResNet
    print(f"\nLoading ResNet from {resnet_checkpoint}...")
    resnet_model = load_resnet_model(resnet_checkpoint, num_classes=len(species_list))
    resnet_model = resnet_model.to(device)
    print("ResNet loaded successfully")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    data_dir = "/zfs/ai4good/datasets/mushroom"
    
    # Use first CLIP model's preprocess (they should be similar)
    preprocess = list(clip_models.values())[0][1]
    
    val_dataset = MushroomDataset(
        csv_path=os.path.join(data_dir, "val.csv"),
        data_dir=data_dir,
        transform=preprocess,
        n_samples=val_samples
    )
    
    test_dataset = MushroomDataset(
        csv_path=os.path.join(data_dir, "test.csv"),
        data_dir=data_dir,
        transform=preprocess,
        n_samples=test_samples
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Run fusion
    results = evaluate_fusion(
        val_loader, test_loader,
        clip_models, text_embeddings,
        resnet_model,
        temperature=temperature,
        device=device
    )
    
    # Save results
    output_file = "final_fusion_results.json"
    results['config'] = {
        'clip_models': list(clip_models.keys()),
        'resnet_checkpoint': resnet_checkpoint,
        'prompts_file': prompts_file,
        'temperature': temperature,
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset)
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
