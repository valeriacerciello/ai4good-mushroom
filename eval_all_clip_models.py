#!/usr/bin/env python3
"""Evaluate with larger CLIP model (ViT-L/14) using all prompts"""

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

def evaluate_clip_model(model_name, prompts_file, n_test_samples=500, temperature=0.05):
    """Evaluate a CLIP model variant"""
    
    print(f"\n{'='*70}")
    print(f"Evaluating CLIP {model_name}")
    print(f"{'='*70}")
    
    # Load CLIP model
    print(f"Loading CLIP {model_name} model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    print(f"Model loaded on {device}")

    # Load prompts
    print(f"Loading prompts from {prompts_file}...")
    with open(prompts_file, "r") as f:
        all_prompts = json.load(f)

    # Get unique species
    species_list = sorted(all_prompts.keys())
    print(f"Found {len(species_list)} species")
    
    # Count total prompts
    total_prompts = sum(len(prompts) for prompts in all_prompts.values())
    print(f"Total prompts: {total_prompts}")

    # Encode all text prompts
    print("Encoding text prompts...")
    text_embeddings = {}
    with torch.no_grad():
        for species in tqdm(species_list, desc="Encoding prompts"):
            prompts = all_prompts[species]
            if len(prompts) == 0:
                prompts = [f"a photo of {species}"]
            
            # Handle long prompt lists in batches
            batch_size = 64
            embeddings_list = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                text_tokens = clip.tokenize(batch_prompts, truncate=True).to(device)
                batch_embeddings = model.encode_text(text_tokens)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                embeddings_list.append(batch_embeddings)
            
            embeddings = torch.cat(embeddings_list, dim=0)
            text_embeddings[species] = embeddings

    # Load test data
    print("Loading test data...")
    data_dir = "/zfs/ai4good/datasets/mushroom"
    test_dataset = MushroomDataset(
        csv_path=os.path.join(data_dir, "test.csv"),
        data_dir=data_dir,
        transform=preprocess,
        n_samples=n_test_samples
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Test samples: {len(test_dataset)}")

    # Evaluate
    print(f"Evaluating with temperature={temperature}...")
    all_preds = []
    all_labels = []
    all_top5 = []

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
    print("\n" + "="*70)
    print(f"CLIP {model_name} Results")
    print(f"Prompts: {prompts_file} ({total_prompts} prompts)")
    print(f"Temperature: {temperature}")
    print("="*70)
    print(f"Test Samples:         {len(all_labels)}")
    print(f"Top-1 Accuracy:       {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"Top-5 Accuracy:       {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"Balanced Accuracy:    {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"Macro F1-Score:       {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print("="*70 + "\n")

    # Return results
    return {
        "model": model_name,
        "prompts_file": prompts_file,
        "total_prompts": total_prompts,
        "temperature": temperature,
        "test_samples": len(all_labels),
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist()
    }

if __name__ == "__main__":
    # Configuration
    models = ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
    prompts_file = "enhanced_with_common_names_prompts_clean.json"
    n_test_samples = 500
    temperature = 0.05
    
    # Store all results
    all_results = {}
    predictions_for_ensemble = {}
    
    # Evaluate each model
    for model_name in models:
        print(f"\n{'#'*70}")
        print(f"# Evaluating {model_name}")
        print(f"{'#'*70}\n")
        
        try:
            result = evaluate_clip_model(model_name, prompts_file, n_test_samples, temperature)
            all_results[model_name] = result
            predictions_for_ensemble[model_name] = {
                "predictions": result["predictions"],
                "labels": result["labels"]
            }
            
            # Save individual result
            output_file = f"clip_{model_name.replace('/', '_')}_all_prompts.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"ERROR evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Ensemble predictions
    if len(predictions_for_ensemble) > 1:
        print(f"\n{'='*70}")
        print("ENSEMBLE OF ALL MODELS")
        print("="*70)
        
        # Get labels (same for all models)
        labels = np.array(predictions_for_ensemble[models[0]]["labels"])
        
        # Collect all predictions
        all_model_preds = []
        for model_name in predictions_for_ensemble.keys():
            preds = np.array(predictions_for_ensemble[model_name]["predictions"])
            all_model_preds.append(preds)
        
        # Voting ensemble (majority vote)
        all_model_preds = np.array(all_model_preds)  # Shape: (n_models, n_samples)
        
        # Majority voting
        ensemble_preds = []
        for i in range(all_model_preds.shape[1]):
            votes = all_model_preds[:, i]
            # Get most common prediction
            unique, counts = np.unique(votes, return_counts=True)
            ensemble_preds.append(unique[np.argmax(counts)])
        
        ensemble_preds = np.array(ensemble_preds)
        
        # Compute ensemble metrics
        ensemble_top1 = (ensemble_preds == labels).mean()
        ensemble_balanced = balanced_accuracy_score(labels, ensemble_preds)
        ensemble_f1 = f1_score(labels, ensemble_preds, average='macro', zero_division=0)
        
        print(f"Ensemble Strategy:    Majority Voting")
        print(f"Models:               {', '.join(predictions_for_ensemble.keys())}")
        print(f"Top-1 Accuracy:       {ensemble_top1:.4f} ({ensemble_top1*100:.2f}%)")
        print(f"Balanced Accuracy:    {ensemble_balanced:.4f} ({ensemble_balanced*100:.2f}%)")
        print(f"Macro F1-Score:       {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)")
        print("="*70 + "\n")
        
        # Save ensemble results
        ensemble_results = {
            "strategy": "majority_voting",
            "models": list(predictions_for_ensemble.keys()),
            "top1_accuracy": float(ensemble_top1),
            "balanced_accuracy": float(ensemble_balanced),
            "macro_f1": float(ensemble_f1),
            "test_samples": len(labels)
        }
        
        with open("clip_ensemble_results.json", "w") as f:
            json.dump(ensemble_results, f, indent=2)
        print("Ensemble results saved to clip_ensemble_results.json")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Model':<20} {'Top-1':<10} {'Top-5':<10} {'Balanced':<10} {'Macro-F1':<10}")
    print("-"*70)
    
    for model_name, result in all_results.items():
        print(f"{model_name:<20} {result['top1_accuracy']*100:>6.2f}%   "
              f"{result['top5_accuracy']*100:>6.2f}%   "
              f"{result['balanced_accuracy']*100:>6.2f}%   "
              f"{result['macro_f1']*100:>6.2f}%")
    
    if len(predictions_for_ensemble) > 1:
        print("-"*70)
        print(f"{'Ensemble (Voting)':<20} {ensemble_top1*100:>6.2f}%   "
              f"{'N/A':<10} {ensemble_balanced*100:>6.2f}%   "
              f"{ensemble_f1*100:>6.2f}%")
    
    print("="*70)
    
    print("\nAll evaluations complete!")
