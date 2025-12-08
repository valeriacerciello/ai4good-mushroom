#!/usr/bin/env python3
"""
Final fusion script combining:
- Best CLIP model (or ensemble)
- Trained ResNet18 (20 epochs)
- All prompts (19,141)
"""

import argparse
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
from collections import defaultdict

# Add clip_utils to path
sys.path.append('/zfs/ai4good/student/hgupta')
from clip_utils import get_text_embeddings

class MushroomDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, n_samples=-1, species_to_idx=None):
        self.df = pd.read_csv(csv_path)
        if n_samples > 0:
            self.df = self.df.sample(n=min(n_samples, len(self.df)), random_state=42)
        self.data_dir = data_dir
        self.transform = transform
        self.species_to_idx = species_to_idx
        # detect image path column
        if 'image_path' in self.df.columns:
            self._img_col = 'image_path'
        elif 'path' in self.df.columns:
            self._img_col = 'path'
        else:
            # fallback to first column
            self._img_col = self.df.columns[0]
        # label column usually 'label' or 'species'
        if 'label' in self.df.columns:
            self._label_col = 'label'
        elif 'species' in self.df.columns:
            self._label_col = 'species'
        else:
            # fallback to second column if present
            self._label_col = self.df.columns[1] if len(self.df.columns) > 1 else self._img_col
        # create resolved absolute paths and filter missing files early to avoid DataLoader worker crashes
        def _resolve(p):
            if pd.isna(p):
                return ''
            p = str(p)
            return p if os.path.isabs(p) else os.path.join(self.data_dir, p)

        self.df['__resolved_path'] = self.df[self._img_col].apply(_resolve)
        exists_mask = self.df['__resolved_path'].apply(lambda p: os.path.exists(p))
        if exists_mask.sum() < len(self.df):
            print(f"Warning: {len(self.df) - int(exists_mask.sum())} missing images in {csv_path}; filtering them out.")
        self.df = self.df[exists_mask].reset_index(drop=True)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.get('__resolved_path') or (os.path.join(self.data_dir, row[self._img_col]))
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        species_name = row[self._label_col]
        if self.species_to_idx is not None:
            label_idx = int(self.species_to_idx.get(species_name, -1))
        else:
            # try to coerce to int, otherwise -1
            try:
                label_idx = int(species_name)
            except Exception:
                label_idx = -1

        return image, label_idx, species_name

    def __len__(self):
        return len(self.df)


class FewShotDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, shots=100, species_list=None, species_to_idx=None):
        df = pd.read_csv(csv_path)
        # detect which column contains species/label names
        if 'species' in df.columns:
            label_col = 'species'
        elif 'label' in df.columns:
            label_col = 'label'
        else:
            label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        # compute resolved image paths and filter missing images before sampling
        def _resolve(p):
            if pd.isna(p):
                return ''
            p = str(p)
            return p if os.path.isabs(p) else os.path.join(data_dir, p)

        df['__resolved_path'] = df[df.columns[0]].apply(_resolve) if 'image_path' not in df.columns and 'path' not in df.columns else df.get('image_path', df.get('path')).apply(_resolve)
        exists_mask_all = df['__resolved_path'].apply(lambda p: os.path.exists(p))
        if exists_mask_all.sum() < len(df):
            print(f"Warning: {len(df) - int(exists_mask_all.sum())} missing images in {csv_path}; they will be ignored for few-shot sampling.")
        df = df[exists_mask_all].reset_index(drop=True)

        # ensure consistent species ordering
        if species_list is None:
            species_list = sorted(df[label_col].unique())

        rows = []
        for sp in species_list:
            sp_rows = df[df[label_col] == sp]
            if len(sp_rows) == 0:
                continue
            if shots <= 0:
                sampled = sp_rows
            else:
                sampled = sp_rows.sample(n=min(shots, len(sp_rows)), random_state=42)
            rows.append(sampled)

        if len(rows) == 0:
            self.df = pd.DataFrame(columns=df.columns)
        else:
            self.df = pd.concat(rows).reset_index(drop=True)

        self.data_dir = data_dir
        self.transform = transform
        self.species_to_idx = species_to_idx
        # detect image path column
        if 'image_path' in self.df.columns:
            self._img_col = 'image_path'
        elif 'path' in self.df.columns:
            self._img_col = 'path'
        else:
            self._img_col = self.df.columns[0]
        # label column as detected earlier
        self._label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row[self._img_col])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        species_name = row[self._label_col]
        if self.species_to_idx is not None:
            label_idx = int(self.species_to_idx.get(species_name, -1))
        else:
            try:
                label_idx = int(species_name)
            except Exception:
                label_idx = -1

        return image, label_idx, species_name

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


def train_linear_probe(clip_model, train_loader, num_classes, device='cpu', lr=3e-2, weight_decay=1e-4, epochs=10):
    """Train a linear classifier on top of frozen CLIP image features."""
    clip_model.eval()
    # get feature dim
    # run one batch to infer dim
    it = iter(train_loader)
    imgs, _, _ = next(it)
    imgs = imgs.to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(imgs)

    feat_dim = feats.shape[1]
    feat_dtype = feats.dtype

    # Create classifier with the same dtype as CLIP image features to avoid dtype-mismatch
    # (some CLIP models may run in half precision on CUDA)
    classifier = torch.nn.Linear(feat_dim, num_classes).to(device, dtype=feat_dtype)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        classifier.train()
        pbar = tqdm(train_loader, desc=f"LinearProbe Epoch {epoch+1}/{epochs}")
        for images, labels, species in pbar:
            images = images.to(device)
            labels_idx = torch.tensor([int(l) for l in labels]).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(images)

            logits = classifier(image_features)
            loss = loss_fn(logits, labels_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': float(loss.detach().cpu())})

    def compute_probs(images_tensor):
        with torch.no_grad():
            feats = clip_model.encode_image(images_tensor.to(device))
            logits = classifier(feats)
            probs = torch.softmax(logits, dim=1)
        return probs

    return classifier, compute_probs

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
    # If per-class alphas were supplied, use them for the final test fusion; otherwise use scalar best_alpha
    if class_alphas is not None:
        ca_tensor_test = torch.tensor(class_alphas, dtype=torch.float32).unsqueeze(0).to(test_resnet_probs.device)
        test_fused_probs = test_resnet_probs * ca_tensor_test + test_clip_probs * (1.0 - ca_tensor_test)
        results['fusion_method'] = 'per_class'
    else:
        test_fused_probs = best_alpha * test_resnet_probs + (1 - best_alpha) * test_clip_probs
        results['fusion_method'] = 'global_alpha'

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
    if class_alphas is not None:
        print("Fusion (per-class alphas):")
    else:
        print(f"Fusion (α={best_alpha:.1f}):")
    print(f"  Top-1 Accuracy:    {test_top1:.4f} ({test_top1*100:.2f}%)")
    print(f"  Top-5 Accuracy:    {test_top5:.4f} ({test_top5*100:.2f}%)")
    print(f"  Balanced Accuracy: {test_balanced:.4f} ({test_balanced*100:.2f}%)")
    print(f"  Macro F1-Score:    {test_macro_f1:.4f} ({test_macro_f1*100:.2f}%)")
    print()
    print(f"Improvement over CLIP: {(test_top1 - clip_top1)*100:+.2f}%")
    print(f"Improvement over ResNet: {(test_top1 - resnet_top1)*100:+.2f}%")
    print("="*70)
    
    # When per-class alphas were provided, `best_alpha` is not meaningful — record null and set fusion_method
    if class_alphas is not None:
        best_alpha_value = None
        fusion_method = 'per_class'
    else:
        best_alpha_value = float(best_alpha)
        fusion_method = 'global_alpha'

    return {
        'best_alpha': best_alpha_value,
        'fusion_method': fusion_method,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_backbone', type=str, default='ViT-B/32')
    parser.add_argument('--prompts_file', type=str, default='data/delta_prompts.json')
    parser.add_argument('--shots', type=int, default=100)
    parser.add_argument('--mode', type=str, default='linear', choices=['linear', 'zero'])
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--val_samples', type=int, default=1000)
    parser.add_argument('--test_samples', type=int, default=1000)
    parser.add_argument('--output', type=str, default='final_fusion_results_shots100.json')
    parser.add_argument('--class_alphas_file', type=str, default=None,
                        help='Optional JSON file mapping species->alpha (per-class fusion weight for ResNet).')
    parser.add_argument('--compare_global_sweep', action='store_true',
                        help='If set and --class_alphas_file provided, also run the global alpha sweep for comparison.')
    parser.add_argument('--use_cached_train_features', action='store_true',
                        help='If set and cached train features exist in --features_dir, train linear probe from cached features instead of encoding images each batch.')
    parser.add_argument('--features_dir', type=str, default='/zfs/ai4good/student/hgupta/clip_features',
                        help='Directory where precomputed clip features are stored (train.pt, val.pt, test.pt)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"CLI args: {args}")

    # Configuration
    prompts_file = args.prompts_file
    temperature = args.temperature
    val_samples = args.val_samples
    test_samples = args.test_samples
    
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
    
    # Load CLIP model (try OpenAI CLIP first, then open_clip as a fallback)
    print("\nLoading CLIP model...")
    clip_models = {}
    try:
        model, preprocess = clip.load(args.clip_backbone, device=device)
        clip_models[args.clip_backbone] = (model, preprocess)
        print(f"Loaded CLIP backbone (clip): {args.clip_backbone}")
    except Exception as e:
        print(f"WARN: clip.load failed for {args.clip_backbone}: {e}")
        print("Trying open_clip fallback...")
        try:
            import open_clip

            # Try several name variants for the open_clip model (try exact user request first)
            requested = args.clip_backbone
            oc_variants = [requested, requested.replace('-', '_'), requested.lower(), requested.replace('/', '_'), requested.replace('/', '_').lower()]
            # Also include a small mapping for legacy names
            oc_variants += ['PE-Core-bigG-14-448', 'PE-Core-bigG-14-224']

            # Try a few common pretrained sources; prefer 'meta' tag when available
            pretrained_candidates = ['meta', 'openai', 'laion2b_s13b_b90k', 'laion400m_e32']
            oc_model = None
            oc_preprocess = None
            tried = set()
            for name in oc_variants:
                if name in tried:
                    continue
                tried.add(name)
                for src in pretrained_candidates:
                    try:
                        print(f"  open_clip: trying model={name}, pretrained={src} ...")
                        oc_model, _, oc_preprocess = open_clip.create_model_and_transforms(name, pretrained=src)
                        print(f"  open_clip: loaded {name} @ {src}")
                        break
                    except Exception as e2:
                        print(f"    open_clip try failed ({name}@{src}): {e2}")
                if oc_model is not None:
                    break

            if oc_model is None:
                # Last resort: try creating model architecture without weights for the requested name
                try:
                    print(f"  open_clip: trying to create architecture for {requested} without pretrained weights...")
                    oc_model = open_clip.create_model(requested, pretrained=False)
                    # open_clip doesn't always provide a preprocess in this API, fall back to torchvision transforms
                    oc_preprocess = None
                    print(f"  open_clip: created model architecture for {requested} (no pretrained weights)")
                except Exception as e3:
                    print(f"ERROR: open_clip could not load or create model {requested}: {e3}")
                    oc_model = None

            if oc_model is None:
                print(f"ERROR: Could not load CLIP backbone via either clip or open_clip for '{args.clip_backbone}'")
                sys.exit(1)

            # Move model to device and set eval mode
            oc_model = oc_model.to(device)
            oc_model.eval()

            # If open_clip provided a preprocess, use it; otherwise reuse a simple transform from clip if available
            if oc_preprocess is None:
                try:
                    _, oc_preprocess = clip.load('ViT-B/32', device='cpu')
                except Exception:
                    oc_preprocess = None

            clip_models[args.clip_backbone] = (oc_model, oc_preprocess)
        except Exception as e_open:
            print(f"ERROR: open_clip fallback failed: {e_open}")
            sys.exit(1)
    
    # Load prompts and encode text
    print(f"\nLoading prompts from {prompts_file}...")
    with open(prompts_file, 'r') as f:
        all_prompts = json.load(f)
    
    species_list = sorted(all_prompts.keys())
    print(f"Species: {len(species_list)}")

    # mapping from species name to numeric class index used in CSVs
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    # Load per-class alphas if provided
    class_alphas = None
    if args.class_alphas_file is not None:
        try:
            with open(args.class_alphas_file, 'r') as _f:
                class_alphas_raw = json.load(_f)
            # Build ordered alpha list matching species_list order
            missing = []
            alpha_list = []
            for s in species_list:
                a = class_alphas_raw.get(s)
                if a is None:
                    missing.append(s)
                    alpha_list.append(0.0)
                else:
                    alpha_list.append(float(a))
            if missing:
                print(f"Warning: {len(missing)} species missing in {args.class_alphas_file}; filling with alpha=0.0 for those.")
            class_alphas = alpha_list
            print(f"Loaded per-class alphas from {args.class_alphas_file}")
        except Exception as e:
            print(f"ERROR: Could not load class alphas file '{args.class_alphas_file}': {e}")
            class_alphas = None
    
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
                # Use open_clip.tokenize when working with open_clip models to ensure proper seq length
                if 'open_clip' in model.__class__.__module__:
                    import open_clip as _oc
                    # Try to use the model's expected context length so tokenization matches positional embeddings
                    ctx_len = getattr(model, 'context_length', None)
                    if ctx_len is None:
                        # Try common attribute locations
                        ctx_len = getattr(getattr(model, 'transformer', None), 'context_length', None)
                        if ctx_len is None:
                            ctx_len = getattr(getattr(model, 'text', None), 'context_length', None)

                    if ctx_len is not None:
                        text_tokens = _oc.tokenize(batch_prompts, context_length=int(ctx_len)).to(device)
                    else:
                        text_tokens = _oc.tokenize(batch_prompts).to(device)
                else:
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
    
    # Use CLIP preprocess
    preprocess = list(clip_models.values())[0][1]

    # Create dataloaders
    val_dataset = MushroomDataset(
        csv_path=os.path.join(data_dir, "val.csv"),
        data_dir=data_dir,
        transform=preprocess,
        n_samples=val_samples,
        species_to_idx=species_to_idx
    )

    test_dataset = MushroomDataset(
        csv_path=os.path.join(data_dir, "test.csv"),
        data_dir=data_dir,
        transform=preprocess,
        n_samples=test_samples,
        species_to_idx=species_to_idx
    )

    # Prepare few-shot training dataset (shots per class)
    train_csv = os.path.join(data_dir, 'train.csv')
    if args.mode == 'linear':
        # If cached train features exist and user requested, load them and create a lightweight DataLoader
        train_loader = None
        cached_train_path = os.path.join(args.features_dir, 'train.pt')
        if args.use_cached_train_features and os.path.exists(cached_train_path):
            try:
                print(f"Loading cached train features from {cached_train_path}...")
                cached = torch.load(cached_train_path, map_location='cpu')
                train_feats = cached.get('features')
                train_labels = cached.get('labels')
                if train_feats is None or train_labels is None:
                    raise RuntimeError('cached train file missing keys')
                # Create TensorDataset and DataLoader
                from torch.utils.data import TensorDataset
                train_dataset = TensorDataset(train_feats, train_labels)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
                print(f'Using cached train features: {train_feats.shape[0]} samples, feat_dim={train_feats.shape[1]}')
            except Exception as e:
                print(f"Could not load cached train features: {e}; falling back to on-the-fly dataset")

        if train_loader is None:
            print(f"Preparing few-shot train dataset with {args.shots} shots per class...")
            train_dataset = FewShotDataset(csv_path=train_csv, data_dir=data_dir, transform=preprocess, shots=args.shots, species_list=species_list, species_to_idx=species_to_idx)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            if len(train_dataset) == 0:
                print("Warning: Few-shot train dataset is empty after filtering missing images. Falling back to zero-shot mode (no linear probe).")
                train_loader = None
                args.mode = 'zero'
    else:
        train_loader = None
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # If test set is empty after filtering missing files, we cannot evaluate — exit with a clear message.
    if len(test_dataset) == 0:
        print(f"ERROR: test dataset is empty after filtering missing image files (test=0). Cannot proceed.")
        sys.exit(1)
    
    # If linear probing requested, train linear probe on CLIP backbone
    clip_model = list(clip_models.values())[0][0]

    if args.mode == 'linear':
        # If we created the train_loader from cached features, call a specialized trainer
        if args.use_cached_train_features and os.path.exists(os.path.join(args.features_dir, 'train.pt')):
            def train_linear_probe_from_features(train_loader, num_classes, device='cpu', lr=3e-2, weight_decay=1e-4, epochs=10):
                # infer feature dim from first batch
                it = iter(train_loader)
                feats, labels = next(it)
                feat_dim = feats.shape[1]
                feat_dtype = feats.dtype
                classifier = torch.nn.Linear(feat_dim, num_classes).to(device, dtype=feat_dtype)
                optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, weight_decay=weight_decay)
                loss_fn = torch.nn.CrossEntropyLoss()
                for epoch in range(epochs):
                    classifier.train()
                    pbar = tqdm(train_loader, desc=f"LinearProbe Epoch {epoch+1}/{epochs}")
                    for feats_b, labels_b in pbar:
                        feats_b = feats_b.to(device)
                        labels_idx = labels_b.to(device)
                        logits = classifier(feats_b)
                        loss = loss_fn(logits, labels_idx)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        pbar.set_postfix({'loss': float(loss.detach().cpu())})

                def compute_probs_from_feats(features_tensor):
                    with torch.no_grad():
                        logits = classifier(features_tensor.to(device))
                        probs = torch.softmax(logits, dim=1)
                    return probs

                return classifier, compute_probs_from_feats

            classifier, compute_probs_from_feats = train_linear_probe_from_features(
                train_loader, num_classes=len(species_list), device=device, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
            )
            # The trainer above returns a `compute_probs_from_feats` that expects precomputed
            # feature tensors. Wrap it so callers can pass raw image tensors (as elsewhere
            # in the code) — we encode images with the same CLIP model and normalize them
            # before passing to the classifier.
            def compute_clip_probs_fn(images_tensor):
                with torch.no_grad():
                    feats = clip_model.encode_image(images_tensor.to(device))
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                return compute_probs_from_feats(feats)
        else:
            classifier, compute_clip_probs_fn = train_linear_probe(
                clip_model, train_loader, num_classes=len(species_list), device=device,
                lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
            )
    else:
        # zero-shot: compute probs via prompt-ensemble as before
        classifier = None

    # Evaluate fusion with alpha sweep
    # We'll compute CLIP probs using the trained linear probe if available
    def compute_clip_probs_from_model(images):
        if classifier is not None:
            return compute_clip_probs_fn(images)
        else:
            return compute_clip_ensemble_probs(images, clip_models, text_embeddings, temperature)

    # Collect predictions on validation set
    print("\nCollecting validation predictions...")
    val_clip_probs = []
    val_resnet_probs = []
    val_labels = []

    for images, labels, _ in tqdm(val_loader, desc="Validation"):
        images = images.to(device)

        # CLIP probabilities (linear probe or zero-shot)
        clip_probs = compute_clip_probs_from_model(images)

        # ResNet probabilities
        resnet_probs = compute_resnet_probs(images, resnet_model)

        val_clip_probs.append(clip_probs.cpu())
        val_resnet_probs.append(resnet_probs.cpu())
        val_labels.append(labels)

    val_clip_probs = torch.cat(val_clip_probs, dim=0)
    val_resnet_probs = torch.cat(val_resnet_probs, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    print("\nPerforming fusion on validation set...")
    alpha_results = {}

    # If per-class alphas were provided, apply them elementwise across classes.
    # Otherwise fall back to a scalar sweep over global alphas.
    best_alpha = 0.0
    best_top1 = 0.0

    if class_alphas is not None:
        # build a tensor (1, C) for broadcasting
        ca_tensor = torch.tensor(class_alphas, dtype=torch.float32).unsqueeze(0).to(val_resnet_probs.device)
        fused_probs = val_resnet_probs * ca_tensor + val_clip_probs * (1.0 - ca_tensor)
        preds = torch.argmax(fused_probs, dim=1).numpy()
        labels_np = val_labels.numpy()

        top1 = (preds == labels_np).mean()
        balanced = balanced_accuracy_score(labels_np, preds)
        macro_f1 = f1_score(labels_np, preds, average='macro', zero_division=0)

        alpha_results['per_class'] = {
            'top1': float(top1),
            'balanced': float(balanced),
            'macro_f1': float(macro_f1)
        }
        print(f"  Per-class fusion: Top-1={top1:.4f}, Balanced={balanced:.4f}, F1={macro_f1:.4f}")

        # Optionally also run global scalar sweep for comparison
        if args.compare_global_sweep:
            alphas = [i/20 for i in range(21)]
            for alpha in alphas:
                fused_probs_g = alpha * val_resnet_probs + (1 - alpha) * val_clip_probs
                preds_g = torch.argmax(fused_probs_g, dim=1).numpy()
                top1_g = (preds_g == labels_np).mean()
                balanced_g = balanced_accuracy_score(labels_np, preds_g)
                macro_f1_g = f1_score(labels_np, preds_g, average='macro', zero_division=0)
                alpha_results[alpha] = {
                    'top1': float(top1_g),
                    'balanced': float(balanced_g),
                    'macro_f1': float(macro_f1_g)
                }
                print(f"  Global α={alpha:.2f}: Top-1={top1_g:.4f}, Balanced={balanced_g:.4f}, F1={macro_f1_g:.4f}")
                if top1_g > best_top1:
                    best_top1 = top1_g
                    best_alpha = alpha
    else:
        alphas = [i/20 for i in range(21)]
        for alpha in alphas:
            fused_probs = alpha * val_resnet_probs + (1 - alpha) * val_clip_probs
            preds = torch.argmax(fused_probs, dim=1).numpy()
            labels_np = val_labels.numpy()

            top1 = (preds == labels_np).mean()
            balanced = balanced_accuracy_score(labels_np, preds)
            macro_f1 = f1_score(labels_np, preds, average='macro', zero_division=0)

            alpha_results[alpha] = {
                'top1': float(top1),
                'balanced': float(balanced),
                'macro_f1': float(macro_f1)
            }

            print(f"  α={alpha:.2f}: Top-1={top1:.4f}, Balanced={balanced:.4f}, F1={macro_f1:.4f}")

            if top1 > best_top1:
                best_top1 = top1
                best_alpha = alpha

    print(f"\nBest α on validation: {best_alpha:.2f} (Top-1={best_top1:.4f})")

    # Evaluate on test set with best alpha
    print(f"\nEvaluating on test set with α={best_alpha:.2f}...")
    test_clip_probs = []
    test_resnet_probs = []
    test_labels = []
    test_top5_preds = []

    # For testing, if per-class alphas exist, apply them elementwise; otherwise use scalar best_alpha
    if class_alphas is not None:
        ca_tensor_test = torch.tensor(class_alphas, dtype=torch.float32).unsqueeze(0).to(device)
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            clip_probs = compute_clip_probs_from_model(images)
            resnet_probs = compute_resnet_probs(images, resnet_model)

            fused_probs = resnet_probs * ca_tensor_test + clip_probs * (1.0 - ca_tensor_test)

            top5 = torch.topk(fused_probs, k=5, dim=1).indices

            test_clip_probs.append(clip_probs.cpu())
            test_resnet_probs.append(resnet_probs.cpu())
            test_labels.append(labels)
            test_top5_preds.append(top5.cpu())
    else:
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            clip_probs = compute_clip_probs_from_model(images)
            resnet_probs = compute_resnet_probs(images, resnet_model)

            fused_probs = best_alpha * resnet_probs + (1 - best_alpha) * clip_probs

            top5 = torch.topk(fused_probs, k=5, dim=1).indices

            test_clip_probs.append(clip_probs.cpu())
            test_resnet_probs.append(resnet_probs.cpu())
            test_labels.append(labels)
            test_top5_preds.append(top5.cpu())

    test_clip_probs = torch.cat(test_clip_probs, dim=0)
    test_resnet_probs = torch.cat(test_resnet_probs, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    test_top5_preds = torch.cat(test_top5_preds, dim=0)

    test_fused_probs = best_alpha * test_resnet_probs + (1 - best_alpha) * test_clip_probs
    test_preds = torch.argmax(test_fused_probs, dim=1).numpy()
    test_labels_np = test_labels.numpy()

    test_top1 = (test_preds == test_labels_np).mean()
    test_top5_correct = torch.any(test_top5_preds == test_labels.unsqueeze(1), dim=1).numpy()
    test_top5 = test_top5_correct.mean()
    test_balanced = balanced_accuracy_score(test_labels_np, test_preds)
    test_macro_f1 = f1_score(test_labels_np, test_preds, average='macro', zero_division=0)

    clip_preds = torch.argmax(test_clip_probs, dim=1).numpy()
    resnet_preds = torch.argmax(test_resnet_probs, dim=1).numpy()

    clip_top1 = (clip_preds == test_labels_np).mean()
    resnet_top1 = (resnet_preds == test_labels_np).mean()

    results = {
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

    results['config'] = {
        'clip_backbone': args.clip_backbone,
        'resnet_checkpoint': resnet_checkpoint,
        'prompts_file': prompts_file,
        'temperature': temperature,
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'shots': args.shots,
        'mode': args.mode,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'batch_size': args.batch_size
    }

    # Record whether per-class alphas were supplied and include them for provenance
    results['class_alphas_provided'] = class_alphas is not None
    if class_alphas is not None:
        results['class_alphas'] = class_alphas

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
