#!/usr/bin/env python3
"""
test_few_shots_overall.py (prompt-aware few-shot extensions)

Unified few-shot + zero-shot evaluation with prompt integration.
- Zero-shot (mean prompt & prompt pooling via direct-eval)
- Few-shot: prototype + linear probe (both run for every shot)
- Prompt-aware few-shot:
    * prototype combining image prototypes + text prompt embeddings
    * linear probe trained with prompt embeddings as pseudo-examples
- Supports multiple CLIP-style backbones via open_clip
- Caches text and image embeddings
- Nested tqdm progress bars when running in a TTY (interactive) to minimize overhead on clusters

Usage:
    python merged_few_shots.py --data-root ./data --train-csv splits/train.csv --val-csv splits/val.csv \
        --test-csv splits/test.csv --labels labels.tsv --backbones ViT-B-32 ViT-L-14 --shots 0 1 5 \
        --device cuda --direct-eval --use-prompts-for-prototype --use-prompts-for-linear --prompt-alpha 0.6

Author: merged from user's scripts + improvements
"""
from __future__ import annotations
import os
import sys
import argparse
import json
import hashlib
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import csv
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import f1_score, confusion_matrix

# require open_clip
try:
    import open_clip
except Exception as e:
    raise ImportError("open_clip (open_clip_torch) is required. Install with: pip install open_clip_torch") from e

# tqdm: use it only if available; we'll choose to enable bars only in TTY
try:
    from tqdm.auto import tqdm
except Exception:
    # fallback: dummy tqdm
    def tqdm(x, *args, **kwargs):
        return x
    
# Limit CPU threads (avoid oversubscription on shared node)
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_MAX_THREADS"] = "16"
# Enable faster GPU memory allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Allow TensorFloat32 on modern GPUs (L40S supports this)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


# ---------- globals ----------
RESURLTS_DIR="/home/c/dkorot/AI4GOOD/ai4good-mushroom/results"
DEFAULT_CACHE_DIR = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/features"
DEFAULT_TEXT_CACHE_DIR = Path(".cache_text_embeddings")
DEFAULT_PROMPTS_JSON = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/data_prompts_label/delta_prompts.json"
torch.set_float32_matmul_precision('high')
DEFAULT_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PRETRAINED = {
					"PE-Core-bigG-14-448": "meta",
                    "ViT-gopt-16-SigLIP2-384": "webli",
					"ViT-H-14-378-quickgelu": "dfn5b",
					"ViT-B-32-quickgelu": "openai",
					"ViT-L-14": "openai",
					"ViT-B-16": "openai",
					}
SHOTS = [0, 1, 5, 10, 20, 50, 100]
ALPHAS = [0.0,0.2,0.4,0.6,0.8,1.0]

DATA_ROOT = "/home/c/dkorot/AI4GOOD/provided_dir/datasets/mushroom/merged_dataset"
TRAIN_CSV = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/train.csv"
VAL_CSV = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/val.csv"
TEST_CSV = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/test.csv"
LABELS = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/data_prompts_label/labels.tsv"

with open('/home/c/dkorot/AI4GOOD/ai4good-mushroom/data_prompts_label/best_alphas.json', 'r') as f:
    BEST_ALPHAS = json.load(f)


# ------------------------------
# Prompt templates & clip_utils-like functions
# ------------------------------
SHORT_PROMPTS = [
    "a photo of {label}",
    "a close-up of {label} mushroom",
    "a photograph of {label}",
    "a picture of {label} in nature",
    "close-up of {label} cap",
]

NAME_PROMPT_TEMPLATES = [
    "a photo of {name}",
    "{name} mushroom",
    "{name} fungus",
]


def _build_prompt_list_for_label(label: str, prompt_set: str = "ensemble",
                                 include_common_names: bool = False, mg: Optional[Any] = None) -> List[str]:
    if prompt_set == "v1":
        return [t.format(label=label) for t in SHORT_PROMPTS]
    elif prompt_set == "names":
        names = [label]
        if include_common_names and mg is not None:
            if label in mg.species_knowledge:
                info = mg.species_knowledge[label]
                for n in info.get("common_names", []):
                    if n not in names:
                        names.append(n)
        prompts = []
        for n in names:
            for tmpl in NAME_PROMPT_TEMPLATES:
                prompts.append(tmpl.format(name=n))
        return prompts
    elif prompt_set == "ensemble":
        prompts = [t.format(label=label) for t in SHORT_PROMPTS]
        names = [label]
        if include_common_names and mg is not None and label in getattr(mg, "species_knowledge", {}):
            info = mg.species_knowledge[label]
            for n in info.get("common_names", []):
                if n not in names:
                    names.append(n)
        for n in names:
            for tmpl in NAME_PROMPT_TEMPLATES:
                p = tmpl.format(name=n)
                if p not in prompts:
                    prompts.append(p)
        return prompts
    else:
        raise ValueError(f"Unknown prompt_set: {prompt_set}")


def get_text_embeddings(labels: List[str],
                        prompt_set: str = "ensemble",
                        model_name: str = "ViT-B-32",
                        device: Optional[str] = None,
                        batch_size: int = 128,
                        include_common_names: bool = False,
                        common_prompts_path: Optional[str] = None,
                        pretrained: str = "openai",
                        cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Compute and cache text embeddings per label.
    Returns: {label: {'prompts': [...], 'prompt_embeddings': Tensor (n x D), 'label_embedding': Tensor (D,)}}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = (cache_dir or DEFAULT_TEXT_CACHE_DIR)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    try:
        model.encode_text = torch.compile(model.encode_text)
    except Exception:
        pass

    # try load common prompts if requested
    common_prompts = None
    if include_common_names and common_prompts_path and os.path.exists(common_prompts_path):
        try:
            with open(common_prompts_path, "r", encoding="utf-8") as f:
                common_prompts = json.load(f)
        except Exception:
            common_prompts = None

    results: Dict[str, Any] = {}
    for label in labels:
        if common_prompts and label in common_prompts:
            prompts = list(common_prompts[label])
        else:
            prompts = _build_prompt_list_for_label(label, prompt_set, include_common_names, mg=None)

        # cache key
        h = hashlib.sha1(("|".join(prompts) + model_name + pretrained).encode()).hexdigest()
        cache_path = cache_dir / f"{label}_{h}.npz"
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=True)
            pe = torch.tensor(z["prompt_embeddings"])
            le = torch.tensor(z["label_embedding"])
            results[label] = {"prompts": list(z["prompts"]), "prompt_embeddings": pe, "label_embedding": le}
            continue

        all_embs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                toks = tokenizer(prompts[i:i + batch_size]).to(device)
                emb = model.encode_text(toks)
                emb = F.normalize(emb, dim=-1)
                all_embs.append(emb.cpu())
        if all_embs:
            pe = torch.cat(all_embs, dim=0)
            le = F.normalize(pe.mean(dim=0), dim=-1).cpu()
        else:
            D = model.text_projection.shape[1] if hasattr(model, "text_projection") else 512
            pe = torch.empty((0, D))
            le = torch.zeros((D,))
        np.savez_compressed(cache_path, prompts=np.array(prompts, dtype=object),
                            prompt_embeddings=pe.numpy(), label_embedding=le.numpy())
        results[label] = {"prompts": prompts, "prompt_embeddings": pe, "label_embedding": le}
    return results


# ------------------------------
# Labels & prompts
# ------------------------------
def load_labels(labels_tsv: str) -> Tuple[np.ndarray, Dict[str, int]]:
    ids, names = [], []
    with open(labels_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                i = int(parts[0])
                name = parts[1]
                ids.append(i)
                names.append(name)
    return np.array(names, dtype=object), {n: i for i, n in zip(ids, names)}


def load_prompts(prompts_path: Optional[str], label_names: List[str]) -> Dict[str, List[str]]:
    if prompts_path and os.path.exists(prompts_path):
        try:
            with open(prompts_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            prompts: Dict[str, List[str]] = {}
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        prompts[k] = v
                    elif isinstance(v, str):
                        prompts[k] = [v]
            for lbl in label_names:
                if lbl not in prompts:
                    prompts[lbl] = [f"a photo of a {lbl}.", f"a close-up photograph of {lbl}."]
            return prompts
        except Exception:
            pass
    return {lbl: [f"a photo of a {lbl}.", f"a photograph of {lbl}."] for lbl in label_names}


# ------------------------------
# Feature cache loader (image features)
# ------------------------------
def ensure_features(split: str, backbone: str, data_root: str, csv_path: str, labels_tsv: str,
                    save_dir: str = DEFAULT_CACHE_DIR, pretrained: str = "openai") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    cache_path = Path(save_dir) / backbone / f"{split}.npz"
    if cache_path.exists():
        try:
            z = np.load(cache_path, allow_pickle=True)
            return z["X"], z["y"], z.get("paths", None)
        except Exception:
            try:
                cache_path.unlink()
            except Exception:
                pass

    # fallback: call external dump_features script
    os.makedirs(cache_path.parent, exist_ok=True)
    cmd = (
        f'python scripts/dump_features.py '
        f'--data-root "{data_root}" --csv "{csv_path}" --labels "{labels_tsv}" '
        f'--backbone "{backbone}" --pretrained "{pretrained}" --save-dir "{save_dir}"'
    )
    print(f"[ensure_features] Running external feature dump: {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"Feature dumping failed (exit {rc}). Ensure scripts/dump_features.py exists and runs.")
    z = np.load(cache_path, allow_pickle=True)
    return z["X"], z["y"], z.get("paths", None)


# ------------------------------
# Metrics & sampling
# ------------------------------
def topk_acc(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    topk = np.argsort(-scores, axis=1)[:, :k]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def balanced_acc(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(K))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = np.where(cm.sum(1) > 0, np.diag(cm) / cm.sum(1), 0.0)
    return float(np.nanmean(per_class))


def sample_few_shot_indices(y_train: np.ndarray, K: int, n_shot: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    support_indices = []
    support_labels = []
    for k in range(K):
        inds = np.where(y_train == k)[0]
        if len(inds) == 0:
            raise ValueError(f"No training examples for class {k}")
        replace = len(inds) < n_shot
        chosen = rng.choice(inds, size=n_shot, replace=replace)
        support_indices.append(chosen)
        support_labels.append(np.full(n_shot, k, dtype=np.int64))
    return np.concatenate(support_indices), np.concatenate(support_labels)


# ------------------------------
# Prototype & linear probe
# ------------------------------
def prototype_classifier(X_support: np.ndarray, y_support: np.ndarray, K: int) -> np.ndarray:
    D = X_support.shape[1]
    prototypes = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        inds = np.where(y_support == k)[0]
        p = X_support[inds].mean(axis=0)
        n = np.linalg.norm(p)
        if n > 0:
            p = p / n
        prototypes[k] = p
    return prototypes


def prototype_classifier_with_prompts(X_support: np.ndarray, y_support: np.ndarray, K: int,
                                      label_names: List[str], text_embs: Dict[str, Any],
                                      alpha: float = 0.5) -> np.ndarray:
    """
    Build prototypes by combining few-shot image prototypes and text label embeddings.

    alpha: weight for image prototype (0..1). Combined = normalize(alpha * img_proto + (1-alpha) * text_emb)
    text_embs: output of get_text_embeddings -> for each label, 'label_embedding' tensor (cpu)
    """
    D = X_support.shape[1]
    prototypes = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        inds = np.where(y_support == k)[0]
        if len(inds) == 0:
            img_proto = np.zeros((D,), dtype=np.float32)
        else:
            img_proto = X_support[inds].mean(axis=0).astype(np.float32)
            n = np.linalg.norm(img_proto)
            if n > 0:
                img_proto = img_proto / n

        lbl = label_names[k]
        txt_tensor = text_embs.get(lbl, {}).get("label_embedding", None)
        if txt_tensor is None:
            txt_proto = np.zeros((D,), dtype=np.float32)
        else:
            if isinstance(txt_tensor, torch.Tensor):
                txt_proto = txt_tensor.cpu().numpy().astype(np.float32)
            else:
                txt_proto = np.array(txt_tensor, dtype=np.float32)
            n = np.linalg.norm(txt_proto)
            if n > 0:
                txt_proto = txt_proto / n

        combined = alpha * img_proto + (1.0 - alpha) * txt_proto
        n = np.linalg.norm(combined)
        if n > 0:
            combined = combined / n
        prototypes[k] = combined
    return prototypes


class LinearProbe(nn.Module):
    def __init__(self, dim: int, K: int):
        super().__init__()
        self.linear = nn.Linear(dim, K)

    def forward(self, x):
        return self.linear(x)


def train_linear_probe(X_support: np.ndarray, y_support: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                       epochs: int = 200, lr: float = 1e-3, batch_size: int = 32, device: str = "cpu", seed: int = 42,
                       compile_model: bool = False, show_progress: bool = False) -> nn.Module:
    torch.manual_seed(seed)
    X = torch.from_numpy(X_support).float().to(device)
    y = torch.from_numpy(y_support).long().to(device)
    D = X.shape[1]
    K = int(y.max()) + 1

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = LinearProbe(D, K).to(device)
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception:
            pass
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_loss = float('inf')

    # epoch loop with optional tqdm nested bar
    epoch_iter = range(epochs)
    if show_progress and sys.stdout.isatty():
        epoch_iter = tqdm(epoch_iter, desc="epochs", leave=False, miniters=1)

    for ep in epoch_iter:
        model.train()
        ep_loss = 0.0
        n = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        ep_loss /= max(1, n)

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(X_val).float().to(device)
                yv = torch.from_numpy(y_val).long().to(device)
                val_loss = float(loss_fn(model(xv), yv).item())
        else:
            val_loss = ep_loss

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# def train_linear_probe_with_prompts(X_support: np.ndarray, y_support: np.ndarray,
#                                     label_names: List[str], text_embs: Dict[str, Any],
#                                     K: int, alpha: float = 0.5,
#                                     epochs: int = 200, lr: float = 1e-3, batch_size: int = 32, device: str = "cpu",
#                                     seed: int = 42, compile_model: bool = False, show_progress: bool = False) -> nn.Module:
#     """
#     Train linear probe with prompt embeddings added as one pseudo-example per class.

#     - X_support: (N, D) image embeddings
#     - y_support: (N,) labels 0..K-1
#     - text_embs: dict[label] -> {'label_embedding': Tensor}
#     Returns trained nn.Module
#     """
#     # collect text prototypes (one per class)
#     D = X_support.shape[1]
#     text_X = []
#     text_y = []
#     for k, lbl in enumerate(label_names):
#         txt_tensor = text_embs.get(lbl, {}).get("label_embedding", None)
#         if txt_tensor is None:
#             vec = np.zeros((D,), dtype=np.float32)
#         else:
#             if isinstance(txt_tensor, torch.Tensor):
#                 vec = txt_tensor.cpu().numpy().astype(np.float32)
#             else:
#                 vec = np.array(txt_tensor, dtype=np.float32)
#             n = np.linalg.norm(vec)
#             if n > 0:
#                 vec = vec / n
#         text_X.append(vec)
#         text_y.append(k)
#     text_X = np.stack(text_X, axis=0)
#     text_y = np.array(text_y, dtype=np.int64)

#     # # concat image support and text pseudo-examples
#     # X_aug = np.concatenate([X_support.astype(np.float32), text_X], axis=0)
#     # y_aug = np.concatenate([y_support.astype(np.int64), text_y], axis=0)
    
#     # --- Apply alpha weighting between image and text examples ---
#     # Scale the text prototypes before concatenation
#     X_img = X_support.astype(np.float32)
#     X_txt = text_X.astype(np.float32)

#     # normalize again just in case
#     X_img /= np.linalg.norm(X_img, axis=1, keepdims=True) + 1e-8
#     X_txt /= np.linalg.norm(X_txt, axis=1, keepdims=True) + 1e-8

#     # combine with weighting
#     X_aug = np.concatenate([alpha * X_img, (1.0 - alpha) * X_txt], axis=0)
#     y_aug = np.concatenate([y_support.astype(np.int64), text_y], axis=0)

#     # Call existing trainer
#     model = train_linear_probe(X_aug, y_aug, X_val=None, y_val=None,
#                                epochs=epochs, lr=lr, batch_size=batch_size,
#                                device=device, seed=seed, compile_model=compile_model, show_progress=show_progress)
#     return model
def train_linear_probe_with_prompts(
    X_support, y_support, label_names, text_embs, K, alpha=0.5,
    epochs=200, lr=1e-3, batch_size=32, device="cpu",
    seed=42, compile_model=False, show_progress=False):

    # --- mix image & text features per class ---
    X_mixed = []
    y_mixed = []
    for k, lbl in enumerate(label_names):
        inds = np.where(y_support == k)[0]
        if len(inds) == 0:
            continue
        img_feats = X_support[inds]
        txt_vec = text_embs[lbl]["label_embedding"].cpu().numpy()
        txt_vec /= np.linalg.norm(txt_vec) + 1e-8
        mixed = alpha * img_feats + (1 - alpha) * txt_vec
        mixed /= np.linalg.norm(mixed, axis=1, keepdims=True) + 1e-8
        X_mixed.append(mixed)
        y_mixed.append(np.full(len(mixed), k))
    X_aug = np.concatenate(X_mixed, axis=0)
    y_aug = np.concatenate(y_mixed, axis=0)

    # --- train standard linear probe ---
    model = train_linear_probe(
        X_aug, y_aug,
        epochs=epochs, lr=lr, batch_size=batch_size,
        device=device, seed=seed,
        compile_model=compile_model, show_progress=show_progress)
    return model



# ------------------------------
# Zero-shot (mean prompt) caching
# ------------------------------
def zero_shot_scores(label_names: List[str], backbone: str, pretrained: str, device: str,
                     prompts_path: Optional[str] = None, cache_dir: str = DEFAULT_CACHE_DIR,
                     batch_size: int = 128) -> np.ndarray:
    cache_path = Path(cache_dir) / f"{backbone}_zs_embeddings.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        return z["mean_embeddings"]

    model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(backbone)
    prompts_path = prompts_path or DEFAULT_PROMPTS_JSON
    label_prompts = load_prompts(prompts_path, label_names)

    all_prompts, all_labels_idx = [], []
    for idx, label in enumerate(label_names):
        ps = label_prompts.get(label, [f"a photo of a {label}."])
        all_prompts.extend(ps)
        all_labels_idx.extend([idx] * len(ps))

    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_prompts), batch_size):
            toks = tokenizer(all_prompts[i:i + batch_size]).to(device)
            emb = model.encode_text(toks)
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
    if all_embs:
        all_embs = torch.cat(all_embs, dim=0)
    else:
        D = model.text_projection.shape[1] if hasattr(model, "text_projection") else 512
        all_embs = torch.zeros((0, D))
    all_labels_idx = torch.tensor(all_labels_idx)

    mean_embeddings = torch.zeros((len(label_names), all_embs.shape[1]))
    for i in range(len(label_names)):
        mask = (all_labels_idx == i)
        if mask.any():
            mean_embeddings[i] = all_embs[mask].mean(dim=0)
    mean_embeddings = F.normalize(mean_embeddings, dim=-1).numpy().T  # [D, K]
    np.savez_compressed(cache_path, mean_embeddings=mean_embeddings)
    return mean_embeddings


# ------------------------------
# Image encoding utilities (batched) with progress
# ------------------------------
class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], preprocess, root: str = "."):
        self.paths = image_paths
        self.preprocess = preprocess
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        local = p if os.path.isabs(p) else os.path.join(self.root, p)
        if not os.path.exists(local):
            return None
        im = Image.open(local).convert("RGB")
        return self.preprocess(im)


def encode_images(model, preprocess, image_paths: List[str], device: str, batch_size: int = 32,
                  cache_prefix: Optional[str] = None, cache_dir: str = DEFAULT_CACHE_DIR, root: str = ".",
                  show_progress: bool = False):
    if cache_prefix:
        cache_path = Path(cache_dir) / f"{cache_prefix}.npz"
        if cache_path.exists():
            try:
                arr = np.load(cache_path)["embeddings"]
                return [torch.tensor(a) for a in arr]
            except Exception:
                pass

    dataset = ImageDataset(image_paths, preprocess, root=root)
    # loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=min(16, os.cpu_count() // 4),
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=4
                        )


    model.eval()
    emb_list = []
    it = loader
    if show_progress and sys.stdout.isatty():
        it = tqdm(loader, desc="encode_images", leave=False, miniters=1)
    with torch.no_grad():
        for batch in it:
            batch = [b for b in batch if b is not None]
            if not batch:
                continue
            batch = torch.stack(batch).to(device, non_blocking=True)
            # emb = model.encode_image(batch)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                emb = model.encode_image(batch)
            emb = F.normalize(emb, dim=-1)
            emb_list.append(emb.cpu())
    if not emb_list:
        return []
    emb_all = torch.cat(emb_list)
    if cache_prefix:
        np.savez_compressed(Path(cache_dir) / f"{cache_prefix}.npz", embeddings=emb_all.numpy())
    return list(emb_all)


# ------------------------------
# Prompt-pool helpers
# ------------------------------
def get_text_embeddings_bundle_from_clip(model_name: str, model, tokenizer, label_prompts: Dict[str, List[str]],
                                        device: str = "cuda", batch_size: int = 128, show_progress: bool = False):
    results = {}
    label_items = list(label_prompts.items())
    iterator = label_items
    if show_progress and sys.stdout.isatty():
        iterator = tqdm(label_items, desc="encode_prompts", leave=False, miniters=1)
    for lbl, prompts in iterator:
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                toks = tokenizer(prompts[i:i + batch_size]).to(device)
                # emb = model.encode_text(toks)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    emb = model.encode_text(toks)
                emb = F.normalize(emb, dim=-1)
                all_embs.append(emb.cpu())
        if all_embs:
            pe = torch.cat(all_embs, dim=0)
            le = F.normalize(pe.mean(dim=0), dim=-1)
        else:
            D = model.text_projection.shape[1] if hasattr(model, "text_projection") else 512
            pe = torch.empty((0, D))
            le = torch.zeros((D,))
        results[lbl] = {"prompts": prompts, "prompt_embeddings": pe, "label_embedding": le}
    return results


def score_all_vectorized(image_embeddings: List[torch.Tensor], mean_dict: Dict[str, torch.Tensor], labels: List[str]):
    img_mat = torch.stack([e for e in image_embeddings if e is not None])
    mean_mat = torch.stack([mean_dict[l] for l in labels])
    sims = img_mat @ mean_mat.T
    preds_idx = sims.argmax(dim=1)
    return [labels[i] for i in preds_idx]


def acc_from_preds(preds: List[str], trues: List[str]) -> float:
    return float(sum(p == t for p, t in zip(preds, trues)) / len(trues))


# ------------------------------
# High-level per-backbone evaluation (cached features) with nested progress bars
# ------------------------------
def evaluate_backbone(backbone: str, args: argparse.Namespace, label_names: List[str], K: int, show_progress: bool = False):
    model_pretrained = args.pretrained.get(backbone, "openai")
    if show_progress and sys.stdout.isatty():
        print(f"Evaluating backbone: {backbone} (pretrained={model_pretrained})")

    if show_progress and sys.stdout.isatty():
        print("~~~~~~~~~~~~~~endure features")
    X_tr, y_tr, _ = ensure_features("train", backbone, args.data_root, args.train_csv, args.labels, save_dir=args.save_dir, pretrained=model_pretrained)
    X_vl, y_vl, _ = ensure_features("val", backbone, args.data_root, args.val_csv, args.labels, save_dir=args.save_dir, pretrained=model_pretrained)
    X_te, y_te, _ = ensure_features("test", backbone, args.data_root, args.test_csv, args.labels, save_dir=args.save_dir, pretrained=model_pretrained)

    # Normalize all embeddings to unit length (important for cosine-space CLIP)
    if show_progress and sys.stdout.isatty():
        print("~~~~~~~~~~~~~~normalized embeddings")
    X_tr = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-8)
    X_vl = X_vl / (np.linalg.norm(X_vl, axis=1, keepdims=True) + 1e-8)
    X_te = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-8)


    # Zero-shot text embeddings (for zero-shot only)
    if show_progress and sys.stdout.isatty():
        print("~~~~~~~~~~~~~~text embedding stuff")
    text_embeddings = zero_shot_scores(label_names, backbone, model_pretrained, args.device, args.prompts_path, cache_dir=args.save_dir)

    # Optionally load per-label prompt embeddings if few-shot prompt-aware mode requested
    text_embs_bundle = None
    if args.use_prompts_for_prototype or args.use_prompts_for_linear:
        # get_text_embeddings returns dict[label] -> {'prompts':..., 'prompt_embeddings': Tensor, 'label_embedding': Tensor}
        text_embs_bundle = get_text_embeddings(list(label_names),
                                               prompt_set="ensemble",
                                               model_name=backbone,
                                               device=args.device,
                                               batch_size=128,
                                               include_common_names=False,
                                               common_prompts_path=args.prompts_path,
                                               pretrained=model_pretrained,
                                               cache_dir=Path(args.save_dir) / "text_cache")

    device = args.device
    X_vl_gpu = torch.from_numpy(X_vl).float().to(device)
    X_te_gpu = torch.from_numpy(X_te).float().to(device)
    records = []

    # Zero-shot if requested
    if show_progress and sys.stdout.isatty():
        print("~~~~~~~~~~~~~~zero shot")
    if 0 in args.shots:
        for split_name, X_gpu, y in [("val", X_vl_gpu, y_vl), ("test", X_te_gpu, y_te)]:
            scores_zs = X_gpu @ torch.from_numpy(text_embeddings).float().to(device)
            yhat = torch.argmax(scores_zs, dim=1).cpu().numpy()
            top1 = (yhat == y).mean()
            bal = balanced_acc(y, yhat, K)
            top5 = topk_acc(y, scores_zs.cpu().numpy(), 5)
            macro = f1_score(y, yhat, average='macro')
            records.append((0, "zero-shot", 0.0, split_name, backbone, top1, top5, bal, macro))

    shot_list = sorted([s for s in args.shots if s > 0])
    shot_iter = shot_list
    if show_progress and sys.stdout.isatty():
        shot_iter = tqdm(shot_list, desc=f"shots ({backbone})", leave=False, miniters=1)

    if show_progress and sys.stdout.isatty():
        print("~~~~~~~~~~~~~~few shot")
    for shot in shot_iter:
        rng = np.random.default_rng(args.seed + int(shot))
        sup_idx, sup_labels = sample_few_shot_indices(y_tr, K, shot, rng)
        X_sup, y_sup = X_tr[sup_idx], sup_labels
        # Normalize the few-shot subset (safe redundant normalization)
        X_sup = X_sup / (np.linalg.norm(X_sup, axis=1, keepdims=True) + 1e-8)


        # Prototype (standard)
        prototypes = prototype_classifier(X_sup, y_sup, K)
        for split_name, X_gpu, y in [("val", X_vl_gpu, y_vl), ("test", X_te_gpu, y_te)]:
            scores = X_gpu @ torch.from_numpy(prototypes).float().to(device).T
            yhat = torch.argmax(scores, axis=1).cpu().numpy()
            top1 = (yhat == y).mean()
            bal = balanced_acc(y, yhat, K)
            top5 = topk_acc(y, scores.cpu().numpy(), 5)
            macro = f1_score(y, yhat, average='macro')
            records.append((shot, "prototype", 1.0, split_name, backbone, top1, top5, bal, macro))

        # --- Prompt-aware prototype sweep ---
        if args.use_prompts_for_prototype:
            if text_embs_bundle is None:
                raise RuntimeError("Prompt embeddings not available for prototype fusion.")
            best_alpha, best_acc = None, -1.0
            for alpha in args.alpha_grid:
                prototypes_prompts = prototype_classifier_with_prompts(
                    X_sup, y_sup, K, list(label_names), text_embs_bundle, alpha=alpha)
                split_metrics = []
                for split_name, X_gpu, y in [("val", X_vl_gpu, y_vl), ("test", X_te_gpu, y_te)]:
                    scores = X_gpu @ torch.from_numpy(prototypes_prompts).float().to(device).T
                    yhat = torch.argmax(scores, axis=1).cpu().numpy()
                    top1 = (yhat == y).mean()
                    bal = balanced_acc(y, yhat, K)
                    top5 = topk_acc(y, scores.cpu().numpy(), 5)
                    macro = f1_score(y, yhat, average='macro')
                    records.append((shot, f"prototype+prompts(alpha={alpha:.2f})",alpha, split_name,
                                    backbone, top1, top5, bal, macro))
                    if split_name == "val":
                        split_metrics.append(top1)
                mean_val = np.mean(split_metrics) if split_metrics else 0
                if mean_val > best_acc:
                    best_acc, best_alpha = mean_val, alpha
            if best_alpha is not None and show_progress and sys.stdout.isatty():
                print(f"[{backbone}] best alpha for prototype={best_alpha:.2f} (val acc={best_acc:.3f})")


        # Linear probe (standard)
        if show_progress and sys.stdout.isatty():
            print(f"[{backbone}] training linear probe for {shot}-shot...")
        model_lin = train_linear_probe(X_sup, y_sup, X_val=None, y_val=None,
                                       epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                       device=args.device, seed=args.seed + int(shot), compile_model=args.no_compile,
                                       show_progress=show_progress)
        model_lin.eval()
        with torch.no_grad():
            logits_vl = model_lin(torch.from_numpy(X_vl).float().to(args.device)).cpu().numpy()
            logits_te = model_lin(torch.from_numpy(X_te).float().to(args.device)).cpu().numpy()
        for split_name, logits, y in [("val", logits_vl, y_vl), ("test", logits_te, y_te)]:
            yhat = logits.argmax(axis=1)
            top1 = (yhat == y).mean()
            bal = balanced_acc(y, yhat, K)
            top5 = topk_acc(y, logits, 5)
            macro = f1_score(y, yhat, average='macro')
            records.append((shot, "linear", 1.0, split_name, backbone, top1, top5, bal, macro))

        # --- Prompt-aware linear probe sweep ---
        if args.use_prompts_for_linear:
            if text_embs_bundle is None:
                raise RuntimeError("Prompt embeddings not available for linear probe.")
            best_alpha, best_acc = None, -1.0
            for alpha in args.alpha_grid:
                # we reuse alpha to weight text pseudo-examples (optional, could modulate augmentation ratio)
                model_lin_p = train_linear_probe_with_prompts(
                    X_sup, y_sup, list(label_names), text_embs_bundle, K, 
                    alpha=alpha, epochs=args.epochs, lr=args.lr, 
                    batch_size=args.batch_size, device=args.device, 
                    seed=args.seed + int(shot), compile_model=args.no_compile, 
                    show_progress=show_progress)
                model_lin_p.eval()
                with torch.no_grad():
                    logits_vl_p = model_lin_p(torch.from_numpy(X_vl).float().to(args.device)).cpu().numpy()
                    logits_te_p = model_lin_p(torch.from_numpy(X_te).float().to(args.device)).cpu().numpy()
                val_acc = (logits_vl_p.argmax(axis=1) == y_vl).mean()
                if val_acc > best_acc:
                    best_acc, best_alpha = val_acc, alpha
                    best_logits_vl, best_logits_te = logits_vl_p, logits_te_p
                for split_name, logits, y in [("val", logits_vl_p, y_vl), ("test", logits_te_p, y_te)]:
                    yhat = logits.argmax(axis=1)
                    top1 = (yhat == y).mean()
                    bal = balanced_acc(y, yhat, K)
                    top5 = topk_acc(y, logits, 5)
                    macro = f1_score(y, yhat, average='macro')
                    records.append((shot, f"linear+prompts(alpha={alpha:.2f})", alpha, split_name,
                                    backbone, top1, top5, bal, macro))
            if best_alpha is not None and show_progress and sys.stdout.isatty():
                print(f"[{backbone}] best alpha for linear probe={best_alpha:.2f} (val acc={best_acc:.3f})")


    return records


# ------------------------------
# Direct on-the-fly evaluation (prompt pooling & prototypes built from model)
# ------------------------------
def build_few_shot_prototypes_direct(labels: List[str], train_csv: str, shots: int, model, preprocess, device: str,
                                    cache_dir: str = DEFAULT_CACHE_DIR, root: str = ".", random_state: int = 42,
                                    show_progress: bool = False):
    df_train = pd.read_csv(train_csv)
    try:
        with torch.no_grad():
            emb = model.encode_image(torch.zeros((1, 3, 224, 224), device=device))
        emb_dim = emb.shape[1]
    except Exception:
        emb_dim = 512

    protos = {}
    label_list = list(labels)
    iterator = label_list
    if show_progress and sys.stdout.isatty():
        iterator = tqdm(label_list, desc="build_prototypes", leave=False, miniters=1)

    for lbl in iterator:
        df_lbl = df_train[df_train["label"] == lbl]
        if df_lbl.empty:
            protos[lbl] = torch.zeros(emb_dim)
            continue
        n = min(shots, len(df_lbl))
        df_samp = df_lbl.sample(n=n, random_state=random_state)
        img_paths = df_samp["filepath"].tolist()
        embs = encode_images(model, preprocess, img_paths, device, cache_prefix=f"proto_{lbl}_{shots}", cache_dir=cache_dir, root=root, show_progress=show_progress)
        embs = [e for e in embs if e is not None]
        if not embs:
            protos[lbl] = torch.zeros(emb_dim)
            continue
        proto = torch.stack(embs).mean(dim=0)
        protos[lbl] = F.normalize(proto, dim=-1).cpu()
    return protos


def evaluate_prompts_and_few_shot_direct(labels: List[str], train_csv: str, split_csv: str, shots=(1, 5, 10),
                                         model_name: str = "ViT-B-32", device: str = "cuda",
                                         num_samples: Optional[int] = None, temp: float = 0.12,
                                         delta_prompts_path: Optional[str] = None, pretrained: str = "openai",
                                         random_state: int = 42, cache_dir: str = DEFAULT_CACHE_DIR, root: str = ".",
                                         show_progress: bool = False):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    try:
        model.encode_image = torch.compile(model.encode_image, mode="max-autotune")
        model.encode_text = torch.compile(model.encode_text, mode="max-autotune")
    except Exception:
        pass

    df = pd.read_csv(split_csv)
    if num_samples:
        df = df.sample(frac=1, random_state=random_state).head(num_samples)
    image_paths = df["filepath"].tolist()
    true_labels = df["label"].tolist()

    image_embeddings = encode_images(model, preprocess, image_paths, device, cache_prefix="direct_test_images", cache_dir=cache_dir, root=root, show_progress=show_progress)

    label_prompts = load_prompts(delta_prompts_path or DEFAULT_PROMPTS_JSON, labels)
    prompt_bundle = get_text_embeddings_bundle_from_clip(model_name, model, tokenizer, label_prompts, device=device, show_progress=show_progress)

    # mean prompt embeddings
    delta_mean = {lbl: prompt_bundle[lbl]["label_embedding"] for lbl in labels}
    preds_delta_mean = score_all_vectorized(image_embeddings, delta_mean, labels)
    acc_delta_mean = acc_from_preds(preds_delta_mean, true_labels)

    # prompt pooling
    # preds_delta_pool = []
    # for img_emb in image_embeddings:
    #     best_label, best_score = None, -1e9
    #     for lbl, info in prompt_bundle.items():
    #         pe = info["prompt_embeddings"]
    #         if pe.numel() == 0:
    #             continue
    #         sims = torch.cosine_similarity(img_emb.unsqueeze(0), pe.to(img_emb.dtype), dim=1)
    #         w = torch.softmax(sims / temp, dim=0)
    #         pooled = (w * sims).sum().item()
    #         if pooled > best_score:
    #             best_label, best_score = lbl, pooled
    #     preds_delta_pool.append(best_label)
    # acc_delta_pool = acc_from_preds(preds_delta_pool, true_labels)
    # --- Vectorized prompt-pooling (GPU-accelerated) ---
    img_mat = torch.stack(image_embeddings).to(device)
    labels = list(prompt_bundle.keys())
    all_prompts = [info["prompt_embeddings"].to(device) for info in prompt_bundle.values()]
    # Precompute pooled prompt embeddings (mean or softmax-weighted)
    pooled_prompts = []
    for pe in all_prompts:
        if pe.numel() == 0:
            pooled_prompts.append(torch.zeros_like(all_prompts[0][0]))
        else:
            sims = F.normalize(pe, dim=-1)
            pooled_prompts.append(sims.mean(dim=0))
    label_mat = torch.stack(pooled_prompts)
    # Compute all cosine similarities at once
    sims = F.normalize(img_mat, dim=-1) @ F.normalize(label_mat, dim=-1).T
    preds_idx = sims.argmax(dim=1).cpu().numpy()
    preds_delta_pool = [labels[i] for i in preds_idx]
    acc_delta_pool = acc_from_preds(preds_delta_pool, true_labels)


    # few-shot prototypes built via model
    few_shot_results = {}
    for k in shots:
        proto_dict = build_few_shot_prototypes_direct(labels, train_csv, k, model, preprocess, device, cache_dir=cache_dir, root=root, random_state=random_state, show_progress=show_progress)
        preds_proto = score_all_vectorized(image_embeddings, proto_dict, labels)
        acc_proto = acc_from_preds(preds_proto, true_labels)
        few_shot_results[int(k)] = acc_proto

    # linear probe baseline
    df_train = pd.read_csv(train_csv)
    X_train, y_train = [], []
    for lbl in labels:
        df_lbl = df_train[df_train["label"] == lbl]
        if df_lbl.empty:
            continue
        df_samp = df_lbl.sample(n=min(max(shots), len(df_lbl)), random_state=random_state)
        img_paths = df_samp["filepath"].tolist()
        embs = encode_images(model, preprocess, img_paths, device, cache_prefix=f"linear_{lbl}", cache_dir=cache_dir, root=root, show_progress=show_progress)
        embs = [e for e in embs if e is not None]
        X_train.extend(embs)
        y_train.extend([lbl] * len(embs))

    if len(X_train) == 0:
        acc_linear = None
    else:
        X_train = torch.stack(X_train).to(device)
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        y_train_idx = torch.tensor([label_to_idx[y] for y in y_train], device=device)
        linear_head = nn.Linear(X_train.shape[1], len(labels)).to(device)
        optimizer = torch.optim.Adam(linear_head.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        epoch_iter = range(5)
        if show_progress and sys.stdout.isatty():
            epoch_iter = tqdm(epoch_iter, desc="linear_probe_epochs", leave=False, miniters=1)
        for _ in epoch_iter:
            optimizer.zero_grad()
            logits = linear_head(X_train)
            loss = loss_fn(logits, y_train_idx)
            loss.backward()
            optimizer.step()
        X_test = torch.stack([e for e in image_embeddings if e is not None]).to(device)
        with torch.no_grad():
            logits = linear_head(X_test)
            preds_idx = torch.argmax(logits, dim=1).cpu().tolist()
        preds_linear = [labels[i] for i in preds_idx]
        acc_linear = acc_from_preds(preds_linear, true_labels)

    return {
        "n_samples": len(image_embeddings),
        "delta_mean": acc_delta_mean,
        "delta_pool": acc_delta_pool,
        "few_shot_results": few_shot_results,
        "linear_probe": acc_linear,
    }


# ------------------------------
# CLI parsing & main
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Unified few-shot + zero-shot evaluation for CLIP-style models (prompt-aware few-shot extensions)")
    ap.add_argument("--data-root", default=DATA_ROOT, help="Path to dataset root")
    ap.add_argument("--train-csv", default=TRAIN_CSV, help="Training CSV file")
    ap.add_argument("--val-csv", default=VAL_CSV, help="Validation CSV file")
    ap.add_argument("--test-csv", default=TEST_CSV, help="Test CSV file")
    ap.add_argument("--labels", default=LABELS, help="TSV with id\tlabel_name")
    ap.add_argument("--prompts-path", default=DEFAULT_PROMPTS_JSON)
    ap.add_argument("--backbones", nargs="+",default=[b for b in DEFAULT_PRETRAINED.keys()], help="Model backbones to use (default uses DEFAULT_BACKBONES values).")    
    # ap.add_argument("--pretrained", nargs="+", default=None, help="Optional list of pretrained names matching backbones (e.g. openai, dfn5b). If not set, 'openai' will be used for all.")
    ap.add_argument("--pretrained",nargs="+", default=[DEFAULT_PRETRAINED.get(b, "openai") for b in DEFAULT_PRETRAINED.keys()], help="Pretrained weights for each backbone (default uses DEFAULT_PRETRAINED map).")
    ap.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    ap.add_argument("--save-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--direct-eval", action="store_true", help="Run direct prompt-based evaluation (encodes images on the fly).")
    ap.add_argument("--num-samples", type=int, default=None, help="If using direct-eval, optionally subsample test set.")
    ap.add_argument("--temp", type=float, default=0.12, help="Temp for prompt pooling softmax")
    ap.add_argument("--no-compile", action="store_false", help="Dont attempt torch.compile() for speed where possible")
    ap.add_argument("--max-workers", type=int, default=8)

    # Prompt-aware few-shot options
# replace in parse_args()

# Prompt-aware few-shot options (now defaults ON)
    ap.add_argument("--no-prompts-for-prototype", action="store_true", help="Disable text-prompt fusion for prototypes")
    ap.add_argument("--no-prompts-for-linear", action="store_true", help="Disable text-prompt pseudo-examples for linear probe")
    ap.add_argument("--alpha-grid", type=str, default=ALPHAS, help="Comma-separated list of prompt-alpha values to sweep over")

    args = ap.parse_args()

    # ---- 👇 add these right after parsing ----
    args.use_prompts_for_prototype = not args.no_prompts_for_prototype
    args.use_prompts_for_linear = not args.no_prompts_for_linear
    # args.alpha_grid = [float(x) for x in args.alpha_grid.split(",")]
    # ------------------------------------------

    return args


def main():
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # map pretrained names to backbones
    # if args.pretrained:
    #     if len(args.pretrained) != len(args.backbones):
    #         print("[WARN] --pretrained length does not match --backbones; using 'openai' for missing entries")
    #     pretrained_map = {b: args.pretrained[i] if i < len(args.pretrained) else "openai" for i, b in enumerate(args.backbones)}
    # else:
    #     pretrained_map = {b: "openai" for b in args.backbones}
    # --- NEW default-pretrained handling ---
    pretrained_map = {}

    for b in args.backbones:
        # Case 1: user provided --pretrained list → use that first
        if hasattr(args, "pretrained") and args.pretrained and len(args.pretrained) > 0:
            idx = args.backbones.index(b)
            pretrained_map[b] = args.pretrained[idx] if idx < len(args.pretrained) else \
                                DEFAULT_PRETRAINED.get(b, "openai")
        else:
            # Case 2: no user override → use DEFAULT_PRETRAINED or fallback to openai
            pretrained_map[b] = DEFAULT_PRETRAINED.get(b, "openai")
    # Optional: log for debugging
    # if sys.stdout.isatty():
    #     print("🔧 Using pretrained mappings:")
    #     for b, p in pretrained_map.items():
    #         print(f"  {b:35s} → {p}")
    args.pretrained = pretrained_map

    # print("~~~~~~~~~~~~~~loadin labels")
    label_names, name2id = load_labels(args.labels)
    K = len(label_names)

    # decide whether to show progress bars (TTY) - nested bars will be enabled only in TTY mode
    show_progress = sys.stdout.isatty()

    records_all = []

    if args.direct_eval:
        # direct eval per backbone; nested progress across backbones and shots
        backbone_iter = args.backbones
        if show_progress:
            backbone_iter = tqdm(backbone_iter, desc="backbones (direct)", leave=True, miniters=1)
        for backbone in backbone_iter:
            res = evaluate_prompts_and_few_shot_direct(
                labels=list(label_names),
                train_csv=args.train_csv,
                split_csv=args.test_csv,
                shots=args.shots,
                model_name=backbone,
                device=args.device,
                num_samples=args.num_samples,
                temp=args.temp,
                delta_prompts_path=args.prompts_path,
                pretrained=args.pretrained.get(backbone, "openai"),
                random_state=args.seed,
                cache_dir=args.save_dir,
                root=args.data_root,
                show_progress=show_progress
            )
            outp = Path(args.results_dir) / f"direct_eval_{backbone.replace('/', '_')}.json"
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
    else:
        backbone_iter = args.backbones
        if show_progress:
            backbone_iter = tqdm(backbone_iter, desc="backbones", leave=True, miniters=1)
        for backbone in backbone_iter:
            recs = evaluate_backbone(backbone, args, list(label_names), K, show_progress=show_progress)
            records_all.extend(recs)

        out_csv = Path(args.results_dir) / "few_shot_table_all_backbones.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["shot", "model", "alpha", "split", "backbone", "top1", "top5", "balanced_acc", "macro_f1"])
            for r in records_all:
                w.writerow(r)
        if show_progress:
            print(f"Wrote results -> {out_csv}")
        else:
            print(f"Wrote results -> {out_csv}")


if __name__ == "__main__":
    main()
