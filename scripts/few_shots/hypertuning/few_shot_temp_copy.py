#!/usr/bin/env python3
"""
Modified test_few_shot script with hyperparameter sweep across:
 alpha, temp, lr, wds

- CPU tasks (prototype alpha sweep, prompt-aware linear probe with CPU training)
  are parallelized with ProcessPoolExecutor.
- GPU tasks (standard linear probe trained on args.device == 'cuda') are run sequentially
  to avoid GPU contention (single GPU available).
- Outputs CSV with columns alpha, temp, lr, wds (lr/wds set to 0 for methods that don't use them).
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
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
DEFAULT_TEMPS = [0.01, 0.05, 0.1, 0.2]
DEFAULT_LRS = [1e-4, 3e-4, 1e-3]
DEFAULT_WDS = [0.0, 1e-5, 1e-4]

DATA_ROOT = "/home/c/dkorot/AI4GOOD/provided_dir/datasets/mushroom/merged_dataset"
TRAIN_CSV = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/train.csv"
VAL_CSV = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/val.csv"
TEST_CSV = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/test.csv"
LABELS = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/data_prompts_label/labels.tsv"

with open('/home/c/dkorot/AI4GOOD/ai4good-mushroom/data_prompts_label/best_alphas.json', 'r') as f:
    BEST_ALPHAS = json.load(f)


import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)


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
                       epochs: int = 200, lr: float = 1e-3, batch_size: int = 32, device: str = "cpu", seed: int = 42, temp=1.0,
                       compile_model: bool = False, show_progress: bool = False, weight_decay: float = 1e-4) -> nn.Module:
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            # logits = model(xb)
            # loss = loss_fn(logits, yb)
            logits = model(xb) / float(temp)
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


def train_linear_probe_with_prompts(
    X_support, y_support, label_names, text_embs, K, alpha=0.5,
    epochs=200, lr=1e-3, batch_size=32, device="cpu", temp=1.0,
    seed=42, compile_model=False, show_progress=False, weight_decay: float = 1e-4):

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
        device=device, seed=seed, temp=temp,
        compile_model=compile_model, show_progress=show_progress, weight_decay=weight_decay)
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
# New helper functions for CPU-parallel alpha evaluation & extended sweeps
# ------------------------------
def _serialize_text_embs_to_numpy(text_embs_bundle):
    # Convert text_embs_bundle (which may contain torch tensors) into numpy arrays
    out = {}
    for k, v in text_embs_bundle.items():
        le = v.get("label_embedding", None)
        if isinstance(le, torch.Tensor):
            out[k] = {"label_embedding": le.cpu().numpy()}
        elif le is None:
            out[k] = {"label_embedding": None}
        else:
            out[k] = {"label_embedding": np.array(le)}
    return out

def eval_prototype_alpha_serial(alpha, X_sup_np, y_sup_np, label_names_list, text_embs_numpy, K):
    """
    CPU-only evaluation of prototype fusion for a single alpha.
    Returns: alpha, protos (K x D)
    """
    protos = np.zeros((K, X_sup_np.shape[1]), dtype=np.float32)
    for k in range(K):
        inds = np.where(y_sup_np == k)[0]
        if len(inds) == 0:
            img_proto = np.zeros((X_sup_np.shape[1],), dtype=np.float32)
        else:
            img_proto = X_sup_np[inds].mean(axis=0).astype(np.float32)
            n = np.linalg.norm(img_proto)
            if n > 0:
                img_proto = img_proto / n
        lbl = label_names_list[k]
        txt = text_embs_numpy.get(lbl, {}).get("label_embedding", None)
        if txt is None:
            txt_proto = np.zeros_like(img_proto)
        else:
            txt_proto = np.array(txt, dtype=np.float32)
            n = np.linalg.norm(txt_proto)
            if n > 0:
                txt_proto = txt_proto / n
        combined = alpha * img_proto + (1.0 - alpha) * txt_proto
        n = np.linalg.norm(combined)
        if n > 0:
            combined = combined / n
        protos[k] = combined
    return alpha, protos

def eval_linear_alpha_lr_wd_serial(alpha, temp, lr, wds, X_sup_np, y_sup_np, label_names_list, text_embs_numpy, K,
                                   X_vl_np, y_vl_np, X_te_np, y_te_np, base_seed, batch_size, epochs):
    """
    CPU-only evaluation of linear probe trained with prompts for a single (alpha, lr, wds).
    Returns alpha, lr, wds, val_acc, test_acc, logits_vl, logits_te
    """
    # rebuild text_embs_for_train structure expected by train_linear_probe_with_prompts
    text_embs_for_train = {}
    for lbl in label_names_list:
        arr = text_embs_numpy.get(lbl, {}).get("label_embedding", None)
        if arr is None:
            vec = torch.zeros((X_sup_np.shape[1],), dtype=torch.float32)
        else:
            vec = torch.from_numpy(np.array(arr, dtype=np.float32))
        text_embs_for_train[lbl] = {"label_embedding": vec}

    model_lin_p = train_linear_probe_with_prompts(
        X_sup_np.astype(np.float32), y_sup_np.astype(np.int64),
        label_names_list, text_embs_for_train, K,
        alpha=alpha, epochs=epochs, lr=lr, batch_size=batch_size,
        device="cpu", seed=base_seed, compile_model=False, show_progress=False, weight_decay=wds, temp=temp)
    )
    model_lin_p.eval()
    with torch.no_grad():
        # logits_vl_p = model_lin_p(torch.from_numpy(X_vl_np).float()).cpu().numpy()
        # logits_te_p = model_lin_p(torch.from_numpy(X_te_np).float()).cpu().numpy()
        logits_vl_p = (model_lin_p(torch.from_numpy(X_vl_np).float()) / float(temp)).cpu().numpy()
        logits_te_p = (model_lin_p(torch.from_numpy(X_te_np).float()) / float(temp)).cpu().numpy()


    val_acc = (logits_vl_p.argmax(axis=1) == y_vl_np).mean()
    test_acc = (logits_te_p.argmax(axis=1) == y_te_np).mean()
    return alpha, lr, wds, val_acc, test_acc, logits_vl_p, logits_te_p

def score_prototypes_vectorized(X_gpu, protos_t, temps):
    """
    X_gpu: (N, D) torch.float32 on device
    protos_t: (K, D) torch.float32 on device
    temps: list/array of temperatures to apply

    Returns:
        dict[temp] -> logits tensor (N, K)
    """
    base_scores = X_gpu @ protos_t.T  # (N, K)
    out = {}
    for t in temps:
        out[t] = base_scores / float(t)
    return out

def evaluate_prototypes_with_alpha_temp(
    protos_by_alpha,                    # dict[alpha] = np.array(K,D)
    X_vl_gpu, y_vl,
    X_te_gpu, y_te,
    temps,                              # args.temp_grid
    K,
    shot, backbone,
    records,
    show_progress: bool = False
):
    device = X_vl_gpu.device

    for temp_val in temps:
        best_alpha, best_acc = None, -1.0
        for alpha_val, protos in protos_by_alpha.items():
            # move prototypes to GPU
            protos_t = torch.from_numpy(protos).float().to(device)

            # vectorized scoring for val & test
            scores_dict_val = score_prototypes_vectorized(X_vl_gpu, protos_t, [temp_val])
            scores_dict_te = score_prototypes_vectorized(X_te_gpu, protos_t, [temp_val])

            # --- Validation ---
            scores_val = scores_dict_val[temp_val]
            yhat_val = scores_val.argmax(dim=1).cpu().numpy()
            val_top1 = (yhat_val == y_vl).mean()
            val_bal = balanced_acc(y_vl, yhat_val, K)
            val_top5 = topk_acc(y_vl, scores_val.cpu().numpy(), 5)
            val_m = f1_score(y_vl, yhat_val, average='macro')

            # Log validation row
            records.append((
                shot,
                f"prototype+prompts(alpha={alpha_val:.2f})",
                alpha_val,
                temp_val,
                0.0,
                0.0,
                "val",
                backbone,
                val_top1,
                val_top5,
                val_bal,
                val_m
            ))

            # Track best alpha per temp
            if val_top1 > best_acc:
                best_acc = val_top1
                best_alpha = alpha_val

            # --- Test ---
            scores_te = scores_dict_te[temp_val]
            yhat_te = scores_te.argmax(dim=1).cpu().numpy()
            top1 = (yhat_te == y_te).mean()
            bal = balanced_acc(y_te, yhat_te, K)
            top5 = topk_acc(y_te, scores_te.cpu().numpy(), 5)
            macro = f1_score(y_te, yhat_te, average='macro')

            # Log test row
            records.append((
                shot,
                f"prototype+prompts(alpha={alpha_val:.2f})",
                alpha_val,
                temp_val,
                0.0,
                0.0,
                "test",
                backbone,
                top1,
                top5,
                bal,
                macro
            ))

        # Optional progress printing

        if best_alpha is not None and show_progress and sys.stdout.isatty():
            print(f"[{backbone}] shot={shot} temp={temp_val:.3f} → best alpha={best_alpha:.2f} (val acc={best_acc:.3f})")
            print(f"[{backbone}] best alpha for prototype={best_alpha:.2f} (val acc={best_acc:.3f})")

def score_logits_with_temps(logits, temps):
    """
    logits: (N, K) torch tensor (on CPU or GPU)
    temps: list of float temperature values

    Returns:
        dict[temp_val] → (N, K) numpy array
    """
    out = {}
    for t in temps:
        out[t] = (logits / float(t)).cpu().numpy()
    return out

def run_standard_linear_probe_sweep(
    X_sup, y_sup,
    X_vl, y_vl,
    X_te, y_te,
    lr_grid, wd_grid, temp_grid,
    K, backbone, shot,
    device, seed, batch_size, epochs,
    compile_model,
    records,
):
    """
    Sweep over (lr, wd, temp) for the standard linear probe.
    Adds rows directly to the records list.
    """

    for lr_val in lr_grid:
        for wd_val in wd_grid:

            # Train the model ONCE per (lr, wd)
            model_lin = train_linear_probe(
                X_support=X_sup,
                y_support=y_sup,
                X_val=None,
                y_val=None,
                lr=float(lr_val),
                weight_decay=float(wd_val),
                temp=1.0,                  # temp only for TRAINING logits
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                seed=seed,
                compile_model=compile_model,
                show_progress=False,
            )
            model_lin.eval()

            # Precompute raw logits ONCE
            with torch.no_grad():
                logits_vl = model_lin(torch.from_numpy(X_vl).float().to(device))
                logits_te = model_lin(torch.from_numpy(X_te).float().to(device))

            # Now sweep temperatures efficiently
            val_scores_by_temp = score_logits_with_temps(logits_vl, temp_grid)
            te_scores_by_temp  = score_logits_with_temps(logits_te, temp_grid)

            for temp_val in temp_grid:

                # --- Val split ---
                logits = val_scores_by_temp[temp_val]
                yhat = logits.argmax(axis=1)
                top1 = (yhat == y_vl).mean()
                bal  = balanced_acc(y_vl, yhat, K)
                top5 = topk_acc(y_vl, logits, 5)
                macro= f1_score(y_vl, yhat, average="macro")

                records.append((
                    shot, "linear",
                    1.0, temp_val,
                    float(lr_val), float(wd_val),
                    "val", backbone,
                    top1, top5, bal, macro
                ))

                # --- Test split ---
                logits = te_scores_by_temp[temp_val]
                yhat = logits.argmax(axis=1)
                top1 = (yhat == y_te).mean()
                bal  = balanced_acc(y_te, yhat, K)
                top5 = topk_acc(y_te, logits, 5)
                macro= f1_score(y_te, yhat, average="macro")

                records.append((
                    shot, "linear",
                    1.0, temp_val,
                    float(lr_val), float(wd_val),
                    "test", backbone,
                    top1, top5, bal, macro
                ))

def eval_linear_prompt_worker(
    alpha, temp_val, lr_val, wd_val,
    X_sup_np, y_sup_np,
    label_names_list, text_embs_numpy,
    K,
    X_vl_np, y_vl_np,
    X_te_np, y_te_np,
    seed, batch_size, epochs
):
    """
    Worker run on CPU. Returns:
    (alpha, temp_val, lr_val, wd_val, val_logits, test_logits)
    """

    # Build torch text embeddings for the worker
    text_embs_for_train = {
        lbl: {
            "label_embedding": torch.from_numpy(text_embs_numpy[lbl]["label_embedding"]).float()
        }
        for lbl in label_names_list
    }

    model_lin = train_linear_probe_with_prompts(
        X_sup_np, y_sup_np,
        label_names_list, text_embs_for_train, K,
        alpha=alpha,
        epochs=epochs,
        lr=lr_val,
        batch_size=batch_size,
        device="cpu",
        seed=seed,
        temp=1.0,                       # training temperature fixed
        compile_model=False,
        show_progress=False,
        weight_decay=wd_val,
    )

    model_lin.eval()
    with torch.no_grad():
        vl_logits = model_lin(torch.from_numpy(X_vl_np).float()) / float(temp_val)
        te_logits = model_lin(torch.from_numpy(X_te_np).float()) / float(temp_val)

    return alpha, temp_val, lr_val, wd_val, vl_logits.numpy(), te_logits.numpy()

def run_prompt_linear_probe_sweep(
    protos_alpha_list,
    X_sup_np, y_sup_np,
    X_vl_np, y_vl_np,
    X_te_np, y_te_np,
    label_names_list, text_embs_numpy,
    lr_grid, wd_grid, temp_grid,
    K, backbone, shot, seed, batch_size, epochs, max_workers,
    records
):
    combos = [
        (alpha, temp_val, lr_val, wd_val)
        for alpha in protos_alpha_list
        for temp_val in temp_grid
        for lr_val in lr_grid
        for wd_val in wd_grid
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as pool:

        futures = {
            pool.submit(
                eval_linear_prompt_worker,
                alpha, temp_val, lr_val, wd_val,
                X_sup_np, y_sup_np,
                label_names_list, text_embs_numpy,
                K,
                X_vl_np, y_vl_np,
                X_te_np, y_te_np,
                seed, batch_size, epochs
            ): (alpha, temp_val, lr_val, wd_val)
            for (alpha, temp_val, lr_val, wd_val) in combos
        }

        for fut in as_completed(futures):
            alpha, temp_val, lr_val, wd_val = futures[fut]
            try:
                (a, t_val, lr_r, wd_r,
                 logits_vl, logits_te) = fut.result()

            except Exception as e:
                print(f"[ERROR] worker failed for alpha={alpha} temp={temp_val} lr={lr_val} wd={wd_val} : {e}")
                continue

            # --- Log VAL ---
            yhat = logits_vl.argmax(axis=1)
            top1 = (yhat == y_vl_np).mean()
            bal  = balanced_acc(y_vl_np, yhat, K)
            top5 = topk_acc(y_vl_np, logits_vl, 5)
            macro= f1_score(y_vl_np, yhat, average='macro')

            records.append((
                shot,
                f"linear+prompts(alpha={a:.2f})",
                a, t_val, lr_r, wd_r,
                "val", backbone,
                top1, top5, bal, macro
            ))

            # --- Log TEST ---
            yhat = logits_te.argmax(axis=1)
            top1 = (yhat == y_te_np).mean()
            bal  = balanced_acc(y_te_np, yhat, K)
            top5 = topk_acc(y_te_np, logits_te, 5)
            macro= f1_score(y_te_np, yhat, average='macro')

            records.append((
                shot,
                f"linear+prompts(alpha={a:.2f})",
                a, t_val, lr_r, wd_r,
                "test", backbone,
                top1, top5, bal, macro
            ))


# ------------------------------
# High-level per-backbone evaluation (cached features) with nested progress bars
# ------------------------------
def evaluate_backbone(backbone: str, args: argparse.Namespace, label_names: List[str], K: int, show_progress: bool = False):
    model_pretrained = args.pretrained.get(backbone, "openai")
    if show_progress and sys.stdout.isatty():
        print(f"Evaluating backbone: {backbone} (pretrained={model_pretrained})")

    X_tr, y_tr, _ = ensure_features("train", backbone, args.data_root, args.train_csv, args.labels, save_dir=args.save_dir, pretrained=model_pretrained)
    X_vl, y_vl, _ = ensure_features("val", backbone, args.data_root, args.val_csv, args.labels, save_dir=args.save_dir, pretrained=model_pretrained)
    X_te, y_te, _ = ensure_features("test", backbone, args.data_root, args.test_csv, args.labels, save_dir=args.save_dir, pretrained=model_pretrained)

    # Normalize all embeddings to unit length (important for cosine-space CLIP)
    X_tr = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-8)
    X_vl = X_vl / (np.linalg.norm(X_vl, axis=1, keepdims=True) + 1e-8)
    X_te = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-8)


    # Zero-shot text embeddings (for zero-shot only)
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

    # Zero-shot
    if 0 in args.shots:
        for split_name, X_gpu, y in [("val", X_vl_gpu, y_vl), ("test", X_te_gpu, y_te)]:
            scores_zs = X_gpu @ torch.from_numpy(text_embeddings).float().to(device)
            yhat = torch.argmax(scores_zs, dim=1).cpu().numpy()
            top1 = (yhat == y).mean()
            bal = balanced_acc(y, yhat, K)
            top5 = topk_acc(y, scores_zs.cpu().numpy(), 5)
            macro = f1_score(y, yhat, average='macro')
            # zero-shot has no lr/wds, and typically no alpha/temp (we include temp as args.temp if user supplied; else 0)
            temp_val = args.temp if args.temp is not None else 0.0
            records.append((0, "zero-shot", 0.0, float(temp_val), 0.0, 0.0, split_name, backbone, top1, top5, bal, macro))

    shot_list = sorted([s for s in args.shots if s > 0])
    shot_iter = shot_list
    if show_progress and sys.stdout.isatty():
        shot_iter = tqdm(shot_list, desc=f"shots ({backbone})", leave=False, miniters=1)

    for shot in shot_iter:
        rng = np.random.default_rng(args.seed + int(shot))
        sup_idx, sup_labels = sample_few_shot_indices(y_tr, K, shot, rng)
        X_sup, y_sup = X_tr[sup_idx], sup_labels
        # Normalize the few-shot subset (safe redundant normalization)
        X_sup = X_sup / (np.linalg.norm(X_sup, axis=1, keepdims=True) + 1e-8)

        # Prototype (standard) -- no lr/wds applicable (record 0)
        prototypes = prototype_classifier(X_sup, y_sup, K)
        for split_name, X_gpu, y in [("val", X_vl_gpu, y_vl), ("test", X_te_gpu, y_te)]:
            # scores = X_gpu @ torch.from_numpy(prototypes).float().to(device).T
            scores = (X_gpu @ torch.from_numpy(prototypes).float().to(device).T) / float(args.temp)
            yhat = torch.argmax(scores, axis=1).cpu().numpy()
            top1 = (yhat == y).mean()
            bal = balanced_acc(y, yhat, K)
            top5 = topk_acc(y, scores.cpu().numpy(), 5)
            macro = f1_score(y, yhat, average='macro')
            # records.append((shot, "prototype", 1.0, 0.0, 0.0, 0.0, split_name, backbone, top1, top5, bal, macro))
            records.append((shot, "prototype", 1.0, float(args.temp), 0.0, 0.0, split_name, backbone, top1, top5, bal, macro))


        # --- Prompt-aware prototype sweep (parallel, CPU) over alpha_grid ---
        if args.use_prompts_for_prototype:
            if text_embs_bundle is None:
                raise RuntimeError("Prompt embeddings not available for prototype fusion.")

            text_embs_numpy = _serialize_text_embs_to_numpy(text_embs_bundle)
            label_names_list = list(label_names)
            X_sup_np = X_sup.astype(np.float32)
            y_sup_np = y_sup.astype(np.int64)

            protos_by_alpha = {}
            # parallelize across alphas (CPU)
            with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
                futures = {pool.submit(eval_prototype_alpha_serial, float(alpha), X_sup_np, y_sup_np, label_names_list, text_embs_numpy, K): float(alpha) for alpha in args.alpha_grid}
                for fut in as_completed(futures):
                    alpha = futures[fut]
                    try:
                        a, protos = fut.result()
                        protos_by_alpha[a] = protos
                    except Exception as e:
                        print(f"[{backbone}][prototype] alpha={alpha} failed: {e}", file=sys.stderr)
                        # log failed rows (val & test) with NaNs
                        records.append((shot, f"prototype+prompts(alpha={alpha:.2f})", alpha, float(0.0), 0.0, 0.0, 0.0, "val", backbone, np.nan, np.nan, np.nan, np.nan))
                        records.append((shot, f"prototype+prompts(alpha={alpha:.2f})", alpha, float(0.0), 0.0, 0.0, 0.0, "test", backbone, np.nan, np.nan, np.nan, np.nan))
                        continue

            # # Evaluate prototypes computed by workers
            # best_alpha, best_acc = None, -1.0
            # for alpha, protos in protos_by_alpha.items():
            #     protos_t = torch.from_numpy(protos).float().to(device)
            #     split_metrics = []
            #     for split_name, X_gpu, y in [("val", X_vl_gpu, y_vl), ("test", X_te_gpu, y_te)]:
            #         # scores = X_gpu @ protos_t.T
            #         scores = (X_gpu @ protos_t.T) / float(temper)
            #         yhat = torch.argmax(scores, axis=1).cpu().numpy()
            #         top1 = (yhat == y).mean()
            #         bal = balanced_acc(y, yhat, K)
            #         top5 = topk_acc(y, scores.cpu().numpy(), 5)
            #         macro = f1_score(y, yhat, average='macro')
            #         # prototype+prompts doesn't use lr/wds; record lr=0,wds=0 and temp=0
            #         records.append((shot, f"prototype+prompts(alpha={alpha:.2f})", alpha, float(temper), 0.0, 0.0, 0.0, split_name, backbone, top1, top5, bal, macro))
            #         if split_name == "val":
            #             split_metrics.append(top1)
            #     mean_val = np.mean(split_metrics) if split_metrics else 0
            #     if mean_val > best_acc:
            #         best_acc, best_alpha = mean_val, alpha
            evaluate_prototypes_with_alpha_temp(
                protos_by_alpha=protos_by_alpha,
                X_vl_gpu=X_vl_gpu,
                y_vl=y_vl,
                X_te_gpu=X_te_gpu,
                y_te=y_te,
                temps=args.temp_grid,
                K=K,
                shot=shot,
                backbone=backbone,
                records=records,
                show_progress=show_progress
            )


            # if best_alpha is not None and show_progress and sys.stdout.isatty():
            #     print(f"[{backbone}] best alpha for prototype={best_alpha:.2f} (val acc={best_acc:.3f})")

        # --- Standard linear probe (trained on args.device) sweeping lr & wds sequentially on GPU to avoid GPU contention ---
        if show_progress and sys.stdout.isatty():
            print(f"[{backbone}] training standard linear probe for {shot}-shot (sweeping lr/wds on device={args.device})...")
        # We'll loop over lr_grid and wds_grid sequentially (since training will run on the GPU if args.device=='cuda').
        for lr_val in args.lr_grid:
            for wds_val in args.wds_grid:
                for temper in args.temp_grid:
                    # train on device (may be GPU) - sequentially
                    model_lin = train_linear_probe(X_sup, y_sup, X_val=None, y_val=None,
                                                epochs=args.epochs, lr=float(lr_val), batch_size=args.batch_size,
                                                device=args.device, seed=args.seed + int(shot), compile_model=args.no_compile,
                                                show_progress=show_progress, weight_decay=float(wds_val), temp=temper)
                    model_lin.eval()
                    with torch.no_grad():
                        # logits_vl = model_lin(torch.from_numpy(X_vl).float().to(args.device)).cpu().numpy()
                        # logits_te = model_lin(torch.from_numpy(X_te).float().to(args.device)).cpu().numpy()
                        logits_vl = model_lin(torch.from_numpy(X_vl).float().to(args.device)) / float(temper)
                        logits_te = model_lin(torch.from_numpy(X_te).float().to(args.device)) / float(temper)
                    for split_name, logits, y in [("val", logits_vl, y_vl), ("test", logits_te, y_te)]:
                        yhat = logits.argmax(axis=1)
                        top1 = (yhat == y).mean()
                        bal = balanced_acc(y, yhat, K)
                        top5 = topk_acc(y, logits, 5)
                        macro = f1_score(y, yhat, average='macro')
                        # standard linear (no prompts) => alpha recorded as 1.0 to indicate pure image linear head (or you can choose "linear")
                        # records.append((shot, "linear", 1.0,  float(temp), 0.0, float(lr_val), float(wds_val), split_name, backbone, top1, top5, bal, macro))
                        records.append((shot, "linear", 1.0, float(temper), 0.0, float(lr_val), float(wds_val), split_name, backbone, top1, top5, bal, macro))

        # --- Prompt-aware linear probe sweep (parallel, CPU) across (alpha x lr_grid x wds_grid) ---
        if args.use_prompts_for_linear:
            if text_embs_bundle is None:
                raise RuntimeError("Prompt embeddings not available for linear probe.")

            text_embs_numpy = _serialize_text_embs_to_numpy(text_embs_bundle)
            label_names_list = list(label_names)
            X_sup_np = X_sup.astype(np.float32)
            y_sup_np = y_sup.astype(np.int64)
            X_vl_np = X_vl.astype(np.float32)
            y_vl_np = y_vl.astype(np.int64)
            X_te_np = X_te.astype(np.float32)
            y_te_np = y_te.astype(np.int64)

            # build list of parameter combinations
            combos = []
            for alpha in args.alpha_grid:
                for temp in args.temp_grid:
                    for lr_val in args.lr_grid:
                        for wds_val in args.wds_grid:
                            combos.append((float(alpha), float(temp), float(lr_val), float(wds_val)))

            # parallelize CPU training across combos
            with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
                futures = {}
                for alpha, temp_val, lr_val, wds_val in combos:
                    # fut = pool.submit(
                    #     eval_linear_alpha_lr_wd_serial,
                    #     alpha, lr_val, wds_val,
                    #     X_sup_np, y_sup_np, label_names_list, text_embs_numpy, K,
                    #     X_vl_np, y_vl_np, X_te_np, y_te_np, args.seed + int(shot), args.batch_size, args.linear_epochs_for_prompts
                    # )
                    fut = pool.submit(
                        eval_linear_alpha_lr_wd_serial,
                        alpha, temp_val, lr_val, wds_val,
                        X_sup_np, y_sup_np, label_names_list, text_embs_numpy, K,
                        X_vl_np, y_vl_np, X_te_np, y_te_np, args.seed + int(shot), args.batch_size, args.linear_epochs_for_prompts
                    )


                    futures[fut] = (alpha, temp_val, lr_val, wds_val)

                for fut in as_completed(futures):
                    alpha, lr_val, wds_val = futures[fut]
                    try:
                        a, temp_r, lr_r, wds_r, val_acc, test_acc, logits_vl_p, logits_te_p = fut.result()
                    except Exception as e:
                        print(f"[{backbone}][linear+prompts] combo alpha={alpha}, lr={lr_val}, wds={wds_val} failed: {e}", file=sys.stderr)
                        records.append((shot, f"linear+prompts(alpha={alpha:.2f})", float(temp_r), alpha, 0.0, float(lr_val), float(wds_val), "val", backbone, np.nan, np.nan, np.nan, np.nan))
                        records.append((shot, f"linear+prompts(alpha={alpha:.2f})", float(temp_r), alpha, 0.0, float(lr_val), float(wds_val), "test", backbone, np.nan, np.nan, np.nan, np.nan))
                        continue

                    # record metrics for val & test
                    for split_name, logits, y in [("val", logits_vl_p, y_vl), ("test", logits_te_p, y_te)]:
                        yhat = logits.argmax(axis=1)
                        top1 = (yhat == y).mean()
                        bal = balanced_acc(y, yhat, K)
                        top5 = topk_acc(y, logits, 5)
                        macro = f1_score(y, yhat, average='macro')
                        # prompt-aware linear: record alpha, lr, wds. temp is not used here => record 0
                        records.append((shot, f"linear+prompts(alpha={a:.2f})", a, float(temp_r), 0.0, float(lr_r), float(wds_r), split_name, backbone, top1, top5, bal, macro))
    return records


# ------------------------------
# Direct on-the-fly evaluation (prompt pooling & prototypes built from model)
# ------------------------------
# (left unmodified except we will record temp properly in final writer)
# ... (the direct eval functions are present above and unchanged) ...
# For brevity, we reuse evaluate_prompts_and_few_shot_direct as-is from original file.
# (If you want sweeps over temp for direct eval, we could call evaluate_prompts_and_few_shot_direct with different temps
#  in parallel but note it will encode images on-the-fly on GPU; avoid parallel GPU encodes.)

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

    # Vectorized prompt-pooling (GPU-accelerated)
    img_mat = torch.stack(image_embeddings).to(device)
    labels = list(prompt_bundle.keys())
    all_prompts = [info["prompt_embeddings"].to(device) for info in prompt_bundle.values()]
    pooled_prompts = []
    for pe in all_prompts:
        if pe.numel() == 0:
            pooled_prompts.append(torch.zeros_like(all_prompts[0][0]))
        else:
            sims = F.normalize(pe, dim=-1)
            pooled_prompts.append(sims.mean(dim=0))
    label_mat = torch.stack(pooled_prompts)
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

    # linear probe baseline (on device)
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
    ap = argparse.ArgumentParser(description="Unified few-shot + zero-shot evaluation for CLIP-style models (prompt-aware few-shot extensions) with hyperparam sweep")
    ap.add_argument("--data-root", default=DATA_ROOT, help="Path to dataset root")
    ap.add_argument("--train-csv", default=TRAIN_CSV, help="Training CSV file")
    ap.add_argument("--val-csv", default=VAL_CSV, help="Validation CSV file")
    ap.add_argument("--test-csv", default=TEST_CSV, help="Test CSV file")
    ap.add_argument("--labels", default=LABELS, help="TSV with id\tlabel_name")
    ap.add_argument("--prompts-path", default=DEFAULT_PROMPTS_JSON)
    ap.add_argument("--backbones", nargs="+",default=[b for b in DEFAULT_PRETRAINED.keys()], help="Model backbones to use (default uses DEFAULT_BACKBONES values).")
    ap.add_argument("--pretrained",nargs="+", default=[DEFAULT_PRETRAINED.get(b, "openai") for b in DEFAULT_PRETRAINED.keys()], help="Pretrained weights for each backbone (default uses DEFAULT_PRETRAINED map).")
    ap.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    ap.add_argument("--save-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100, help="Epochs for standard linear probe on device")
    ap.add_argument("--linear-epochs-for-prompts", type=int, default=50, help="Epochs for CPU prompt-aware linear training within workers")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--direct-eval", action="store_true", help="Run direct prompt-based evaluation (encodes images on the fly).")
    ap.add_argument("--num-samples", type=int, default=None, help="If using direct-eval, optionally subsample test set.")
    ap.add_argument("--temp", type=float, default=0.12, help="Temp for prompt pooling softmax")
    ap.add_argument("--no-compile", action="store_false", help="Dont attempt torch.compile() for speed where possible")
    ap.add_argument("--max-workers", type=int, default=8)

    # Prompt-aware few-shot options (defaults ON)
    ap.add_argument("--no-prompts-for-prototype", action="store_true", help="Disable text-prompt fusion for prototypes")
    ap.add_argument("--no-prompts-for-linear", action="store_true", help="Disable text-prompt pseudo-examples for linear probe")

    # grids (accept comma-separated string or Python list string or leave default)
    ap.add_argument("--alpha-grid", type=str, default=",".join([str(a) for a in ALPHAS]), help="Comma-separated list of prompt-alpha values to sweep over")
    ap.add_argument("--temp-grid", type=str, default=",".join([str(t) for t in DEFAULT_TEMPS]), help="Comma-separated list of temp values to sweep over (used for direct eval/pooling)")
    ap.add_argument("--lr-grid", type=str, default=",".join([str(l) for l in DEFAULT_LRS]), help="Comma-separated list of learning rates to sweep over")
    ap.add_argument("--wds-grid", type=str, default=",".join([str(w) for w in DEFAULT_WDS]), help="Comma-separated list of weight decay values to sweep over")

    args = ap.parse_args()

    args.use_prompts_for_prototype = not args.no_prompts_for_prototype
    args.use_prompts_for_linear = not args.no_prompts_for_linear

    # normalize grids
    def parse_grid(s, default_list):
        if s is None:
            return list(default_list)
        if isinstance(s, (list, tuple)):
            return [float(x) for x in s]
        try:
            if isinstance(s, str):
                parts = [p.strip() for p in s.split(",") if p.strip() != ""]
                return [float(x) for x in parts] if parts else list(default_list)
        except Exception:
            return list(default_list)
        return list(default_list)

    args.alpha_grid = parse_grid(args.alpha_grid, ALPHAS)
    args.temp_grid = parse_grid(args.temp_grid, DEFAULT_TEMPS)
    args.lr_grid = parse_grid(args.lr_grid, DEFAULT_LRS)
    args.wds_grid = parse_grid(args.wds_grid, DEFAULT_WDS)

    return args


def main():
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # map pretrained names to backbones
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
    args.pretrained = pretrained_map

    label_names, name2id = load_labels(args.labels)
    K = len(label_names)

    show_progress = sys.stdout.isatty()

    records_all = []

    if args.direct_eval:
        # If we want to sweep temps for direct eval, run sequentially (GPU image encoding) to avoid multiple GPU jobs.
        for backbone in args.backbones:
            for temp_val in args.temp_grid:
                res = evaluate_prompts_and_few_shot_direct(
                    labels=list(label_names),
                    train_csv=args.train_csv,
                    split_csv=args.test_csv,
                    shots=args.shots,
                    model_name=backbone,
                    device=args.device,
                    num_samples=args.num_samples,
                    temp=float(temp_val),
                    delta_prompts_path=args.prompts_path,
                    pretrained=args.pretrained.get(backbone, "openai"),
                    random_state=args.seed,
                    cache_dir=args.save_dir,
                    root=args.data_root,
                    show_progress=show_progress
                )
                outp = Path(args.results_dir) / f"direct_eval_{backbone.replace('/', '_')}_temp{temp_val}.json"
                with open(outp, "w", encoding="utf-8") as f:
                    json.dump({"temp": temp_val, **res}, f, indent=2)
    else:
        for backbone in (tqdm(args.backbones, desc="backbones", leave=True, miniters=1) if show_progress and sys.stdout.isatty() else args.backbones):
            recs = evaluate_backbone(backbone, args, list(label_names), K, show_progress=show_progress)
            records_all.extend(recs)

        out_csv = Path(args.results_dir) / "few_shot_table_all_backbones_sweep.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # Add columns: shot, model, alpha, temp, lr, wds, split, backbone, top1, top5, balanced_acc, macro_f1
            w.writerow(["shot", "model", "alpha", "temp", "lr", "wds", "split", "backbone",
                        "top1", "top5", "balanced_acc", "macro_f1"])
            for r in records_all:
                # normalized to the tuple shape we appended:
                # (shot, model, alpha, temp, lr, wds, split_name, backbone, top1, top5, bal, macro)
                (shot, model, alpha, temp_val, lr_val, wds_val, split_name, backbone, top1, top5, bal, macro) = r
                w.writerow([
                            shot, model, alpha, temp_val, lr_val, wds_val, split_name, backbone,
                            top1, top5, bal, macro
                            ])

        if show_progress:
            print(f"Wrote results -> {out_csv}")
        else:
            print(f"Wrote results -> {out_csv}")


if __name__ == "__main__":
    main()
