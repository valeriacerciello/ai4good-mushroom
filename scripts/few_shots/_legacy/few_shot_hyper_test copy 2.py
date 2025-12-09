#!/usr/bin/env python3
"""
Patched & parallelized version of test_few_shots_overall_patched.py

Major changes:
- Parallelized CPU-bound sweep operations using ThreadPoolExecutor while avoiding GPU oversubscription.
- Added --cpu-probes flag to force small linear-probe trainings to run on CPU (default True) so that multiple sweeps can be parallelized across CPU cores while the GPU remains available for heavy encoding work.
- Added safer max_workers handling: when device is 'cuda' we avoid parallel GPU-using tasks across workers.
- Vectorized / batched evaluations where safe. Kept all sweep grids identical.
- Minor DataLoader and caching tweaks retained, and torch.compile invocations preserved where available.

Save as new file and run similarly to the original.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import math
import time

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
    def tqdm(x, *args, **kwargs): return x

# Limit CPU threads (avoid oversubscription on shared node)
MAX_WORKERS = 64
BATTCH_SIZE = 512

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")
# Enable faster GPU memory allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))


# ---------- globals ----------
RESULTS_DIR = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/results"
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
TEMP_GRID = [1,5,10,20,50,100]
PROMPT_SET = ["ensemble","v1","names"]
LR_GRID = [1e-4,5e-4,1e-3,3e-3]
WD_GRID = [0,1e-4,5e-4]
MIX_STRATEGIES = ["normalize","none"]

DATA_ROOT = "/home/c/dkorot/AI4GOOD/provided_dir/datasets/mushroom/merged_dataset"
TRAIN_CSV = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/train.csv"
VAL_CSV = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/val.csv"
TEST_CSV = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/test.csv"
LABELS = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/data_prompts_label/labels.tsv"

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
    # we can parallelize labels here (CPU-bound + batched model.encode_text calls)
    # but ensure not to create too many threads that contend for GPU if device == 'cuda'
    # We'll do simple sequential here to keep memory deterministic; users may re-run with CPU device for parallel text builds.
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
                                      alpha: float = 0.5, mix_strategy: str = 'normalize') -> np.ndarray:
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
        if mix_strategy == 'normalize':
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
                       compile_model: bool = False, show_progress: bool = False, weight_decay: float = 1e-4) -> nn.Module:
    """
    Training kept basically identical. Use device arg to move training to CPU or GPU.
    """
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

def train_linear_probe_with_prompts(
    X_support, y_support, label_names, text_embs, K, alpha=0.5,
    epochs=200, lr=1e-3, batch_size=32, device="cpu",
    seed=42, compile_model=False, show_progress=False):

    # mix image & text features per class and then call train_linear_probe
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
    if not X_mixed:
        # nothing to train
        D = X_support.shape[1]
        model = LinearProbe(D, K)
        return model
    X_aug = np.concatenate(X_mixed, axis=0)
    y_aug = np.concatenate(y_mixed, axis=0)

    model = train_linear_probe(
        X_aug, y_aug, epochs=epochs, lr=lr, batch_size=batch_size,
        device=device, seed=seed, compile_model=compile_model, show_progress=show_progress)
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
    # unchanged caching but more robust DataLoader settings
    if cache_prefix:
        cache_path = Path(cache_dir) / f"{cache_prefix}.npz"
        if cache_path.exists():
            try:
                arr = np.load(cache_path)["embeddings"]
                return [torch.tensor(a) for a in arr]
            except Exception:
                pass

    dataset = ImageDataset(image_paths, preprocess, root=root)
    n_workers = min(16, max(0, (os.cpu_count() or 4) // 2))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else 2
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
    """
    Main evaluation function. Major parallelization strategy:
      - For prototype prompt-alpha sweeps we parallelize alpha evaluations using ThreadPoolExecutor (cpu-bound).
      - For linear probe sweeps: training uses device=args.probe_device (CPU by default to avoid GPU oversubscription).
        Trainings across (lr,wd) are submitted to ThreadPoolExecutor if probe_device == 'cpu' (parallel).
        If probe_device == 'cuda' then linear probe training is performed sequentially to avoid GPU contention.
    """
    model_pretrained = args.pretrained.get(backbone, "openai")
    if show_progress and sys.stdout.isatty():
        print(f"Evaluating backbone: {backbone} (pretrained={model_pretrained})")

    X_tr, y_tr, _ = ensure_features("train", backbone, args.data_root, args.train_csv, args.labels, save_dir=args.save_dir, pretrained=model_pretrained)
    X_vl, y_vl, _ = ensure_features("val", backbone, args.data_root, args.val_csv, args.labels, save_dir=args.save_dir, pretrained=model_pretrained)
    X_te, y_te, _ = ensure_features("test", backbone, args.data_root, args.test_csv, args.labels, save_dir=args.save_dir, pretrained=model_pretrained)

    # Normalize embeddings
    X_tr = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-8)
    X_vl = X_vl / (np.linalg.norm(X_vl, axis=1, keepdims=True) + 1e-8)
    X_te = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-8)

    # Zero-shot text embeddings
    text_embeddings = zero_shot_scores(label_names, backbone, model_pretrained, args.device, args.prompts_path, cache_dir=args.save_dir)

    # Precompute per-prompt-set text embeddings if requested
    text_embs_bundles = {"none": None}
    if args.use_prompts_for_prototype or args.use_prompts_for_linear:
        for pset in args.prompt_sets:
            try:
                if show_progress and sys.stdout.isatty():
                    print(f"[{backbone}] computing text embeddings for prompt_set={pset}...")
                text_embs_bundles[pset] = get_text_embeddings(list(label_names),
                                               prompt_set=pset,
                                               model_name=backbone,
                                               device=args.device,
                                               batch_size=128,
                                               include_common_names=False,
                                               common_prompts_path=args.prompts_path,
                                               pretrained=model_pretrained,
                                               cache_dir=Path(args.save_dir) / "text_cache")
            except Exception as e:
                print(f"[warn] failed to get_text_embeddings for prompt_set={pset}: {e}")
                text_embs_bundles[pset] = None

    # Move val/test to torch tensors on device for evaluation
    device = args.device
    X_vl_gpu = torch.from_numpy(X_vl).float().to(device)
    X_te_gpu = torch.from_numpy(X_te).float().to(device)
    records = []

    # Zero-shot if requested
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

    # Decide probe device: if user requested CPU probes (default True) we use cpu to allow parallel training; else use same device
    probe_device = "cpu" if args.cpu_probes else device

    for shot in shot_iter:
        rng = np.random.default_rng(args.seed + int(shot))
        sup_idx, sup_labels = sample_few_shot_indices(y_tr, K, shot, rng)
        X_sup, y_sup = X_tr[sup_idx], sup_labels
        # Normalize the few-shot subset
        X_sup = X_sup / (np.linalg.norm(X_sup, axis=1, keepdims=True) + 1e-8)

        # Precompute the image-only prototypes once per shot (reused across temps)
        prototypes = prototype_classifier(X_sup, y_sup, K)
        prototypes_torch = torch.from_numpy(prototypes).float().to(device)

        # Prototype sweeps (prompt_set, mix_strategy, temp)
        prototype_tasks = []
        # We'll parallelize alpha loops (they are CPU-heavy mixing + evaluation) using ThreadPoolExecutor
        for prompt_set in (args.prompt_sets if args.use_prompts_for_prototype else ["none"]):
            text_embs_for_set = text_embs_bundles.get(prompt_set, None)
            for mix_strategy in args.mix_strategies:
                for temp in args.temp_grid:
                    # First evaluate standard image-only prototype (cheap) and store
                    for split_name, X_gpu, y in [("val", X_vl_gpu, y_vl), ("test", X_te_gpu, y_te)]:
                        scores = (X_gpu @ prototypes_torch.T) * temp
                        yhat = torch.argmax(scores, axis=1).cpu().numpy()
                        top1 = (yhat == y).mean()
                        bal = balanced_acc(y, yhat, K)
                        top5 = topk_acc(y, scores.cpu().numpy(), 5)
                        macro = f1_score(y, yhat, average='macro')
                        records.append((shot, "prototype", 0.0, temp, prompt_set, 0.0, 0.0, mix_strategy, split_name, backbone, top1, top5, bal, macro))

                    # If prompt fusion is enabled, we need to sweep alphas -> CPU-bound combination + eval; parallelize alphas
                    if args.use_prompts_for_prototype and text_embs_for_set is not None:
                        # Submit per-alpha jobs to executor later
                        prototype_tasks.append((prompt_set, mix_strategy, temp, text_embs_for_set, X_sup, y_sup))

        # Efficient execution of prototype alpha sweeps in parallel across alphas
        if prototype_tasks:
            # Number of threads: limit to args.max_workers but also avoid more than needed
            max_workers = min(args.max_workers, max(1, (os.cpu_count() or 4) // 2))
            # If the main device is gpu and users didn't enable cpu_probes, still safe to CPU-parallelize these (they use numpy/text_embs)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = []
                for (prompt_set, mix_strategy, temp, text_embs_for_set, X_sup_loc, y_sup_loc) in prototype_tasks:
                    # For each alpha, spawn tasks for all alphas (alpha loop is independent)
                    for alpha in args.alpha_grid:
                        futures.append(ex.submit(_eval_prototype_alpha,
                                                 X_sup_loc, y_sup_loc, K,
                                                 list(label_names), text_embs_for_set, alpha, mix_strategy, temp,
                                                 X_vl_gpu, y_vl, X_te_gpu, y_te, backbone, shot, prompt_set))
                # collect results
                for fut in as_completed(futures):
                    recs = fut.result()
                    # recs is a list of record tuples to append
                    records.extend(recs)

        # Linear probe (standard) - mode-specific lr/wd sweeps
        # We'll try to parallelize training across (lr,wd) only when probe_device == 'cpu' (i.e., CPU-based trainings)
        lin_training_jobs = []
        for lr in args.lr_grid:
            for wd in args.wd_grid:
                lin_training_jobs.append((lr, wd))

        if probe_device == "cpu" and lin_training_jobs:
            max_workers = min(args.max_workers, max(1, (os.cpu_count() or 4) // 2))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = []
                for lr, wd in lin_training_jobs:
                    futs.append(ex.submit(_train_and_eval_linear_probe,
                                           X_sup, y_sup, X_vl, y_vl, X_te, y_te, K,
                                           args.epochs, lr, args.batch_size, probe_device, args.seed + int(shot), args.no_compile, wd, args.temp_grid, backbone, shot))
                for f in as_completed(futs):
                    recs = f.result()
                    records.extend(recs)
        else:
            # sequential training (safer if using GPU for probes)
            for lr, wd in lin_training_jobs:
                recs = _train_and_eval_linear_probe(
                    X_sup, y_sup, X_vl, y_vl, X_te, y_te, K,
                    args.epochs, lr, args.batch_size, probe_device, args.seed + int(shot), args.no_compile, wd, args.temp_grid, backbone, shot)
                records.extend(recs)

        # Prompt-aware linear probe sweep
        if args.use_prompts_for_linear:
            # We'll iterate prompt_set -> alpha -> lr -> wd. We parallelize (lr,wd) if probe_device == 'cpu'
            for prompt_set in args.prompt_sets:
                text_embs_for_set = text_embs_bundles.get(prompt_set, None)
                if text_embs_for_set is None:
                    continue
                # precompute text label embeddings array for quick access inside train func
                for alpha in args.alpha_grid:
                    # Prepare training function bound args
                    # if probe_device == 'cpu' -> parallelize lr/wd
                    p_jobs = []
                    for lr in args.lr_grid:
                        for wd in args.wd_grid:
                            p_jobs.append((alpha, lr, wd))
                    if probe_device == "cpu" and p_jobs:
                        max_workers = min(args.max_workers, max(1, (os.cpu_count() or 4) // 2))
                        with ThreadPoolExecutor(max_workers=max_workers) as ex:
                            futs = []
                            for alpha, lr, wd in p_jobs:
                                futs.append(ex.submit(_train_and_eval_linear_probe_with_prompts,
                                                      X_sup, y_sup, list(label_names), text_embs_for_set, K,
                                                      alpha, args.epochs, lr, args.batch_size, probe_device,
                                                      args.seed + int(shot), args.no_compile, args.temp_grid, backbone, shot, prompt_set, lr, wd))
                            for f in as_completed(futs):
                                recs = f.result()
                                records.extend(recs)
                    else:
                        for alpha, lr, wd in p_jobs:
                            recs = _train_and_eval_linear_probe_with_prompts(
                                X_sup, y_sup, list(label_names), text_embs_for_set, K,
                                alpha, args.epochs, lr, args.batch_size, probe_device,
                                args.seed + int(shot), args.no_compile, args.temp_grid, backbone, shot, prompt_set, lr, wd)
                            records.extend(recs)

    return records

# Helper functions for parallelized tasks ------------------------------------------------

def _eval_prototype_alpha(X_sup, y_sup, K, label_names, text_embs_for_set, alpha, mix_strategy, temp,
                          X_vl_gpu, y_vl, X_te_gpu, y_te, backbone, shot, prompt_set):
    """
    Evaluate a single alpha value for prototype+prompts combination.
    Returns list of record tuples.
    """
    recs = []
    try:
        prototypes_prompts = prototype_classifier_with_prompts(X_sup, y_sup, K, label_names, text_embs_for_set, alpha=alpha, mix_strategy=mix_strategy)
        prot = torch.from_numpy(prototypes_prompts).float().to(X_vl_gpu.device)
        for split_name, X_gpu, y in [("val", X_vl_gpu, y_vl), ("test", X_te_gpu, y_te)]:
            scores = (X_gpu @ prot.T) * temp
            yhat = torch.argmax(scores, axis=1).cpu().numpy()
            top1 = (yhat == y).mean()
            bal = balanced_acc(y, yhat, K)
            top5 = topk_acc(y, scores.cpu().numpy(), 5)
            macro = f1_score(y, yhat, average='macro')
            recs.append((shot, f"prototype+prompts(alpha={alpha:.2f})", alpha, temp, prompt_set, 0.0, 0.0, mix_strategy, split_name, backbone, top1, top5, bal, macro))
    except Exception as e:
        print(f"[warn] prototype alpha eval failed: {e}")
    return recs

def _train_and_eval_linear_probe(X_sup, y_sup, X_vl, y_vl, X_te, y_te, K,
                                 epochs, lr, batch_size, device, seed, compile_flag, weight_decay, temp_grid, backbone, shot):
    """
    Train a linear probe (image-only) and evaluate across temps.
    Returns list of record tuples.
    """
    recs = []
    try:
        model_lin = train_linear_probe(X_sup, y_sup, X_val=None, y_val=None,
                                       epochs=epochs, lr=lr, batch_size=batch_size,
                                       device=device, seed=seed, compile_model=compile_flag,
                                       show_progress=False, weight_decay=weight_decay)
        model_lin.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_vl).float().to(device)
            xt = torch.from_numpy(X_te).float().to(device)
            logits_vl = model_lin(xv).cpu().numpy()
            logits_te = model_lin(xt).cpu().numpy()
        for temp in temp_grid:
            logits_vl_t = logits_vl * temp
            logits_te_t = logits_te * temp
            for split_name, logits, y in [("val", logits_vl_t, y_vl), ("test", logits_te_t, y_te)]:
                yhat = logits.argmax(axis=1)
                top1 = (yhat == y).mean()
                bal = balanced_acc(y, yhat, K)
                top5 = topk_acc(y, logits, 5)
                macro = f1_score(y, yhat, average='macro')
                recs.append((shot, "linear", 0.0, temp, "none", lr, weight_decay, "none", split_name, backbone, top1, top5, bal, macro))
    except Exception as e:
        print(f"[warn] linear probe train/eval failed: {e}")
    return recs

def _train_and_eval_linear_probe_with_prompts(X_sup, y_sup, label_names, text_embs_for_set, K,
                                              alpha, epochs, lr, batch_size, device, seed, compile_flag, temp_grid, backbone, shot, prompt_set, lr_in, wd_in):
    """
    Train linear probe with prompts pseudo-examples (alpha weighting) and evaluate across temps.
    Returns list of record tuples.
    """
    recs = []
    try:
        model_lin_p = train_linear_probe_with_prompts(X_sup, y_sup, label_names, text_embs_for_set, K,
                                                      alpha=alpha, epochs=epochs, lr=lr, batch_size=batch_size,
                                                      device=device, seed=seed, compile_model=compile_flag, show_progress=False)
        model_lin_p.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_vl).float().to(device)
            xt = torch.from_numpy(X_te).float().to(device)
            logits_vl_p = model_lin_p(xv).cpu().numpy()
            logits_te_p = model_lin_p(xt).cpu().numpy()
        for temp in temp_grid:
            logits_vl_tp = logits_vl_p * temp
            logits_te_tp = logits_te_p * temp
            for split_name, logits, y in [("val", logits_vl_tp, y_vl), ("test", logits_te_tp, y_te)]:
                yhat = logits.argmax(axis=1)
                top1 = (yhat == y).mean()
                bal = balanced_acc(y, yhat, K)
                top5 = topk_acc(y, logits, 5)
                macro = f1_score(y, yhat, average='macro')
                recs.append((shot, f"linear+prompts(alpha={alpha:.2f})", alpha, temp, prompt_set, lr, wd_in, "none", split_name, backbone, top1, top5, bal, macro))
    except Exception as e:
        print(f"[warn] linear+prompts train/eval failed: {e}")
    return recs

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

    # linear probe baseline (kept sequential small training)
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
    ap.add_argument("--pretrained",nargs="+", default=[DEFAULT_PRETRAINED.get(b, "openai") for b in DEFAULT_PRETRAINED.keys()], help="Pretrained weights for each backbone (default uses DEFAULT_PRETRAINED map).")
    ap.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    ap.add_argument("--save-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=BATTCH_SIZE)
    ap.add_argument("--direct-eval", action="store_true", help="Run direct prompt-based evaluation (encodes images on the fly).")
    ap.add_argument("--num-samples", type=int, default=None, help="If using direct-eval, optionally subsample test set.")
    ap.add_argument("--temp", type=float, default=0.12, help="Temp for prompt pooling softmax")
    ap.add_argument("--no-compile", action="store_false", dest="no_compile", help="Dont attempt torch.compile() for speed where possible")
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--cpu-probes", action="store_true", default=True, help="Run linear-probe trainings on CPU to avoid GPU oversubscription (default True). Set to False to force probe training on same device.")
    # Prompt-aware few-shot options
    ap.add_argument("--no-prompts-for-prototype", action="store_true", help="Disable text-prompt fusion for prototypes")
    ap.add_argument("--no-prompts-for-linear", action="store_true", help="Disable text-prompt pseudo-examples for linear probe")
    ap.add_argument("--alpha-grid", default=ALPHAS, help="Comma-separated list of prompt-alpha values to sweep over")
    ap.add_argument("--temp-grid", default=TEMP_GRID, help="Comma-separated list of temperatures for logit scaling")
    ap.add_argument("--prompt-sets", default=PROMPT_SET, help="Comma-separated prompt sets to sweep")
    ap.add_argument("--lr-grid", default=LR_GRID, help="Comma-separated learning rates for linear probe")
    ap.add_argument("--wd-grid", default=WD_GRID, help="Comma-separated weight decay values for linear probe")
    ap.add_argument("--mix-strategies", default=MIX_STRATEGIES, help="Comma-separated mixing strategies: normalize,none")

    args = ap.parse_args()
    args.use_prompts_for_prototype = not args.no_prompts_for_prototype
    args.use_prompts_for_linear = not args.no_prompts_for_linear

    # Ensure lists (argparse sometimes gives string)
    if isinstance(args.alpha_grid, str):
        args.alpha_grid = [float(x) for x in args.alpha_grid.split(",")]
    if isinstance(args.temp_grid, str):
        args.temp_grid = [float(x) for x in args.temp_grid.split(",")]
    if isinstance(args.lr_grid, str):
        args.lr_grid = [float(x) for x in args.lr_grid.split(",")]
    if isinstance(args.wd_grid, str):
        args.wd_grid = [float(x) for x in args.wd_grid.split(",")]
    if isinstance(args.prompt_sets, str):
        args.prompt_sets = args.prompt_sets.split(",")
    if isinstance(args.mix_strategies, str):
        args.mix_strategies = args.mix_strategies.split(",")

    # Make sure max_workers reasonable
    args.max_workers = max(1, int(min(args.max_workers, (os.cpu_count() or 4))))

    return args

def main():
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    pretrained_map = {}
    for b in args.backbones:
        if hasattr(args, "pretrained") and args.pretrained and len(args.pretrained) > 0:
            idx = args.backbones.index(b)
            pretrained_map[b] = args.pretrained[idx] if idx < len(args.pretrained) else DEFAULT_PRETRAINED.get(b, "openai")
        else:
            pretrained_map[b] = DEFAULT_PRETRAINED.get(b, "openai")
    args.pretrained = pretrained_map

    label_names, name2id = load_labels(args.labels)
    K = len(label_names)
    show_progress = sys.stdout.isatty()
    records_all = []

    if args.direct_eval:
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
        # If the device is cuda and cpu_probes False, we should avoid parallelizing backbones to prevent GPU contention.
        # We handle sequential backbone processing here; internal functions parallelize CPU tasks when safe.
        for backbone in backbone_iter:
            recs = evaluate_backbone(backbone, args, list(label_names), K, show_progress=show_progress)
            records_all.extend(recs)

        out_csv = Path(args.results_dir) / "few_shot_table_all_backbones.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "shot", "model", "alpha", "temp", "prompt_set",
                "lr", "weight_decay", "mix_strategy",
                "split", "backbone",
                "top1", "top5", "balanced_acc", "macro_f1"
            ])
            for r in records_all:
                w.writerow(r)
        print(f"Wrote results -> {out_csv}")

if __name__ == "__main__":
    main()
