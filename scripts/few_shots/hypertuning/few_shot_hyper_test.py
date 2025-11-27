#!/usr/bin/env python3

# ===== HARD DISABLE TORCH DYNAMO BEFORE ANY PYTORCH IMPORTS =====
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch._dynamo
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True
# =================================================================

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
import datetime
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

# ---------- global environment & torch tuning (optimized) ----------
# Limit CPU threads (avoid oversubscription on shared node)
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")
# Enable faster GPU memory allocator (if available)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
# Reduce torch.cudnn nondeterminism cost but keep performance heuristics
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# Set reasonable default threads
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))

# TorchDynamo/compiler behavior:
# - Dynamo can cause many costly recompiles when global state (grad_mode, device) changes.
# - For long-running sweeping/eval jobs it's frequently faster to disable it.
try:
    import torch._dynamo as _dynamo
    # disable dynamo by default, but allow override via env var
    if os.environ.get("ENABLE_TORCH_DYNAMO", "0") == "1":
        _dynamo.config.suppress_errors = True
    else:
        _dynamo.config.suppress_errors = True
        _dynamo.disable()
except Exception:
    # If torch._dynamo is missing or errors, continue without failing
    pass

# Helpful debug env flags (uncomment if you want recompilation diagnostics)
# os.environ.setdefault("TORCH_LOGS", "recompiles")



# ---------- globals ----------
torch.set_float32_matmul_precision('high')

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
TEMP_GRID = [1] #,5,10,20,50,100]
PROMPT_SET = ["ensemble", "v1", "names", "delta"]
LR_GRID = [1e-3, 3e-3, 3e-2, 3e-1]
WD_GRID = [0,1e-4,5e-4]
MIX_STRATEGIES = ["none"]# ["normalize","none"]

WORK_ENV = "/home/c/dkorot/AI4GOOD"
DATA_ROOT = WORK_ENV + "/provided_dir/datasets/mushroom/merged_dataset"
TRAIN_CSV = WORK_ENV + "/ai4good-mushroom/splits/train.csv"
VAL_CSV = WORK_ENV + "/ai4good-mushroom/splits/val.csv"
TEST_CSV = WORK_ENV + "/ai4good-mushroom/splits/test.csv"
LABELS = WORK_ENV + "/ai4good-mushroom/data_prompts_label/labels.tsv"
RESULTS_DIR = WORK_ENV + "/ai4good-mushroom/results"
DEFAULT_CACHE_DIR = "/zfs/ai4good/student/dkorot/ai4good-mushroom/features"
DEFAULT_PROMPTS_JSON = WORK_ENV + "/ai4good-mushroom/data_prompts_label/delta_prompts.json"

DEFAULT_TEXT_CACHE_DIR = Path(".cache_text_embeddings")
DEFAULT_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
    elif prompt_set == "delta":
        # load delta prompts from JSON
        try:
            with open(DEFAULT_PROMPTS_JSON, "r", encoding="utf-8") as f:
                delta = json.load(f)
            if label in delta:
                return list(delta[label])
        except Exception:
            raise ValueError(f"Delta prompts requested but failed to load from {DEFAULT_PROMPTS_JSON}")
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
        try:
            with open(DEFAULT_PROMPTS_JSON, "r", encoding="utf-8") as f:
                delta = json.load(f)
            if label in delta:
                for dp in delta[label]:
                    if dp not in prompts:
                        prompts.append(dp)
        except Exception:
            raise ValueError(f"Delta prompts requested but failed to load from {DEFAULT_PROMPTS_JSON}")
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
                        cache_dir: Optional[Path] = None,
                        model=None, tokenizer=None) -> Dict[str, Any]:
    """
    Compute and cache text embeddings per label.

    Improvements:
      - Accepts already-created `model` and `tokenizer` so callers can reuse a single CLIP instance per-backbone.
      - Uses a deterministic cache key based on (backbone, pretrained, prompt_set, label).
      - Avoids torch.compile on encode_text to prevent repeated slow compile attempts.
      - Forces text encoding on CPU when device == 'cpu' for easy parallelism.
    Returns: {label: {'prompts': [...], 'prompt_embeddings': Tensor (n x D), 'label_embedding': Tensor (D,)}}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(cache_dir or DEFAULT_TEXT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # create model/tokenizer once if not provided
    created_local = False
    if model is None or tokenizer is None:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name)
        created_local = True

    # Do NOT torch.compile(model.encode_text) here globally; compiling can cause repeated slow recompiles.
    # If you want to experiment with torch.compile, do so at top-level once with a controlled flag.

    # load common prompts file if requested
    common_prompts = None
    if include_common_names and common_prompts_path and os.path.exists(common_prompts_path):
        try:
            with open(common_prompts_path, "r", encoding="utf-8") as f:
                common_prompts = json.load(f)
        except Exception:
            common_prompts = None

    results: Dict[str, Any] = {}
    model.eval()
    # We prefer encoding text on CPU (unless device explicitly set to cuda) to make parallel text encodes inexpensive.
    txt_device = "cpu" if device == "cpu" else device

    for label in labels:
        if common_prompts and label in common_prompts:
            prompts = list(common_prompts[label])
        else:
            prompts = _build_prompt_list_for_label(label, prompt_set, include_common_names, mg=None)


        # cache key unique per-backbone/prompt_set/label list
        h_input = "|".join(prompts) + "|" + model_name + "|" + pretrained + "|" + prompt_set
        h = hashlib.sha1(h_input.encode()).hexdigest()
        cache_path = cache_dir / f"{label}_{h}.npz"
        if cache_path.exists():
            try:
                z = np.load(cache_path, allow_pickle=True)
                pe = torch.from_numpy(z["prompt_embeddings"])
                le = torch.from_numpy(z["label_embedding"])
                results[label] = {"prompts": list(z["prompts"]), "prompt_embeddings": pe, "label_embedding": le}
                continue
            except Exception:
                # if cache load fails, regenerate
                try:
                    cache_path.unlink()
                except Exception:
                    pass

        # Encode prompts in batches
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                toks = tokenizer(prompts[i:i + batch_size])
                # move tokens to txt_device
                toks = toks.to(txt_device)
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
        # Save compact numpy arrays for fast reload later
        np.savez_compressed(cache_path, prompts=np.array(prompts, dtype=object),
                            prompt_embeddings=pe.numpy().astype(np.float32),
                            label_embedding=le.numpy().astype(np.float32))
        results[label] = {"prompts": prompts, "prompt_embeddings": pe, "label_embedding": le}

    # If we created a local model instance, free its GPU memory if any and return
    if created_local:
        try:
            # avoid leaving unused GPU fragments
            torch.cuda.empty_cache()
        except Exception:
            pass

    return results


# Simple timestamped logger for sweep jobs
def _log(msg: str):
    try:
        ts = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    except Exception:
        ts = str(datetime.datetime.now())
    print(f"[{ts}] {msg}")


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
# Per-class accuracy helper
# ------------------------------
def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Dict[str, float]:
    accs = {}
    for k, name in enumerate(label_names):
        mask = (y_true == k)
        if mask.sum() == 0:
            accs[name] = float("nan")
        else:
            accs[name] = float((y_pred[mask] == k).mean())
    return accs

def make_record(
    records,
    shot,
    model,
    alpha,
    temp,
    prompt_set,
    lr,
    weight_decay,
    mix_strategy,
    split_name,
    backbone,
    y_true,
    y_pred,
    scores,
    label_names,
):
    """
    Build a unified result record with optional parameters auto-filled as None.
    """
    top1 = (y_pred == y_true).mean()
    bal = balanced_acc(y_true, y_pred,  len(label_names))
    top5 = topk_acc(y_true, scores, 5)
    macro = f1_score(y_true, y_pred, average='macro')
    
    per_class = per_class_accuracy(y_true, y_pred, label_names)

    rec = {
    "shot": shot,
    "model": model,    # e.g. "prototype" or "linear"
    "alpha": alpha,
    "temp": temp,
    "prompt_set": prompt_set,
    "lr": lr,
    "weight_decay": weight_decay,
    "mix_strategy": mix_strategy,
    "split": split_name,
    "backbone": backbone,
    "top1": float(top1),
    "top5": float(top5),
    "balanced_acc": float(bal),
    "macro_f1": float(macro),
    "per_class_acc": per_class_accuracy(y_true, y_pred, label_names)
    }
    records.append(rec)
    return records

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
    Train a linear probe. Safe for CPU or GPU.
    compile_model flag will attempt torch.compile once in try/except (non-fatal if unsupported).
    """
    torch.manual_seed(seed)
    X = torch.from_numpy(X_support).float().to(device)
    y = torch.from_numpy(y_support).long().to(device)
    D = X.shape[1]
    K = int(y.max()) + 1

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = LinearProbe(D, K).to(device)
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception:
            # safe fallback if compile fails
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
    """
    Faster and more deterministic image encoder:
      - Uses vectorized DataLoader with fewer workers to avoid CPU oversubscription.
      - Writes/reads float32 .npy cache for faster IO.
      - Returns a contiguous torch.Tensor on CPU (list in calling code still works).
    """
    cache_dir = Path(cache_dir)
    if cache_prefix:
        cache_path = cache_dir / f"{cache_prefix}.npy"
        if cache_path.exists():
            try:
                arr = np.load(cache_path, mmap_mode="r")
                # convert to list of tensors to keep API compatibility
                return [torch.from_numpy(arr[i]) for i in range(arr.shape[0])]
            except Exception:
                try:
                    cache_path.unlink()
                except Exception:
                    pass

    dataset = ImageDataset(image_paths, preprocess, root=root)
    n_workers = min(8, max(0, (os.cpu_count() or 4) // 2))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
        prefetch_factor=2 if n_workers > 0 else 1
    )
    model.eval()
    embs = []
    it = loader
    if show_progress and sys.stdout.isatty():
        it = tqdm(loader, desc="encode_images", leave=False, miniters=1)
    with torch.no_grad():
        for batch in it:
            batch = [b for b in batch if b is not None]
            if not batch:
                continue
            batch_tensor = torch.stack(batch).to(device, non_blocking=True)
            # encode with the model; avoid automatic mixed precision for smaller models.
            with torch.cuda.amp.autocast(enabled=(device != "cpu")):
                emb = model.encode_image(batch_tensor)
            emb = F.normalize(emb, dim=-1)
            embs.append(emb.cpu())
    if not embs:
        return []
    emb_all = torch.cat(embs, dim=0)
    if cache_prefix:
        # write compact float32 contiguous file for fast mmap later
        np.save(str(Path(cache_dir) / f"{cache_prefix}.npy"), emb_all.numpy().astype(np.float32))
    # Keep API same: return list of tensors
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

    # ------------------------
    # Precompute per-prompt-set text embeddings if requested
    # ------------------------
    # We'll instantiate the CLIP model/tokenizer once per-backbone and reuse for text encodes
    text_embs_bundles = {"none": None}
    if args.use_prompts_for_prototype or args.use_prompts_for_linear:
        try:
            # create CLIP model + tokenizer once (use CPU for text encoding if cpu_probes True to allow parallel threads)
            text_device = "cpu" if args.cpu_probes else args.device
            # clip_model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=model_pretrained, device=args.device)
            # Always load text-encoding model on CPU for consistency and thread-safety
            clip_model, _, _ = open_clip.create_model_and_transforms(    backbone, pretrained=model_pretrained, device="cpu")

            clip_tokenizer = open_clip.get_tokenizer(backbone)
            # Do not torch.compile encode_text here (can cause repeated recompiles). Keep model in eval mode and reuse.
            clip_model.eval()
        except Exception as e:
            print(f"[warn] failed to create CLIP model for text encodes: {e}")
            clip_model, clip_tokenizer = None, None

        for pset in args.prompt_sets:
            if show_progress and sys.stdout.isatty():
                print(f"[{backbone}] computing text embeddings for prompt_set={pset}...")
            if clip_model is None:
                text_embs_bundles[pset] = None
                continue
            try:
                # reuse the same model/tokenizer and rely on caching inside get_text_embeddings
                text_embs_bundles[pset] = get_text_embeddings(
                    list(label_names),
                    prompt_set=pset,
                    model_name=backbone,
                    device=text_device,
                    batch_size=128,
                    include_common_names=False,
                    common_prompts_path=args.prompts_path,
                    pretrained=model_pretrained,
                    cache_dir=(Path(args.save_dir) / "text_cache"),
                    model=clip_model,
                    tokenizer=clip_tokenizer
                )
            except Exception as e:
                print(f"[warn] failed to get_text_embeddings for prompt_set={pset}: {e}")
                text_embs_bundles[pset] = None


    # Move val/test to torch tensors on device for evaluation
    print(f"[{backbone}] moving val/test features to device {args.device}...")
    device = args.device
    X_vl_gpu = torch.from_numpy(X_vl).float().to(device)
    X_te_gpu = torch.from_numpy(X_te).float().to(device)
    records = []

    # Zero-shot if requested
    if 0 in args.shots:
        for split_name, X_gpu, y in [("val", X_vl_gpu, y_vl), ("test", X_te_gpu, y_te)]:
            print(f"[{backbone}] zero-shot evaluation on {split_name}...")
            scores_zs = X_gpu @ torch.from_numpy(text_embeddings).float().to(device)
            yhat = torch.argmax(scores_zs, dim=1).cpu().numpy()
            records = make_record(records, 0, "zero-shot", 0.0, 0.0, "none", 0.0, 0.0, "none", split_name, backbone, y, yhat, scores_zs.cpu().numpy(), label_names)

    shot_list = sorted([s for s in args.shots if s > 0])
    shot_iter = shot_list
    if show_progress and sys.stdout.isatty():
        shot_iter = tqdm(shot_list, desc=f"shots ({backbone})", leave=False, miniters=1)

    # Decide probe device: if user requested CPU probes (default True) we use cpu to allow parallel training; else use same device
    probe_device = "cpu" if args.cpu_probes else device

    for shot in shot_iter:

        # ---- FEW-SHOT SUPPORT SELECTION (CPU) ----
        rng = np.random.default_rng(args.seed + int(shot))
        sup_idx, y_sup = sample_few_shot_indices(y_tr, K, shot, rng)

        # Build support set from cached CPU embeddings
        X_sup = X_tr[sup_idx]          # numpy (already normalized above)
        y_sup = y_sup                  # numpy

        print(f"[{backbone}]   support-set built: {X_sup.shape}")

        # # ---- IMAGE-ONLY PROTOTYPES (GPU) ----

        # ---- Continue prototype sweeps unchanged ----
        prototype_tasks = []

        
        # Efficient execution of prototype alpha sweeps in parallel across alphas
        if prototype_tasks:
            # Number of threads: limit to args.max_workers but also avoid more than needed
            max_workers = min(args.max_workers, max(1, (os.cpu_count() or 4) // 2))
            # If the main device is gpu and users didn't enable cpu_probes, still safe to CPU-parallelize these (they use numpy/text_embs)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = []
                future_to_job = {}
                for (prompt_set, mix_strategy, temp, text_embs_for_set, X_sup_loc, y_sup_loc) in prototype_tasks:
                    # For each alpha, spawn tasks for all alphas (alpha loop is independent)
                    for alpha in args.alpha_grid:
                        job_desc = f"prototype+prompts shot={shot} prompt_set={prompt_set} alpha={alpha} mix={mix_strategy} temp={temp}"
                        _log(f"[{backbone}] SUBMIT {job_desc}")
                        fut = ex.submit(_eval_prototype_alpha,
                                         X_sup_loc, y_sup_loc, K,
                                         list(label_names), text_embs_for_set, alpha, mix_strategy, temp,
                                         X_vl_gpu, y_vl, X_te_gpu, y_te, backbone, shot, prompt_set)
                        futures.append(fut)
                        future_to_job[fut] = job_desc

                # collect results
                for fut in as_completed(futures):
                    job = future_to_job.get(fut, "unknown")
                    try:
                        recs = fut.result()
                        _log(f"[{backbone}] DONE {job} -> {len(recs)} records")
                        # recs is a list of record tuples to append
                        records.extend(recs)
                    except Exception as e:
                        _log(f"[{backbone}] ERROR {job}: {e}")

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
                            future_to_job = {}
                            for alpha, lr, wd in p_jobs:
                                job_desc = f"linear+prompts shot={shot} prompt_set={prompt_set} alpha={alpha} lr={lr} wd={wd}"
                                _log(f"[{backbone}] SUBMIT {job_desc}")
                                fut = ex.submit(
                                    _train_and_eval_linear_probe_with_prompts,
                                    X_sup, y_sup,
                                    X_vl, y_vl,
                                    X_te, y_te,
                                    list(label_names),
                                    text_embs_for_set,
                                    K,
                                    alpha,
                                    args.epochs,
                                    lr,
                                    args.batch_size,
                                    probe_device,
                                    args.seed + int(shot),
                                    args.no_compile,
                                    args.temp_grid,
                                    backbone,
                                    shot,
                                    prompt_set,
                                    wd
                                )
                                futs.append(fut)
                                future_to_job[fut] = job_desc

                            for f in as_completed(futs):
                                job = future_to_job.get(f, "unknown")
                                try:
                                    recs = f.result()
                                    _log(f"[{backbone}] DONE {job} -> {len(recs)} records")
                                    records.extend(recs)
                                except Exception as e:
                                    _log(f"[{backbone}] ERROR {job}: {e}")
                    else:
                        for alpha, lr, wd in p_jobs:
                            job_desc = f"linear+prompts shot={shot} prompt_set={prompt_set} alpha={alpha} lr={lr} wd={wd}"
                            _log(f"[{backbone}] RUN {job_desc}")
                            try:
                                recs = _train_and_eval_linear_probe_with_prompts(
                                    X_sup, y_sup, X_vl, y_vl,
                                        X_te, y_te, list(label_names), text_embs_for_set, K,
                                    alpha, args.epochs, lr, args.batch_size, probe_device,
                                    args.seed + int(shot), args.no_compile, args.temp_grid, backbone, shot, prompt_set, wd)
                                _log(f"[{backbone}] DONE {job_desc} -> {len(recs)} records")
                                records.extend(recs)
                            except Exception as e:
                                _log(f"[{backbone}] ERROR {job_desc}: {e}")

    return records


def _eval_prototype_alpha(X_sup, y_sup, K, label_names, text_embs_for_set, alpha, mix_strategy, temp,
                          X_vl_gpu, y_vl, X_te_gpu, y_te, backbone, shot, prompt_set):
    """
    Evaluate a single alpha value for prototype+prompts combination.
    This function is safe to run inside a ThreadPoolExecutor because:
      - It performs text/image prototype mixing in numpy/CPU.
      - It returns records; final evaluation uses the already-moved-to-device X_vl_gpu/X_te_gpu tensors.
    """
    recs = []
    try:
        _log(f"[{backbone}] START linear+prompts shot={shot} prompt_set={prompt_set} alpha={alpha} lr={lr} wd={weight_decay}")
        # Ensure inputs are on CPU numpy arrays (they may be numpy already)
        if isinstance(X_sup, torch.Tensor):
            X_sup_np = X_sup.cpu().numpy()
        else:
            X_sup_np = np.array(X_sup, dtype=np.float32)

        # Build prototypes (numpy)
        prototypes_prompts = prototype_classifier_with_prompts(X_sup_np, y_sup, K, label_names, text_embs_for_set, alpha=alpha, mix_strategy=mix_strategy)
        # prototypes_prompts is numpy (function returns numpy) -> convert to float32
        prot_cpu = torch.from_numpy(prototypes_prompts.astype(np.float32))

        # Move prototypes to the device used by evaluation tensors (but do this outside tight loops)
        prot_on_device = prot_cpu.to(X_vl_gpu.device)

        # Evaluate on val/test using the pre-moved X_vl_gpu / X_te_gpu
        for split_name, X_gpu, y in [("val", X_vl_gpu, y_vl), ("test", X_te_gpu, y_te)]:
            # similarity and scaling
            scores = (X_gpu @ prot_on_device.T) * float(temp)
            yhat = torch.argmax(scores, axis=1).cpu().numpy()
            recs = make_record(recs, shot, f"prototype+prompts", alpha, float(temp), prompt_set, 0.0, 0.0, mix_strategy, split_name, backbone, y, yhat, scores.cpu().numpy(), label_names)
    except Exception as e:
        print(f"[warn] prototype alpha eval failed: {e}")
    return recs


    # return recs
def _train_and_eval_linear_probe_with_prompts(
    X_sup, y_sup,
    X_vl, y_vl,
    X_te, y_te,
    label_names, text_embs_for_set, K,
    alpha, epochs, lr, batch_size, device, seed, compile_flag,
    temp_grid, backbone, shot, prompt_set, weight_decay
):
    """
    Train linear probe with prompts pseudo-examples and evaluate across temps.
    Returns list of record dicts (appends via make_record).
    - Ensures all numpy arrays are converted to torch tensors BEFORE calling the model.
    - Keeps evaluation inputs on the requested device.
    """
    recs = []
    try:
        # Build and train the linear probe (this function returns a torch.nn.Module on `device`)
        model_lin_p = train_linear_probe_with_prompts(
            X_sup, y_sup, label_names, text_embs_for_set, K,
            alpha=alpha, epochs=epochs, lr=lr, batch_size=batch_size,
            device=device, seed=seed, compile_model=compile_flag, show_progress=False
        )

        model_lin_p.eval()
        for split_name, X_arr, y_arr in [
            ("val", X_vl, y_vl),
            ("test", X_te, y_te)
        ]:
            # ---- Convert X_arr (numpy or torch) to a tensor on device ----
            if isinstance(X_arr, np.ndarray):
                X_tensor = torch.from_numpy(X_arr).float().to(device)
            elif isinstance(X_arr, torch.Tensor):
                X_tensor = X_arr.float().to(device)
            else:
                X_tensor = torch.tensor(np.array(X_arr), dtype=torch.float32).to(device)

            # ---- Forward pass ----
            with torch.no_grad():
                logits_tensor = model_lin_p(X_tensor)

            logits_np = logits_tensor.detach().cpu().numpy()

            # ---- Convert y_arr properly to numpy ----
            if isinstance(y_arr, torch.Tensor):
                y_true = y_arr.cpu().numpy()
            else:
                y_true = np.asarray(y_arr)

            # ---- Temperature sweep ----
            for temp in temp_grid:
                logits_tp = logits_np * float(temp)
                yhat = logits_tp.argmax(axis=1)

                recs = make_record(
                    recs,
                    shot,
                    "linear+prompts",
                    alpha,
                    temp,
                    prompt_set,
                    lr,
                    weight_decay,
                    "none",
                    split_name,
                    backbone,
                    y_true,
                    yhat,
                    logits_tp,
                    label_names
                )

        _log(f"[{backbone}] FINISH linear+prompts shot={shot} prompt_set={prompt_set} alpha={alpha} lr={lr} wd={weight_decay} -> {len(recs)} records")

    
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
    ap.add_argument("--fast", action="store_true", help="Use fast defaults (few epochs, single worker, small hyper grids)")

    args = ap.parse_args()
    args.use_prompts_for_prototype = not args.no_prompts_for_prototype
    args.use_prompts_for_linear = not args.no_prompts_for_linear

    # Fast preset: override some defaults to make runs quick for debugging
    if getattr(args, "fast", False):
        args.epochs = 5
        args.max_workers = 1
        # shrink grids for a quick sweep
        args.lr_grid = [1e-3]
        args.wd_grid = [0.0]
        args.alpha_grid = [0.5]
        args.temp_grid = [1.0]
        # reduce prompt sets to one
        args.prompt_sets = ["ensemble"]

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

        out_overall_results = Path(args.results_dir) / "few_shot_table_all_backbones.csv"
        out_class_results = Path(args.results_dir) / "few_shot_table_all_backbones.csv"
        # with open(out_csv, "w", newline="", encoding="utf-8") as f:
        #     w = csv.writer(f)
        #     w.writerow([
        #         "shot", "model", "alpha", "temp", "prompt_set",
        #         "lr", "weight_decay", "mix_strategy",
        #         "split", "backbone",
        #         "top1", "top5", "balanced_acc", "macro_f1"
        #     ])
        #     for r in records_all:
        #         w.writerow(r)
        # print(f"Wrote results -> {out_csv}")

        # --- NEW: Save overall results as JSON ---
        out_json = Path(args.results_dir) / "few_shot_overall_results.json"
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(records_all, jf, indent=2)
        print(f"Wrote overall results -> {out_json}")

        # # --- NEW: Save per-class results (already included inside each dict) ---
        # out_json_pc = Path(args.results_dir) / "few_shot_per_class_results.json"
        # with open(out_json_pc, "w", encoding="utf-8") as jf:
        #     json.dump(records_all, jf, indent=2)
        # print(f"Wrote per-class results -> {out_json_pc}")


if __name__ == "__main__":
    main()
