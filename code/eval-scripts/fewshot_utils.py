"""
Few-shot utilities shared across scripts:
- `compute_image_embeddings`: batch-encode images through a CLIP image encoder
- `build_prototypes`: build per-class centroids from train embeddings and DataFrame
- `sims_to_probs`: convert similarity matrix to softmax probabilities with temperature

These functions are small, self-contained, and intended to reduce duplicated code
across `run_prototype_knn.py`, `learned_fusion_stack.py`, and related scripts.
"""
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def compute_image_embeddings(df, preprocess, model, batch_size=32, device='cpu'):
    """Compute image embeddings for all images in `df` using `model` and `preprocess`.

    Args:
        df: pandas DataFrame with an `image_path` column
        preprocess: CLIP preprocess transform
        model: CLIP image encoder (has `encode_image`)
        batch_size: batch size for encoding
        device: device string for model

    Returns:
        numpy array of shape (N, D) with normalized embeddings
    """
    model.to(device)
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            paths = df['image_path'].iloc[i:i+batch_size].tolist()
            tensors = []
            for p in paths:
                try:
                    img = Image.open(p).convert('RGB')
                    t = preprocess(img).to(device)
                except Exception:
                    # fallback zero-tensor with expected shape (3, H, W)
                    # try common CLIP sizes: 224x224 as safe default
                    t = torch.zeros((3, 224, 224), device=device)
                tensors.append(t)
            batch = torch.stack(tensors, dim=0)
            b_emb = model.encode_image(batch).float()
            b_emb = F.normalize(b_emb, dim=1)
            embs.append(b_emb.cpu())
    if len(embs) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    embs = torch.cat(embs, dim=0).numpy()
    return embs


def build_prototypes(df_train, emb_train, labels: List[str], shots: int = 0):
    """Build class centroids (prototypes) from `emb_train` corresponding to `df_train`.

    If `shots > 0`, use up to `shots` examples per class (deterministic first-N selection).
    Returns a (C, D) numpy array of normalized centroids where C = len(labels).
    """
    label_to_idx = {l: i for i, l in enumerate(labels)}
    D = emb_train.shape[1] if emb_train.size else 0
    class_sums = np.zeros((len(labels), D), dtype=np.float32)
    class_counts = np.zeros((len(labels),), dtype=np.int32)

    # group train indices by label
    grp = {}
    for i, row in df_train.reset_index(drop=True).iterrows():
        lab = row['label']
        if lab not in label_to_idx:
            continue
        grp.setdefault(lab, []).append(i)

    if shots and shots > 0:
        for lab, idxs in grp.items():
            sel = idxs[:shots] if len(idxs) >= shots else idxs
            idx = label_to_idx[lab]
            for ii in sel:
                if ii < emb_train.shape[0]:
                    class_sums[idx] += emb_train[ii]
                    class_counts[idx] += 1
    else:
        # use all examples in emb_train order
        for i, row in df_train.reset_index(drop=True).iterrows():
            lab = row['label']
            if lab not in label_to_idx:
                continue
            idx = label_to_idx[lab]
            if i < emb_train.shape[0]:
                class_sums[idx] += emb_train[i]
                class_counts[idx] += 1

    class_counts = np.maximum(class_counts, 1)
    centroids = class_sums / class_counts[:, None]
    # normalize (avoid div by zero)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    centroids = centroids / norms
    return centroids


def sims_to_probs(embs: np.ndarray, centroids: np.ndarray, temp: float = 0.05):
    """Convert similarity (embs x centroids) to softmax probabilities with temperature.

    Args:
        embs: (N, D)
        centroids: (C, D)
        temp: temperature scalar
    Returns:
        probs: (N, C) numpy array
    """
    if embs.size == 0 or centroids.size == 0:
        return np.zeros((embs.shape[0], centroids.shape[0]), dtype=np.float32)
    sims = embs @ centroids.T
    sims = sims / temp
    sims = sims - sims.max(axis=1, keepdims=True)
    exp = np.exp(sims)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs
