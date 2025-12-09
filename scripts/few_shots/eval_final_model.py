#!/usr/bin/env python3
"""
Evaluation script for the final trained mushroom classifier.

Metrics computed:
    - Top1 accuracy
    - Top5 accuracy
    - Balanced accuracy
    - Macro F1

Datasets:
    - Validation set
    - Test set
"""

import os
import json
from pathlib import Path
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score

import torch
import open_clip

# Import from your hypertuning script
from scripts.few_shots.hypertuning.few_shot_hyper_test import (
    load_labels,
    ensure_features,
    get_text_embeddings,
    DEFAULT_CACHE_DIR,
    DATA_ROOT, TRAIN_CSV, VAL_CSV, TEST_CSV, LABELS,
)

###############################################################################
# Paths
###############################################################################

MODEL_PATH = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/final_model_b32.pt"
SAVE_JSON = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/final_model_eval_b32.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


###############################################################################
# Metrics
###############################################################################

def top1(pred, gt):
    return float((pred == gt).mean())


def top5(logits, gt):
    top5 = np.argsort(-logits, axis=1)[:, :5]
    return float((top5 == gt[:, None]).any(axis=1).mean())


def balanced_acc(pred, gt, K):
    cm = confusion_matrix(gt, pred, labels=list(range(K)))
    per_class = np.where(cm.sum(1) > 0, cm.diagonal() / cm.sum(1), 0)
    return float(per_class.mean())


def macro_f1(pred, gt):
    return float(f1_score(gt, pred, average="macro"))


###############################################################################
# Load Final Model
###############################################################################

def load_final_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    label_names = ckpt["label_names"]
    alpha_dict = ckpt["best_alpha"]
    backbone = ckpt["backbone"]
    pretrained = ckpt["pretrained"]
    prompt_set = ckpt["prompt_set"]

    print(f"Loading CLIP model: {backbone} ({pretrained})")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        backbone, pretrained=pretrained, device=DEVICE
    )
    clip_model.eval()

    # --- FIX: normalize image_size ---
    try:
        image_size = clip_model.visual.image_size
    except:
        image_size = clip_model.visual.input_resolution

    # Convert to tuple (H,W)
    if isinstance(image_size, int):
        H = W = image_size
    elif isinstance(image_size, (tuple, list)):
        # Some models provide (H, W)
        if len(image_size) == 1:
            H = W = image_size[0]
        else:
            H, W = image_size
    else:
        raise ValueError(f"Unknown image_size format: {image_size}")

    # Create dummy image for dimension inference
    dummy = torch.zeros(1, 3, H, W).to(DEVICE)

    # Get embedding dim
    with torch.no_grad():
        dim = clip_model.encode_image(dummy).shape[1]

    # Load linear classifier head
    linear_head = torch.nn.Linear(dim, len(label_names)).to(DEVICE)

    # Fix classifier weights keys: remove "linear." prefix
    saved_head = ckpt["linear_head"]
    fixed_state = {}
    for k, v in saved_head.items():
        if k.startswith("linear."):
            new_k = k[len("linear."):]
        else:
            new_k = k
        fixed_state[new_k] = v

    linear_head.load_state_dict(fixed_state)
    linear_head.eval()


    # Load cached text embeddings
    text_embs = get_text_embeddings(
        list(label_names),
        prompt_set=prompt_set,
        model_name=backbone,
        pretrained=pretrained,
        cache_dir=Path(DEFAULT_CACHE_DIR) / "text_cache",
        device="cpu",
    )

    return clip_model, preprocess, linear_head, label_names, alpha_dict, text_embs, backbone, pretrained



###############################################################################
# Mixing features
###############################################################################

def mix_features(X, y, label_names, text_embs, alpha_dict):
    X_out = np.zeros_like(X)

    for k, lbl in enumerate(label_names):
        inds = (y == k)
        if not np.any(inds):
            continue

        α = alpha_dict.get(lbl, 0.5)
        txt = text_embs[lbl]["label_embedding"]

        if isinstance(txt, torch.Tensor):
            txt = txt.cpu().numpy()

        txt = txt / (np.linalg.norm(txt) + 1e-8)

        mixed = α * X[inds] + (1 - α) * txt
        mixed /= np.linalg.norm(mixed, axis=1, keepdims=True)

        X_out[inds] = mixed

    return X_out


###############################################################################
# Evaluation
###############################################################################

def evaluate():

    print("\n===== Loading final model =====")
    clip_model, preprocess, linear_head, label_names, alpha_dict, text_embs, backbone, pretrained = load_final_model()
    K = len(label_names)

    print("\n===== Loading cached image features (val/test) =====")

    X_val, y_val, _ = ensure_features(
        "val",
        backbone,
        DATA_ROOT, VAL_CSV, LABELS,
        save_dir=DEFAULT_CACHE_DIR,
        pretrained=pretrained,
    )
    X_test, y_test, _ = ensure_features(
        "test",
        backbone,
        DATA_ROOT, TEST_CSV, LABELS,
        save_dir=DEFAULT_CACHE_DIR,
        pretrained=pretrained,
    )

    # Normalize
    X_val = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-8)
    X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)

    print("Mixing features with alpha…")
    X_val_mix = mix_features(X_val, y_val, label_names, text_embs, alpha_dict)
    X_test_mix = mix_features(X_test, y_test, label_names, text_embs, alpha_dict)

    # Convert to torch
    X_val_t = torch.from_numpy(X_val_mix).float().to(DEVICE)
    X_test_t = torch.from_numpy(X_test_mix).float().to(DEVICE)

    with torch.no_grad():
        logits_val = linear_head(X_val_t).cpu().numpy()
        logits_test = linear_head(X_test_t).cpu().numpy()

    pred_val = logits_val.argmax(axis=1)
    pred_test = logits_test.argmax(axis=1)

    print("\n===== Metrics =====")

    results = {
        "val": {
            "top1": top1(pred_val, y_val),
            "top5": top5(logits_val, y_val),
            "balanced_acc": balanced_acc(pred_val, y_val, K),
            "macro_f1": macro_f1(pred_val, y_val),
        },
        "test": {
            "top1": top1(pred_test, y_test),
            "top5": top5(logits_test, y_test),
            "balanced_acc": balanced_acc(pred_test, y_test, K),
            "macro_f1": macro_f1(pred_test, y_test),
        }
    }

    print(json.dumps(results, indent=2))

    print("\nSaving:", SAVE_JSON)
    with open(SAVE_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDONE.\n")


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    evaluate()
