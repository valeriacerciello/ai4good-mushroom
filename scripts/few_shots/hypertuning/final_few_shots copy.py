#!/usr/bin/env python3

import os
import json
import numpy as np
from pathlib import Path
import torch
from torch import nn


# Import from your sweep script
from scripts.few_shots.hypertuning.few_shot_hyper_test import (
    ensure_features,
    load_labels,
    get_text_embeddings,
    train_linear_probe,
    DEFAULT_PRETRAINED,
    DATA_ROOT, TRAIN_CSV, VAL_CSV, TEST_CSV, LABELS,
    DEFAULT_CACHE_DIR,
)

###############################################################
#                 USER-CHOSEN FINAL PARAMETERS
###############################################################

FINAL_BACKBONE = "PE-Core-bigG-14-448"
FINAL_PRETRAINED = "meta"

FINAL_MODEL_TYPE = "linear+prompts"
FINAL_SHOTS = 100
FINAL_PROMPT_SET = "delta"

FINAL_LR = 0.03
FINAL_WD = 0.0001
FINAL_EPOCHS = 200
FINAL_BATCH_SIZE = 512

BEST_ALPHA_PATH = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/results/best_alpha.json"
SAVE_PATH = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/final_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


###############################################################
#          STEP 1 — Load per-class best α values
###############################################################

def load_best_alpha(path):
    with open(path, "r") as f:
        alpha_dict = json.load(f)
    return alpha_dict


###############################################################
#        STEP 2 — Mix features with class-specific α
###############################################################

def mix_features_with_alpha(X, y, label_names, text_embs, alpha_dict):
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

        X_out[inds] = α * X[inds] + (1 - α) * txt
        X_out[inds] /= np.linalg.norm(X_out[inds], axis=1, keepdims=True)

    return X_out


###############################################################
#                   STEP 3 — Main training
###############################################################

def main():

    print("\n==============================")
    print("  TRAINING FINAL BEST MODEL")
    print("==============================\n")

    print("[1] Loading best α…")
    alpha_dict = load_best_alpha(BEST_ALPHA_PATH)

    print(f"[2] Loading labels from {LABELS}…")
    label_names, _ = load_labels(LABELS)
    K = len(label_names)

    print(f"[3] Loading cached features for backbone {FINAL_BACKBONE} …")
    X_tr, y_tr, _ = ensure_features(
        "train",
        FINAL_BACKBONE,
        DATA_ROOT,
        TRAIN_CSV,
        LABELS,
        save_dir=DEFAULT_CACHE_DIR,
        pretrained=FINAL_PRETRAINED,
    )
    X_val, y_val, _ = ensure_features(
        "val",
        FINAL_BACKBONE,
        DATA_ROOT,
        VAL_CSV,
        LABELS,
        save_dir=DEFAULT_CACHE_DIR,
        pretrained=FINAL_PRETRAINED,
    )

    # Normalize features
    X_tr = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-8)
    X_val = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-8)

    print(f"[4] Using {FINAL_SHOTS} shots per class…")
    rng = np.random.default_rng(42)
    support_idx = []
    support_labels = []
    for k in range(K):
        inds = np.where(y_tr == k)[0]
        replace = len(inds) < FINAL_SHOTS
        chosen = rng.choice(inds, size=FINAL_SHOTS, replace=replace)
        support_idx.append(chosen)
        support_labels.append(np.full(FINAL_SHOTS, k))

    support_idx = np.concatenate(support_idx)
    support_labels = np.concatenate(support_labels)

    X_sup = X_tr[support_idx]
    y_sup = support_labels

    print("[5] Loading text embeddings for prompt set 'delta'…")
    text_embs = get_text_embeddings(
        list(label_names),
        prompt_set=FINAL_PROMPT_SET,
        model_name=FINAL_BACKBONE,
        pretrained=FINAL_PRETRAINED,
        cache_dir=Path(DEFAULT_CACHE_DIR) / "text_cache",
        device="cpu",
    )

    print("[6] Mixing training features using per-class α…")
    X_sup_mixed = mix_features_with_alpha(
        X_sup, y_sup, label_names, text_embs, alpha_dict
    )

    print("[7] Training final linear probe…")
    model = train_linear_probe(
        X_sup_mixed,
        y_sup,
        epochs=FINAL_EPOCHS,
        lr=FINAL_LR,
        weight_decay=FINAL_WD,
        batch_size=FINAL_BATCH_SIZE,
        device=DEVICE,
    )

    print("[8] Saving final model →", SAVE_PATH)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    torch.save(
        {
            "linear_head": model.state_dict(),
            "backbone": FINAL_BACKBONE,
            "pretrained": FINAL_PRETRAINED,
            "prompt_set": FINAL_PROMPT_SET,
            "best_alpha": alpha_dict,
            "label_names": list(label_names),
        },
        SAVE_PATH,
    )

    print("\nDONE! Final model saved.\n")


if __name__ == "__main__":
    main()
