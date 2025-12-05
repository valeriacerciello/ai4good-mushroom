#!/usr/bin/env python3
"""
Unified Training + Inference Script for Best Mushroom Classifier
---------------------------------------------------------------

This script:

1. Loads best hyperparameters
2. Loads OpenCLIP features + text embeddings
3. Mixes image + text features using per-class alpha
4. Trains a final linear+prompts model
5. Saves the final model
6. Provides an inference pipeline (predict(image_path))
7. Provides CLI:
       --train
       --predict <image_path>

Place in: ai4good-mushroom/train_best_model.py
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image

# Import needed components from your sweep file
from scripts.few_shots.hypertuning.few_shot_hyper_test import (
    ensure_features,
    load_labels,
    get_text_embeddings,
    train_linear_probe,
    encode_images,
    DEFAULT_CACHE_DIR,
    DATA_ROOT, TRAIN_CSV, VAL_CSV, TEST_CSV, LABELS,
)
import open_clip

###############################################################################
#                          FINAL BEST PARAMETERS
###############################################################################

# FINAL_BACKBONE = "PE-Core-bigG-14-448"
# FINAL_PRETRAINED = "meta"
FINAL_BACKBONE = "ViT-B-32-quickgelu"
FINAL_PRETRAINED = "openai"

FINAL_MODEL_TYPE = "linear+prompts"
FINAL_SHOTS = 100
FINAL_PROMPT_SET = "delta"

FINAL_LR = 0.03
FINAL_WD = 0.0000
FINAL_EPOCHS = 200
FINAL_BATCH_SIZE = 512

BEST_ALPHA_PATH = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/results/best_alpha_b32.json"
SAVE_PATH = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/final_model_b32.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
#                          Utility Functions
###############################################################################

def load_best_alpha(path):
    """Load per-class alpha values."""
    with open(path, "r") as f:
        alpha_dict = json.load(f)
    return alpha_dict


def mix_features_with_alpha(X, y, label_names, text_embs, alpha_dict):
    """Mix image features with text embeddings using per-class alpha."""
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


###############################################################################
#                         TRAINING PIPELINE
###############################################################################

def train_final_model():

    print("\n==============================")
    print("  TRAINING FINAL BEST MODEL")
    print("==============================")

    print("[1] Loading best α…")
    alpha_dict = load_best_alpha(BEST_ALPHA_PATH)

    print("[2] Loading labels…")
    label_names, _ = load_labels(LABELS)
    K = len(label_names)

    print(f"[3] Loading cached features for {FINAL_BACKBONE} …")
    X_tr, y_tr, _ = ensure_features(
        "train", FINAL_BACKBONE,
        DATA_ROOT, TRAIN_CSV, LABELS,
        save_dir=DEFAULT_CACHE_DIR,
        pretrained=FINAL_PRETRAINED,
    )
    X_val, y_val, _ = ensure_features(
        "val", FINAL_BACKBONE,
        DATA_ROOT, VAL_CSV, LABELS,
        save_dir=DEFAULT_CACHE_DIR,
        pretrained=FINAL_PRETRAINED,
    )

    # Normalize embeddings
    X_tr = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-8)
    X_val = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-8)

    print("[4] Selecting support set…")
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
    y_sup = np.concatenate(support_labels)
    X_sup = X_tr[support_idx]

    print("[5] Loading text embeddings…")
    text_embs = get_text_embeddings(
        list(label_names),
        prompt_set=FINAL_PROMPT_SET,
        model_name=FINAL_BACKBONE,
        pretrained=FINAL_PRETRAINED,
        cache_dir=Path(DEFAULT_CACHE_DIR) / "text_cache",
        device="cpu",
    )

    print("[6] Mixing features (image + text) using α…")
    X_sup_mixed = mix_features_with_alpha(
        X_sup, y_sup, label_names, text_embs, alpha_dict
    )

    print("[7] Training final linear+prompts head…")
    model = train_linear_probe(
        X_sup_mixed,
        y_sup,
        epochs=FINAL_EPOCHS,
        lr=FINAL_LR,
        weight_decay=FINAL_WD,
        batch_size=FINAL_BATCH_SIZE,
        device=DEVICE,
    )

    print("[8] Saving checkpoint:", SAVE_PATH)
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

    print("DONE — Final model saved!\n")


###############################################################################
#                        INFERENCE PIPELINE
###############################################################################

def load_final_model():

    ckpt = torch.load(SAVE_PATH, map_location="cpu")

    label_names = ckpt["label_names"]
    K = len(label_names)
    alpha_dict = ckpt["best_alpha"]

    # Load CLIP backbone
    backbone = ckpt["backbone"]
    pretrained = ckpt["pretrained"]
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        backbone, pretrained=pretrained, device=DEVICE
    )
    clip_model.eval()

    # Load linear head
    # First compute feature dimension
    dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        dim = clip_model.encode_image(dummy).shape[1]

    linear_head = nn.Linear(dim, K).to(DEVICE)
    linear_head.load_state_dict(ckpt["linear_head"])
    linear_head.eval()

    # Load text embeddings (cached)
    text_embs = get_text_embeddings(
        list(label_names),
        prompt_set=ckpt["prompt_set"],
        model_name=backbone,
        pretrained=pretrained,
        cache_dir=Path(DEFAULT_CACHE_DIR) / "text_cache",
        device="cpu",
    )

    return clip_model, preprocess, linear_head, label_names, alpha_dict, text_embs


def predict(image_path):
    """Run inference on a single image."""
    clip_model, preprocess, linear_head, label_names, alpha_dict, text_embs = load_final_model()

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_feat = clip_model.encode_image(image_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    # Build class-mixed features (image + text with alpha)
    mixed_feats = []
    for lbl in label_names:
        α = alpha_dict.get(lbl, 0.5)
        txt = text_embs[lbl]["label_embedding"]
        if isinstance(txt, torch.Tensor):
            txt = txt.to(DEVICE)
        else:
            txt = torch.from_numpy(txt).to(DEVICE)
        txt = txt / txt.norm()

        mixed = α * img_feat + (1 - α) * txt
        mixed = mixed / mixed.norm(dim=-1, keepdim=True)
        mixed_feats.append(mixed)

    mixed_feats = torch.cat(mixed_feats, dim=0).unsqueeze(0)  # [1, K, dim]

    with torch.no_grad():
        logits = linear_head(mixed_feats.squeeze(0))
        pred = logits.argmax().item()

    return label_names[pred]


###############################################################################
#                               CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train final best model")
    parser.add_argument("--predict", type=str, help="Predict label of an image")
    args = parser.parse_args()
    print("Arguments:", args)

    if args.train:
        print("Starting training of final best model...")
        train_final_model()

    if args.predict:
        print(f"Predicting label for image: {args.predict}")
        label = predict(args.predict)
        print("\nPrediction:", label)

    print("run complete.")


if __name__ == "__main__":
    main()
