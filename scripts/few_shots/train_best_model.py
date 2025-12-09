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

from scripts.few_shots.import_n_config.train_setup import *

###############################################################################
#                          Utility Functions
###############################################################################

def load_best_alpha(path):
    """Load per-class alpha values."""
    with open(path, "r") as f:
        alpha_dict = json.load(f)
    return alpha_dict


def mix_features_with_alpha(X, y, label_names, text_embs, alpha_dict):
    """
    Mix image embeddings with text embeddings using per-class alpha weights.
    
    For each class, blends image support features with the corresponding text embedding using
    the class-specific alpha value, then applies L2 normalization.
    
    Args:
        X (np.ndarray): Image embeddings, shape (n_samples, embedding_dim).
        y (np.ndarray): Class labels, shape (n_samples,).
        label_names (np.ndarray): Ordered list/array of class label names.
        text_embs (Dict[str, Dict]): Text embeddings dict with "label_embedding" per label.
        alpha_dict (Dict[str, float]): Per-class alpha values (default 0.5 if not found).
    
    Returns:
        np.ndarray: Mixed features with same shape as X, normalized to unit norm.
    """
    X_out = np.zeros_like(X)

    for k, lbl in enumerate(label_names):
        inds = (y == k)
        if not np.any(inds):
            continue

        a = alpha_dict.get(lbl, 0.5)

        txt = text_embs[lbl]["label_embedding"]
        if isinstance(txt, torch.Tensor):
            txt = txt.cpu().numpy()
        txt = txt / (np.linalg.norm(txt) + 1e-8)

        X_out[inds] = a * X[inds] + (1 - a) * txt
        X_out[inds] /= np.linalg.norm(X_out[inds], axis=1, keepdims=True)

    return X_out


###############################################################################
#                         TRAINING PIPELINE
###############################################################################

def train_final_model():

    """
    Train the final best mushroom classifier using optimized hyperparameters.
    
    Orchestrates the complete training pipeline:
    1. Loads best per-class alpha values
    2. Loads class label names
    3. Loads cached image features (train/val splits)
    4. Selects few-shot support set (FINAL_SHOTS per class)
    5. Loads and mixes text embeddings
    6. Trains final linear probe head
    7. Saves model checkpoint
    
    The final model uses:
        - Backbone: FINAL_BACKBONE (ViT-B-32-quickgelu)
        - Prompt Set: FINAL_PROMPT_SET (delta)
        - Few-shot Samples: FINAL_SHOTS (100)
        - Learning Rate: FINAL_LR (0.03)
        - Weight Decay: FINAL_WD (0.0)
    
    Output checkpoint saved to SAVE_PATH contains:
        - linear_head state_dict
        - backbone and pretrained info
        - best_alpha dictionary
        - label_names
    """
    print("\n==============================")
    print("  TRAINING FINAL BEST MODEL")
    print("==============================")

    print("[1] Loading best a…")
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

    print("[6] Mixing features (image + text) using a…")
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

    """
    Load the trained final model with all necessary components.
    
    Loads the checkpoint and reconstructs:
    - CLIP image encoder (frozen)
    - Linear classification head
    - Text embeddings (cached)
    - Per-class alpha values
    
    Returns:
        Tuple: (clip_model, preprocess, linear_head, label_names, alpha_dict, text_embs)
            - clip_model: CLIP image encoder in eval mode
            - preprocess: CLIP image preprocessing function
            - linear_head: Trained linear classification head
            - label_names: Ordered class label names
            - alpha_dict: Per-class image-text mixing weights
            - text_embs: Dict of text embeddings per label
    """
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
    """
    Run inference on a single image and return predicted mushroom class.
    
    Combines CLIP image encoding with per-class text embeddings using learned alpha values.
    Predicts the class with highest logit from the linear head.
    
    Args:
        image_path (str): Path to image file to classify.
    
    Returns:
        str: Predicted mushroom class label.
    """
    clip_model, preprocess, linear_head, label_names, alpha_dict, text_embs = load_final_model()

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_feat = clip_model.encode_image(image_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    # Build class-mixed features (image + text with alpha)
    mixed_feats = []
    for lbl in label_names:
        a = alpha_dict.get(lbl, 0.5)
        txt = text_embs[lbl]["label_embedding"]
        if isinstance(txt, torch.Tensor):
            txt = txt.to(DEVICE)
        else:
            txt = torch.from_numpy(txt).to(DEVICE)
        txt = txt / txt.norm()

        mixed = a * img_feat + (1 - a) * txt
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
    """
    CLI entry point for model training and inference.
    
    Supports two modes via command-line arguments:
    - --train: Train and save the final model
    - --predict <image_path>: Run inference on a single image
    
    Example usage:
        python train_best_model.py --train
        python train_best_model.py --predict path/to/mushroom.jpg
    """
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
