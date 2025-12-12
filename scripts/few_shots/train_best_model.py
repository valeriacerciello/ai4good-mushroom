#!/usr/bin/env python3
"""
Training Script for Best Mushroom Classifier
---------------------------------------------

This script:

1. Loads best hyperparameters
2. Loads OpenCLIP features + text embeddings
3. Mixes image + text features using per-class alpha
4. Trains a final linear+prompts model
5. Saves the final model

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
        - Backbone: FINAL_BACKBONE (b32 or bigG)
        - Prompt Set: FINAL_PROMPT_SET (delta)
        - Few-shot Samples: FINAL_SHOTS (100)
        - Learning Rate: FINAL_LR (0.03)
        - Weight Decay: FINAL_WD (0 or 1e-4)
    
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

def main():
    """   
    Trains the final best model.
    
    Example usage:
        python train_best_model.py
    """
    print("Starting training of final best model...")
    train_final_model()
    print("Training complete.")


if __name__ == "__main__":
    main()
