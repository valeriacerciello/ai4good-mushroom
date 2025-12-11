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

from scripts.few_shots.import_n_config.eval_setup import *

###############################################################################
# Metrics
###############################################################################

def top1(pred, gt):
    """
    Compute top-1 accuracy.
    
    Args:
        pred (np.ndarray): Predicted class labels, shape (n_samples,).
        gt (np.ndarray): Ground truth class labels, shape (n_samples,).
    
    Returns:
        float: Fraction of correct predictions (0.0 to 1.0).
    """
    return float((pred == gt).mean())


def top5(logits, gt):
    """
    Compute top-5 accuracy.
    
    Args:
        logits (np.ndarray): Prediction logits/scores, shape (n_samples, n_classes).
        gt (np.ndarray): Ground truth class labels, shape (n_samples,).
    
    Returns:
        float: Fraction of samples where ground truth is in top-5 predictions (0.0 to 1.0).
    """
    top5 = np.argsort(-logits, axis=1)[:, :5]
    return float((top5 == gt[:, None]).any(axis=1).mean())


def balanced_acc(pred, gt, K):
    """
    Compute balanced accuracy (mean per-class recall).
    
    Provides fair evaluation for imbalanced datasets by averaging recall across classes.
    
    Args:
        pred (np.ndarray): Predicted class labels, shape (n_samples,).
        gt (np.ndarray): Ground truth class labels, shape (n_samples,).
        K (int): Total number of classes.
    
    Returns:
        float: Average per-class recall (0.0 to 1.0).
    """
    cm = confusion_matrix(gt, pred, labels=list(range(K)))
    per_class = np.where(cm.sum(1) > 0, cm.diagonal() / cm.sum(1), 0)
    return float(per_class.mean())


def macro_f1(pred, gt):
    """
    Compute macro-averaged F1 score (unweighted mean across classes).
    
    Args:
        pred (np.ndarray): Predicted class labels, shape (n_samples,).
        gt (np.ndarray): Ground truth class labels, shape (n_samples,).
    
    Returns:
        float: Macro F1 score (0.0 to 1.0).
    """
    return float(f1_score(gt, pred, average="macro"))


###############################################################################
# Load Final Model
###############################################################################

def load_final_model():
    """
    Load the trained final model and all necessary components for evaluation.
    
    Loads checkpoint saved by train_best_model.py and reconstructs:
    - CLIP image encoder (frozen, on device)
    - Linear classification head (on device)
    - Text embeddings (cached)
    - Per-class alpha values
    - Model configuration metadata
    
    Handles image size normalization across different CLIP model variants.
    
    Returns:
        Tuple: (clip_model, preprocess, linear_head, label_names, alpha_dict, text_embs, backbone, pretrained)
            - clip_model: CLIP image encoder in eval mode
            - preprocess: CLIP image preprocessing function
            - linear_head: Trained linear classification head
            - label_names: Ordered class label names (np.ndarray)
            - alpha_dict: Per-class image-text mixing weights (dict)
            - text_embs: Dictionary of text embeddings per label
            - backbone: Model name (str)
            - pretrained: Pretrained source (str)
    """
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
    """
    Mix image embeddings with text embeddings using per-class alpha weights.
    
    For each class, blends image features with text embeddings using the class-specific alpha value,
    then normalizes to unit norm. This reproduces the training-time feature mixing at test time.
    
    Args:
        X (np.ndarray): Image embeddings, shape (n_samples, embedding_dim).
        y (np.ndarray): Class labels, shape (n_samples,).
        label_names (np.ndarray): Ordered array of class label names.
        text_embs (Dict[str, Dict]): Text embeddings with "label_embedding" per label.
        alpha_dict (Dict[str, float]): Per-class alpha values (default 0.5 if not found).
    
    Returns:
        np.ndarray: Mixed and normalized features with same shape as input X.
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

        mixed = a * X[inds] + (1 - a) * txt
        mixed /= np.linalg.norm(mixed, axis=1, keepdims=True)

        X_out[inds] = mixed

    return X_out


###############################################################################
# Evaluation
###############################################################################

def evaluate():
    """
    Evaluate the final trained model on validation and test sets.
    
    Complete evaluation pipeline:
    1. Loads trained model and all components
    2. Loads cached image features for val/test splits
    3. Mixes features with text embeddings using per-class alpha
    4. Computes linear probe predictions
    5. Evaluates metrics: top1, top5, balanced accuracy, macro F1
    6. Saves results to JSON file
    
    Results include separate metrics for validation and test sets.
    Output saved to SAVE_JSON path.
    """
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
# Inference Pipeline
###############################################################################

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
    clip_model, preprocess, linear_head, label_names, alpha_dict, text_embs, _, _ = load_final_model()

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
        logits = linear_head(mixed_feats.squeeze(0))  # [K, K]
        # one score per class: classifier c evaluated on feature mixed for class c
        logits_per_class = torch.diag(logits)        # [K]
        pred = logits_per_class.argmax().item()

    return label_names[pred]


###############################################################################
# Main
###############################################################################

def main():
    """  
    Supports two modes via command-line arguments:
    - (no args or --eval): Evaluate on validation and test sets
    - --predict <image_path>: Run inference on a single image
    - Both can be used together: --eval --predict <image_path>
    
    Example usage:
        python eval_final_model.py                              # Evaluate only
        python eval_final_model.py --eval                       # Evaluate only
        python eval_final_model.py --predict path/to/mushroom.jpg    # Predict only
        python eval_final_model.py --eval --predict path/to/mushroom.jpg  # Both
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate on val/test sets")
    parser.add_argument("--predict", type=str, help="Predict label of an image")
    args = parser.parse_args()

    # Run both if specified, otherwise default to evaluation if neither is specified
    if args.predict:
        print(f"Predicting label for image: {args.predict}")
        label = predict(args.predict)
        print("\nPrediction:", label)
    
    if args.eval or (not args.predict and not args.eval):
        # Run evaluation if --eval is specified or if no arguments are provided
        if not args.predict:
            # Only print header if not doing prediction
            pass
        evaluate()


if __name__ == "__main__":
    main()
