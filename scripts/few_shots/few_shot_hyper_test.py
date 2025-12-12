#!/usr/bin/env python3

# ===== HARD DISABLE TORCH DYNAMO BEFORE ANY PYTORCH IMPORTS =====
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import hashlib
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True
# =================================================================

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

"""
Giant Hyperparameter sweep of open CLIP few shots with prompts
"""

from scripts.few_shots.import_n_config.hyper_setup import *

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
    """
    Build a list of text prompts for a given class label using different prompt strategies.
    
    Args:
        label (str): The class label/mushroom name to generate prompts for.
        prompt_set (str): Type of prompt set to use. Options are:
            - "v1": Short template-based prompts
            - "names": Prompts using scientific and common names
            - "delta": Custom prompts loaded from JSON file
            - "ensemble": Combination of all prompt types
        include_common_names (bool): Whether to include common name variants in prompts.
        mg (Optional[Any]): Metadata/knowledge graph object containing species information.
    
    Returns:
        List[str]: A list of text prompts for the given label.
    
    Raises:
        ValueError: If prompt_set is unknown or if delta prompts cannot be loaded.
    """
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
    Compute and cache CLIP text embeddings for multiple labels with various prompt strategies.
    
    Generates text prompts for each label and encodes them using a CLIP text encoder. Results are
    cached using SHA-1 hashes for fast reloading. Supports prompt reuse across multiple model instances.

    Args:
        labels (List[str]): List of class labels to compute embeddings for.
        prompt_set (str): Type of prompt set ("ensemble", "v1", "names", "delta").
        model_name (str): CLIP model name (e.g., "ViT-B-32").
        device (Optional[str]): Device to run encoding on ("cpu" or "cuda"). Defaults to "cuda" if available.
        batch_size (int): Batch size for encoding prompts (default 128).
        include_common_names (bool): Whether to include common name variants.
        common_prompts_path (Optional[str]): Path to JSON file with custom prompts.
        pretrained (str): Pretrained weights source ("openai", "webli", etc.).
        cache_dir (Optional[Path]): Directory to cache embeddings. Creates if doesn't exist.
        model: Pre-created CLIP model (optional, for reuse across calls).
        tokenizer: Pre-created CLIP tokenizer (optional, for reuse across calls).
    
    Returns:
        Dict[str, Any]: Dictionary mapping label names to embedding data:
            - "prompts": List[str] - The actual text prompts used
            - "prompt_embeddings": Tensor (n_prompts, embedding_dim) - Individual prompt embeddings
            - "label_embedding": Tensor (embedding_dim,) - Mean embedding across prompts
    
    Note:
        Caches embeddings using SHA-1 hash keys for deterministic cache hits. Text encoding 
        is done on CPU for thread-safety when parallelizing across multiple workers.
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
                try:
                    cache_path.unlink()
                except Exception:
                    pass

        # Encode prompts in batches
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                toks = tokenizer(prompts[i:i + batch_size])
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

    if created_local:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return results


def _log(msg: str):
    """
    Print a timestamped log message to console.
    
    Args:
        msg (str): The message to log.
    
    Returns:
        None. Prints to stdout with ISO format timestamp prefix.
    """
    try:
        ts = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    except Exception:
        ts = str(datetime.datetime.now())
    print(f"[{ts}] {msg}")


# ------------------------------
# Labels & prompts
# ------------------------------
def load_labels(labels_tsv: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Load class labels and IDs from a TSV file.
    
    Args:
        labels_tsv (str): Path to TSV file with format: id<tab>label_name (one per line).
    
    Returns:
        Tuple[np.ndarray, Dict[str, int]]: 
            - label_names: Array of label names indexed by class ID
            - name_to_id: Dictionary mapping label names to their integer IDs
    """
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
    """
    Load custom prompts from a JSON file or use default prompts.
    
    Args:
        prompts_path (Optional[str]): Path to JSON file mapping label names to prompt lists.
        label_names (List[str]): List of all possible label names for fallback defaults.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping each label to a list of text prompts.
                              Uses custom prompts if available, otherwise defaults to simple templates.
    """
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
    """
    Load or compute and cache image embeddings for a dataset split using a CLIP backbone.
    
    Args:
        split (str): Dataset split name ("train", "val", "test").
        backbone (str): CLIP model name (e.g., "ViT-B-32").
        data_root (str): Path to image data directory.
        csv_path (str): Path to CSV file with columns [filepath, label].
        labels_tsv (str): Path to TSV file with class labels.
        save_dir (str): Directory to cache computed features.
        pretrained (str): Pretrained weights source ("openai", "webli", etc.).
    
    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
            - X: Image embeddings array of shape (n_samples, embedding_dim)
            - y: Label indices array of shape (n_samples,)
            - paths: Optional image file paths array (or None)
    
    Note:
        If cache doesn't exist, calls external dump_features.py script to compute embeddings.
    """
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
    """
    Compute top-K accuracy given true labels and prediction scores.
    
    Args:
        y_true (np.ndarray): True class labels, shape (n_samples,).
        scores (np.ndarray): Prediction scores/logits, shape (n_samples, n_classes).
        k (int): Number of top predictions to consider.
    
    Returns:
        float: Fraction of samples where true label is in top-K predictions (0.0 to 1.0).
    """
    k = min(k, scores.shape[1])
    topk = np.argsort(-scores, axis=1)[:, :k]
    return float((topk == y_true[:, None]).any(axis=1).mean())

def balanced_acc(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
    """
    Compute balanced accuracy (mean of per-class recall rates).
    
    Args:
        y_true (np.ndarray): True class labels, shape (n_samples,).
        y_pred (np.ndarray): Predicted class labels, shape (n_samples,).
        K (int): Total number of classes.
    
    Returns:
        float: Average per-class recall, weighted equally across classes (0.0 to 1.0).
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(K))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = np.where(cm.sum(1) > 0, np.diag(cm) / cm.sum(1), 0.0)
    return float(np.nanmean(per_class))

def sample_few_shot_indices(y_train: np.ndarray, K: int, n_shot: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample few-shot support indices stratified by class.
    
    Args:
        y_train (np.ndarray): Training set labels, shape (n_train_samples,).
        K (int): Total number of classes.
        n_shot (int): Number of samples to select per class.
        rng (np.random.Generator): NumPy random number generator for reproducibility.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - support_indices: Array of selected training indices
            - support_labels: Corresponding class labels (0 to K-1)
    
    Raises:
        ValueError: If any class has no training examples.
    """
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
    """
    Compute per-class accuracy for each label.
    
    Args:
        y_true (np.ndarray): True class labels, shape (n_samples,).
        y_pred (np.ndarray): Predicted class labels, shape (n_samples,).
        label_names (List[str]): Ordered list of class label names.
    
    Returns:
        Dict[str, float]: Dictionary mapping label names to their individual accuracy values.
    """
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
    prompt_set,
    lr,
    weight_decay,
    split_name,
    backbone,
    y_true,
    y_pred,
    scores,
    label_names,
):
    """
    Build a unified evaluation result record with metrics.
    
    Computes multiple metrics (top1, top5, balanced accuracy, macro F1) and per-class accuracies,
    then appends a comprehensive result dictionary to the records list.
    
    Args:
        records: List to append the result record to.
        shot (int): Number of few-shot samples used.
        model (str): Model type ("prototype", "linear+prompts", "zero-shot").
        alpha (float): Prompt mixing weight (0=image only, 1=text only).
        prompt_set (str): Type of prompt set used.
        lr (float): Learning rate (if linear probe).
        weight_decay (float): Weight decay (if linear probe).
        split_name (str): Dataset split ("train", "val", "test").
        backbone (str): CLIP model backbone name.
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        scores (np.ndarray): Prediction scores/logits.
        label_names (List[str]): Ordered list of class names.
    
    Returns:
        List: Updated records list with the new result appended.
    """
    top1 = (y_pred == y_true).mean()
    bal = balanced_acc(y_true, y_pred,  len(label_names))
    top5 = topk_acc(y_true, scores, 5)
    macro = f1_score(y_true, y_pred, average='macro')
    
    rec = {
    "shot": shot,
    "model": model,    # e.g. "prototype" or "linear"
    "alpha": alpha,
    "prompt_set": prompt_set,
    "lr": lr,
    "weight_decay": weight_decay,
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
    """
    Build prototype classifiers by computing class mean embeddings.
    
    Args:
        X_support (np.ndarray): Support set embeddings, shape (n_support, embedding_dim).
        y_support (np.ndarray): Support set labels, shape (n_support,).
        K (int): Total number of classes.
    
    Returns:
        np.ndarray: Prototype vectors normalized to unit norm, shape (K, embedding_dim).
    """
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
    Build prototypes by mixing image support embeddings with text embeddings.
    
    Args:
        X_support (np.ndarray): Support set embeddings, shape (n_support, embedding_dim).
        y_support (np.ndarray): Support set labels, shape (n_support,).
        K (int): Total number of classes.
        label_names (List[str]): Ordered list of class label names.
        text_embs (Dict[str, Any]): Text embeddings dict with "label_embedding" for each label.
        alpha (float): Mixing weight between image and text features (0=image only, 1=text only).
    
    Returns:
        np.ndarray: Mixed prototype vectors, shape (K, embedding_dim).
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
        prototypes[k] = combined
    return prototypes

class LinearProbe(nn.Module):
    """
    Simple linear classifier head for few-shot learning.
    
    Maps embedding vectors to class logits via a single fully-connected layer.
    
    Attributes:
        linear (nn.Linear): Linear layer from embedding dimension to number of classes.
    """
    def __init__(self, dim: int, K: int):
        """
        Initialize the linear probe.
        
        Args:
            dim (int): Input embedding dimension.
            K (int): Number of output classes.
        """
        super().__init__()
        self.linear = nn.Linear(dim, K)

    def forward(self, x):
        """
        Forward pass through linear layer.
        
        Args:
            x: Input embeddings of shape (..., dim).
        
        Returns:
            Class logits of shape (..., K).
        """
        return self.linear(x)

def train_linear_probe(X_support: np.ndarray, y_support: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                       epochs: int = 200, lr: float = 1e-3, batch_size: int = 32, device: str = "cpu", seed: int = 42,
                       compile_model: bool = False, show_progress: bool = False, weight_decay: float = 1e-4) -> nn.Module:
    """
    Train a linear probe classifier on support set embeddings.
    
    Args:
        X_support (np.ndarray): Support set embeddings, shape (n_support, embedding_dim).
        y_support (np.ndarray): Support set labels, shape (n_support,).
        X_val (Optional[np.ndarray]): Validation set embeddings for early stopping.
        y_val (Optional[np.ndarray]): Validation set labels.
        epochs (int): Number of training epochs (default 200).
        lr (float): Learning rate for AdamW optimizer (default 1e-3).
        batch_size (int): Batch size for training (default 32).
        device (str): Device to train on ("cpu" or "cuda", default "cpu").
        seed (int): Random seed for reproducibility.
        compile_model (bool): Whether to use torch.compile for speedup (default False).
        show_progress (bool): Whether to show progress bar (default False).
        weight_decay (float): Weight decay for AdamW (default 1e-4).
    
    Returns:
        nn.Module: Trained LinearProbe model with best validation state restored.
    
    Note:
        Uses early stopping with best validation loss tracking. Supports both CPU and GPU training.
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
    """
    Train a linear probe with text-guided feature augmentation.
    
    Mixes support set embeddings with text embeddings per-class before training the linear probe.
    This approach provides pseudo-examples of text-visual alignment for each class.
    
    Args:
        X_support (np.ndarray): Support set embeddings, shape (n_support, embedding_dim).
        y_support (np.ndarray): Support set labels.
        label_names (List[str]): List of class label names.
        text_embs (Dict): Text embeddings with "label_embedding" field per label.
        K (int): Number of classes.
        alpha (float): Mixing weight between image and text features (default 0.5).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        device (str): Device to train on ("cpu" or "cuda").
        seed (int): Random seed.
        compile_model (bool): Whether to use torch.compile.
        show_progress (bool): Whether to show progress.
    
    Returns:
        nn.Module: Trained linear probe model.
    """
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
    """
    Compute zero-shot classification scores using prompt embeddings.
    
    Caches per-label mean text embeddings and returns a score matrix for zero-shot prediction.
    
    Args:
        label_names (List[str]): List of class label names.
        backbone (str): CLIP model name.
        pretrained (str): Pretrained weights source.
        device (str): Device for computation.
        prompts_path (Optional[str]): Path to custom prompts JSON.
        cache_dir (str): Directory to cache embeddings.
        batch_size (int): Batch size for encoding.
    
    Returns:
        np.ndarray: Mean text embeddings shape (embedding_dim, n_classes) - transposed for efficient scoring.
    """
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
    """
    PyTorch Dataset for loading images from a list of file paths.
    
    Loads images and applies CLIP preprocessing transformations.
    
    Attributes:
        paths (List[str]): List of image file paths.
        preprocess: CLIP preprocessing function.
        root (str): Root directory for relative paths.
    """
    def __init__(self, image_paths: List[str], preprocess, root: str = "."):
        """
        Initialize the image dataset.
        
        Args:
            image_paths (List[str]): List of image file paths (absolute or relative to root).
            preprocess: CLIP preprocessing function.
            root (str): Root directory for resolving relative paths.
        """
        self.paths = image_paths
        self.preprocess = preprocess
        self.root = root

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Load and preprocess an image.
        
        Args:
            idx (int): Index of the image to load.
        
        Returns:
            Tensor: Preprocessed image tensor, or None if loading fails.
        """
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
    Encode a list of images into CLIP embeddings with caching and batching.
    
    Uses a DataLoader to efficiently encode images in batches. Results can be cached to disk
    for fast reloading on subsequent calls. Handles memory efficiently with fp32 conversion.
    
    Args:
        model: CLIP image encoder model.
        preprocess: CLIP preprocessing function.
        image_paths (List[str]): List of image file paths.
        device (str): Device for encoding ("cuda" or "cpu").
        batch_size (int): Batch size for encoding (default 32).
        cache_prefix (Optional[str]): Prefix for cache file (e.g., "train"). If provided, cache is used/saved.
        cache_dir (str): Directory to store cache files.
        root (str): Root directory for resolving relative image paths.
        show_progress (bool): Whether to show progress bar (default False).
    
    Returns:
        List[torch.Tensor]: List of image embeddings, one per input image.
    
    Note:
        Caches are saved as float32 .npy files for fast mmap loading. Invalid images return None.
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
            with torch.cuda.amp.autocast(enabled=(device != "cpu")):
                emb = model.encode_image(batch_tensor)
            emb = F.normalize(emb, dim=-1)
            embs.append(emb.cpu())
    if not embs:
        return []
    emb_all = torch.cat(embs, dim=0)
    if cache_prefix:
        np.save(str(Path(cache_dir) / f"{cache_prefix}.npy"), emb_all.numpy().astype(np.float32))
    return list(emb_all)


# ------------------------------
# Prompt-pool helpers
# ------------------------------
def get_text_embeddings_bundle_from_clip(model_name: str, model, tokenizer, label_prompts: Dict[str, List[str]],
                                        device: str = "cuda", batch_size: int = 128, show_progress: bool = False):
    """
    Encode all prompts for all labels using a CLIP text encoder.
    
    Args:
        model_name (str): CLIP model name (for documentation).
        model: CLIP text encoder model.
        tokenizer: CLIP tokenizer.
        label_prompts (Dict[str, List[str]]): Dictionary mapping label names to lists of prompts.
        device (str): Device for encoding (default "cuda").
        batch_size (int): Batch size for encoding (default 128).
        show_progress (bool): Whether to show progress bar (default False).
    
    Returns:
        Dict[str, Dict]: Dictionary mapping labels to embedding bundles with keys:
            - "prompts": List of text prompts
            - "prompt_embeddings": Tensor of individual prompt embeddings
            - "label_embedding": Mean embedding across prompts
    """
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
    """
    Compute cosine similarity scores between images and class prototypes in vectorized form.
    
    Args:
        image_embeddings (List[torch.Tensor]): List of image embedding vectors.
        mean_dict (Dict[str, torch.Tensor]): Dictionary mapping labels to prototype embeddings.
        labels (List[str]): Ordered list of label names corresponding to mean_dict keys.
    
    Returns:
        List[str]: Predicted label for each image (most similar prototype).
    """
    img_mat = torch.stack([e for e in image_embeddings if e is not None])
    mean_mat = torch.stack([mean_dict[l] for l in labels])
    sims = img_mat @ mean_mat.T
    preds_idx = sims.argmax(dim=1)
    return [labels[i] for i in preds_idx]

def acc_from_preds(preds: List[str], trues: List[str]) -> float:
    """
    Compute accuracy from prediction and ground truth labels.
    
    Args:
        preds (List[str]): Predicted labels.
        trues (List[str]): Ground truth labels.
    
    Returns:
        float: Fraction of correct predictions (0.0 to 1.0).
    """
    return float(sum(p == t for p, t in zip(preds, trues)) / len(trues))

# ------------------------------
# High-level per-backbone evaluation (cached features) with nested progress bars
# ------------------------------
def evaluate_backbone(backbone: str, args: argparse.Namespace, label_names: List[str], K: int, show_progress: bool = False):
    """
    Main evaluation pipeline for a single CLIP backbone across multiple few-shot configurations.
    
    Performs comprehensive hyperparameter sweeps including:
    - Zero-shot evaluation (if shot=0)
    - Few-shot prototype classifiers with and without text prompts
    - Linear probe classifiers with text-augmented training
    - Temperature and alpha (mixing weight) grids
    
    Uses parallelization where safe:
    - Alpha sweeps parallelized via ThreadPoolExecutor (CPU-bound)
    - Linear probe (lr, wd) sweeps parallelized if probe_device=="cpu"
    
    Args:
        backbone (str): CLIP model name (e.g., "ViT-B-32").
        args (argparse.Namespace): Configuration object with fields:
            - data_root, train/val/test csv paths, labels tsv
            - device, save_dir, results_dir
            - shots, alpha_grid, lr_grid, wd_grid
            - prompt_sets, use_prompts_for_prototype, use_prompts_for_linear
            - cpu_probes, max_workers, epochs, batch_size, seed
        label_names (List[str]): Ordered list of class names.
        K (int): Number of classes.
        show_progress (bool): Whether to display progress bars.
    
    Returns:
        List[Dict]: List of result records, one per configuration tested.
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
    text_embs_bundles = {"none": None}
    if args.use_prompts_for_prototype or args.use_prompts_for_linear:
        try:
            text_device = "cpu" if args.cpu_probes else args.device
            clip_model, _, _ = open_clip.create_model_and_transforms(    backbone, pretrained=model_pretrained, device="cpu")
            clip_tokenizer = open_clip.get_tokenizer(backbone)
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
            records = make_record(records, 0, "zero-shot", 0.0, 0.0, 0.0, "none", split_name, backbone, y, yhat, scores_zs.cpu().numpy(), label_names)

    shot_list = sorted([s for s in args.shots if s > 0])
    shot_iter = shot_list
    if show_progress and sys.stdout.isatty():
        shot_iter = tqdm(shot_list, desc=f"shots ({backbone})", leave=False, miniters=1)

    probe_device = "cpu" if args.cpu_probes else device

    for shot in shot_iter:
        # ---- FEW-SHOT SUPPORT SELECTION (CPU) ----
        rng = np.random.default_rng(args.seed + int(shot))
        sup_idx, y_sup = sample_few_shot_indices(y_tr, K, shot, rng)
        X_sup = X_tr[sup_idx]         
        y_sup = y_sup                  

        print(f"[{backbone}]   support-set built: {X_sup.shape}")
        if args.use_prompts_for_prototype:
            for prompt_set in args.prompt_sets:
                text_embs_for_set = text_embs_bundles.get(prompt_set)
                if text_embs_for_set is None:
                    continue
                max_workers = min(args.max_workers, max(1, (os.cpu_count() or 4) // 2))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = []
                    future_to_job = {}
                    # For each alpha, spawn tasks for all alphas (alpha loop is independent)
                    for alpha in args.alpha_grid:
                        job_desc = f"prototype+prompts shot={shot} prompt_set={prompt_set} alpha={alpha}"
                        _log(f"[{backbone}] SUBMIT {job_desc}")
                        fut = ex.submit(_eval_prototype_alpha,
                                        X_sup, y_sup, K,
                                        list(label_names), text_embs_for_set, alpha,
                                        X_vl_gpu, y_vl, X_te_gpu, y_te, backbone, shot, prompt_set)
                        futures.append(fut)
                        future_to_job[fut] = job_desc

                    # collect results
                    for fut in as_completed(futures):
                        job = future_to_job.get(fut, "unknown")
                        try:
                            recs = fut.result()
                            _log(f"[{backbone}] DONE {job} -> {len(recs)} records")
                            records.extend(recs)
                        except Exception as e:
                            _log(f"[{backbone}] ERROR {job}: {e}")

        # Prompt-aware probe sweep
        if args.use_prompts_for_linear:
            for prompt_set in args.prompt_sets:
                text_embs_for_set = text_embs_bundles.get(prompt_set, None)
                if text_embs_for_set is None:
                    continue
                for alpha in args.alpha_grid:
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
                                    args.seed + int(shot), args.no_compile, backbone, shot, prompt_set, wd)
                                _log(f"[{backbone}] DONE {job_desc} -> {len(recs)} records")
                                records.extend(recs)
                            except Exception as e:
                                _log(f"[{backbone}] ERROR {job_desc}: {e}")
    return records


def _eval_prototype_alpha(X_sup, y_sup, K, label_names, text_embs_for_set, alpha,
                          X_vl_gpu, y_vl, X_te_gpu, y_te, backbone, shot, prompt_set):
    """
    Evaluate a single alpha for prototype+prompts classifier.
    
    Thread-safe function designed to run inside ThreadPoolExecutor. Computes prototypes by mixing
    image and text embeddings, then evaluates on validation and test sets.
    
    Args:
        X_sup (np.ndarray or torch.Tensor): Support set embeddings.
        y_sup (np.ndarray): Support set labels.
        K (int): Number of classes.
        label_names (List[str]): Class label names.
        text_embs_for_set (Dict): Text embeddings for the current prompt set.
        alpha (float): Image-text mixing weight.
        X_vl_gpu (torch.Tensor): Validation features on device.
        y_vl (np.ndarray): Validation labels.
        X_te_gpu (torch.Tensor): Test features on device.
        y_te (np.ndarray): Test labels.
        backbone (str): Model name for logging.
        shot (int): Few-shot count for logging.
        prompt_set (str): Prompt set name for logging.
    
    Returns:
        List[Dict]: Evaluation result records.
    """
    recs = []
    try:
        _log(f"[{backbone}] START prototype+prompts shot={shot} prompt_set={prompt_set} alpha={alpha}")
        if isinstance(X_sup, torch.Tensor):
            X_sup_np = X_sup.cpu().numpy()
        else:
            X_sup_np = np.array(X_sup, dtype=np.float32)

        prototypes_prompts = prototype_classifier_with_prompts(X_sup_np, y_sup, K, label_names, text_embs_for_set, alpha=alpha)
        prot_cpu = torch.from_numpy(prototypes_prompts.astype(np.float32))
        prot_on_device = prot_cpu.to(X_vl_gpu.device)

        # Evaluate on val/test using the pre-moved X_vl_gpu / X_te_gpu
        for split_name, X_gpu, y in [("val", X_vl_gpu, y_vl), ("test", X_te_gpu, y_te)]:
            scores = (X_gpu @ prot_on_device.T)
            yhat = torch.argmax(scores, axis=1).cpu().numpy()
            recs = make_record(recs, shot, f"prototype+prompts", alpha, prompt_set, 0.0, 0.0, split_name, backbone, y, yhat, scores.cpu().numpy(), label_names)
            recs = make_record(recs, shot, f"prototype+prompts", alpha, prompt_set, 0.0, 0.0, split_name, backbone, y, yhat, scores.cpu().numpy(), label_names)
    except Exception as e:
        print(f"[warn] prototype alpha eval failed: {e}")
    return recs

def _train_and_eval_linear_probe_with_prompts(
    X_sup, y_sup,
    X_vl, y_vl,
    X_te, y_te,
    label_names, text_embs_for_set, K,
    alpha, epochs, lr, batch_size, device, seed, compile_flag,
    backbone, shot, prompt_set, weight_decay
):
    """
    Train a linear probe with text-augmented features.
    
    Thread-safe function for parallel execution. Trains a linear probe on mixed image+text features,
    then evaluates on validation/test with scaling applied to logits.
    
    Args:
        X_sup, y_sup: Support set embeddings and labels.
        X_vl, y_vl: Validation set embeddings and labels.
        X_te, y_te: Test set embeddings and labels.
        label_names (List[str]): Class label names.
        text_embs_for_set (Dict): Text embeddings for mixing.
        K (int): Number of classes.
        alpha (float): Image-text mixing weight.
        epochs (int): Training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        device (str): Training device.
        seed (int): Random seed.
        compile_flag (bool): Whether to use torch.compile.
        backbone (str): Model name for logging.
        shot (int): Few-shot count for logging.
        prompt_set (str): Prompt set name for logging.
        weight_decay (float): Weight decay for optimizer.
    
    Returns:
        List[Dict]: Evaluation result records for logits.
    """
    recs = []
    try:
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

            if isinstance(y_arr, torch.Tensor):
                y_true = y_arr.cpu().numpy()
            else:
                y_true = np.asarray(y_arr)

            yhat = logits_np.argmax(axis=1)
            recs = make_record(
                recs,
                shot,
                "linear+prompts",
                alpha,
                prompt_set,
                lr,
                weight_decay,
                split_name,
                backbone,
                y_true,
                yhat,
                logits_np,
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
    """
    Build few-shot prototypes directly by encoding support samples.
    
    Samples support examples for each class, encodes them with the CLIP image encoder,
    and computes class mean embeddings (prototypes).
    
    Args:
        labels (List[str]): Class label names.
        train_csv (str): Path to training CSV with [filepath, label] columns.
        shots (int): Number of support samples per class.
        model: CLIP image encoder model.
        preprocess: CLIP preprocessing function.
        device (str): Device for encoding.
        cache_dir (str): Directory for image encoding cache.
        root (str): Root directory for relative image paths.
        random_state (int): Random seed for reproducible sampling.
        show_progress (bool): Whether to show progress.
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping label names to normalized prototype vectors.
    """
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
                                         num_samples: Optional[int] = None,
                                         delta_prompts_path: Optional[str] = None, pretrained: str = "openai",
                                         random_state: int = 42, cache_dir: str = DEFAULT_CACHE_DIR, root: str = ".",
                                         show_progress: bool = False):
    """
    End-to-end evaluation using direct image encoding and prompt-based classification.
    
    Encodes all test images and evaluates multiple classification strategies:
    - Zero-shot with prompt mean embeddings
    - Prompt pooling (averaged prompt embeddings)
    - Few-shot prototypes (supports variable shot numbers)
    - Linear probe baseline (trained on support examples)
    
    Args:
        labels (List[str]): Class label names.
        train_csv (str): Path to training set CSV.
        split_csv (str): Path to evaluation set CSV.
        shots (tuple): Few-shot shot counts to evaluate (default (1, 5, 10)).
        model_name (str): CLIP model name.
        device (str): Device for computation.
        num_samples (Optional[int]): Limit number of test samples (for quick evaluation).
        temp (float): Temperature for softmax in prompt pooling.
        delta_prompts_path (Optional[str]): Path to custom prompts JSON.
        pretrained (str): Pretrained weights source.
        random_state (int): Random seed.
        cache_dir (str): Cache directory for embeddings.
        root (str): Root directory for image paths.
        show_progress (bool): Whether to show progress bars.
    
    Returns:
        Dict: Results containing:
            - "n_samples": Number of test samples
            - "delta_mean": Zero-shot accuracy with mean prompt embeddings
            - "delta_pool": Accuracy with prompt pooling
            - "few_shot_results": Dict mapping shot counts to accuracies
            - "linear_probe": Linear probe baseline accuracy
    """
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

    # linear probe baseline 
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
# main
# ------------------------------
def parse_args():
    """
    Parse and return command-line arguments for the few-shot evaluation pipeline.
    
    Returns:
        argparse.Namespace: Configuration object with fields for:
            - Data paths (data_root, train/val/test CSVs, labels TSV)
            - Model options (backbones, pretrained weights)
            - Few-shot parameters (shots, alpha_grid, LR/WD grids)
            - Prompt sets and mixing strategies
            - Training options (epochs, batch size, device, CPU probes flag)
            - Output directories (save_dir, results_dir)
            - Advanced flags (direct_eval, fast, no_compile)
    """
    ap = argparse.ArgumentParser(description="Unified few-shot + zero-shot evaluation for CLIP-style models (prompt-aware few-shot extensions)")
    ap.add_argument("--data-root", default=DATA_ROOT, help="Path to dataset root")
    ap.add_argument("--train-csv", default=TRAIN_CSV, help="Training CSV file")
    ap.add_argument("--val-csv", default=VAL_CSV, help="Validation CSV file")
    ap.add_argument("--test-csv", default=TEST_CSV, help="Test CSV file")
    ap.add_argument("--labels", default=LABELS, help="TSV with id\tlabel_name")
    ap.add_argument("--prompts-path", default=DEFAULT_PROMPTS_JSON)
    ap.add_argument("--backbones", nargs="+",default=[b for b in DEFAULT_PRETRAINED.keys()], help="Model backbones to use (default uses DEFAULT_BACKBONES values).")
    #ap.add_argument("--pretrained",nargs="+", default=[DEFAULT_PRETRAINED.get(b, "openai") for b in DEFAULT_PRETRAINED.keys()], help="Pretrained weights for each backbone (default uses DEFAULT_PRETRAINED map).")
    ap.add_argument("--pretrained",nargs="+",  default=None, help="Pretrained weights for each backbone; if omitted, uses DEFAULT_PRETRAINED map.")
    ap.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    ap.add_argument("--save-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--direct-eval", action="store_true", help="Run direct prompt-based evaluation (encodes images on the fly).")
    ap.add_argument("--num-samples", type=int, default=None, help="If using direct-eval, optionally subsample test set.")
    ap.add_argument("--temp", type=float, default=0.12, help="Temp for prompt pooling softmax")
    ap.add_argument("--no-compile", action="store_false", dest="no_compile", help="Dont attempt torch.compile() for speed where possible")
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--cpu-probes", action="store_true", default=True, help="Run linear-probe trainings on CPU to avoid GPU oversubscription (default True). Set to False to force probe training on same device.")
    ap.add_argument("--no-prompts-for-prototype", action="store_true", help="Disable text-prompt fusion for prototypes")
    ap.add_argument("--no-prompts-for-linear", action="store_true", help="Disable text-prompt pseudo-examples for linear probe")
    ap.add_argument("--alpha-grid", default=ALPHAS, help="Comma-separated list of prompt-alpha values to sweep over")
    ap.add_argument("--prompt-sets", default=PROMPT_SET, help="Comma-separated prompt sets to sweep")
    ap.add_argument("--lr-grid", default=LR_GRID, help="Comma-separated learning rates for linear probe")
    ap.add_argument("--wd-grid", default=WD_GRID, help="Comma-separated weight decay values for linear probe")

    args = ap.parse_args()
    args.use_prompts_for_prototype = not args.no_prompts_for_prototype
    args.use_prompts_for_linear = not args.no_prompts_for_linear

    # Ensure lists (argparse sometimes gives string)
    if isinstance(args.alpha_grid, str):
        args.alpha_grid = [float(x) for x in args.alpha_grid.split(",")]
    if isinstance(args.lr_grid, str):
        args.lr_grid = [float(x) for x in args.lr_grid.split(",")]
    if isinstance(args.wd_grid, str):
        args.wd_grid = [float(x) for x in args.wd_grid.split(",")]
    if isinstance(args.prompt_sets, str):
        args.prompt_sets = args.prompt_sets.split(",")

    # Make sure max_workers reasonable
    args.max_workers = max(1, int(min(args.max_workers, (os.cpu_count() or 4))))

    return args

def main():
    """
    Main entry point for few-shot evaluation pipeline.
    
    Orchestrates the full evaluation workflow:
    1. Parses command-line arguments
    2. Creates output directories
    3. Loads class labels
    4. Runs evaluations for each backbone (either direct or cached feature-based)
    5. Saves results as JSON files
    
    Results are saved to:
        - few_shot_overall_results.json: All evaluation records
        - per_backbone JSON files: Direct evaluation results (if --direct-eval flag)
    """
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Build a dict backbone -> pretrained tag
    if not args.pretrained:
        pretrained_map = {b: DEFAULT_PRETRAINED.get(b, "openai") for b in args.backbones}
    else:
        if len(args.pretrained) == 1 and len(args.backbones) > 1:
            pretrained_map = {b: args.pretrained[0] for b in args.backbones}
        else:
            pretrained_map = {}
            for b, tag in zip(args.backbones, args.pretrained):
                pretrained_map[b] = tag
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
        for backbone in backbone_iter:
            recs = evaluate_backbone(backbone, args, list(label_names), K, show_progress=show_progress)
            records_all.extend(recs)

        out_json = Path(args.results_dir) / "few_shot_overall_results.json"
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(records_all, jf, indent=2)
        print(f"Wrote overall results -> {out_json}")


if __name__ == "__main__":
    main()
