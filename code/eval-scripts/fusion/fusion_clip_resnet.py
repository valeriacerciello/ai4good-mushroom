"""
CLIP + ResNet score-level fusion on mushroom dataset.

Pipeline
- Load val/test CSVs (or split test if val.csv unavailable)
- Get CLIP per-class probabilities with prompt pooling (T=0.05 by default)
- Get ResNet per-class probabilities from a checkpoint
- Tune fusion weight alpha on val
- Evaluate fused probabilities on test with Top-1, Top-5, Balanced Acc, Macro-F1
- Save results JSON and plots

Notes
- Uses delta_prompts.json for CLIP prompts
- Defaults to our quick ResNet18 checkpoint if no ckpt provided
- Restricts to a subset (NUM_VAL/NUM_TEST) for speed; adjust as needed
"""
import os
import json
import math
import argparse
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import clip
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt

from clip_utils import get_text_embeddings


# Defaults and paths
DATA_ROOT = "/zfs/ai4good/datasets/mushroom"
LABELS_TSV = "/zfs/ai4good/student/hgupta/ai4good-mushroom/labels.tsv"
DELTA_PROMPTS = "/zfs/ai4good/student/hgupta/delta_prompts.json"
DEFAULT_RESNET_CKPT = "/zfs/ai4good/student/hgupta/logs/mushroom/train/runs/2025-10-21_22-27-28/checkpoints/epoch_000.ckpt"

NUM_VAL = 200
NUM_TEST = 200
CLIP_TEMP = 0.05  # best from sweep
ALPHAS = [round(a, 2) for a in np.linspace(0.0, 1.0, 11)]


def fix_path(p: str) -> str:
    """Map Kaggle-style path to local dataset root."""
    if "/kaggle/working/" in p:
        return os.path.join(DATA_ROOT, p.split("/kaggle/working/")[-1])
    return p


def load_split_df(split: str, limit: int = None) -> pd.DataFrame:
    csv_path = os.path.join(DATA_ROOT, f"{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if limit:
        df = df.head(limit)
    df["image_path"] = df.iloc[:, 0].apply(fix_path)
    df["label"] = df.iloc[:, 1]
    return df[["image_path", "label"]]


def load_labels(df_list: List[pd.DataFrame]) -> List[str]:
    """Build labels list.
    Priority:
    1) Union of labels from train/val/test CSVs under DATA_ROOT (full 169 classes)
    2) Repo labels.tsv if it covers all observed labels
    3) Union from provided dfs
    """
    # Try full dataset union
    try:
        dfs = []
        for split in ["train", "val", "test"]:
            csv_path = os.path.join(DATA_ROOT, f"{split}.csv")
            if os.path.exists(csv_path):
                d = pd.read_csv(csv_path)
                d = d.rename(columns={d.columns[1]: "label"})
                dfs.append(d[[d.columns[0], "label"]])
        if dfs:
            all_labels = sorted(list(set(pd.concat(dfs)["label"].unique().tolist())))
            if len(all_labels) >= 160:  # sanity threshold for near-full coverage
                return all_labels
    except Exception:
        pass

    # Fallback: repo labels if compatible
    union_labels = sorted(list(set(pd.concat(df_list)["label"].unique().tolist())))
    if os.path.exists(LABELS_TSV):
        try:
            repo_labels = [ln.strip() for ln in open(LABELS_TSV, "r").read().splitlines() if ln.strip()]
            if repo_labels and all(l in repo_labels for l in union_labels):
                return repo_labels
        except Exception:
            pass

    return union_labels


def build_clip_model(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def compute_clip_probs(df: pd.DataFrame, labels: List[str], prompts_path: str, temp: float,
                       batch_size: int = 16, device: str = None) -> np.ndarray:
    model, preprocess, device0 = build_clip_model(device)
    device = device or device0

    # Get per-prompt embeddings for each class
    emb_bundle = get_text_embeddings(
        labels,
        prompt_set='ensemble',
        model_name='ViT-B/32',
        device=device,
        include_common_names=True,
        common_prompts_path=prompts_path,
    )

    def score_image(image: Image.Image) -> torch.Tensor:
        img_t = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = model.encode_image(img_t)
            img_emb = F.normalize(img_emb, dim=-1).squeeze(0)
        # Compute pooled score per class
        scores = []
        for label in labels:
            pe = emb_bundle[label]['prompt_embeddings']
            if pe.numel() == 0:
                scores.append(torch.tensor(-1e9, device=device))
                continue
            pe = pe.to(img_emb.dtype).to(device)
            sims = torch.cosine_similarity(img_emb.unsqueeze(0), pe, dim=1)
            w = torch.softmax(sims / temp, dim=0)
            pooled = (w * sims).sum()
            scores.append(pooled)
        scores = torch.stack(scores, dim=0)
        # Convert to probabilities across classes
        probs = torch.softmax(scores, dim=0)  # class-dim softmax
        return probs.detach().cpu()

    probs_list = []
    for idx in range(len(df)):
        path = df.iloc[idx]["image_path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # fallback to zeros (uniform) if missing
            probs_list.append(torch.full((len(labels),), 1.0 / len(labels)))
            continue
        probs = score_image(img)
        probs_list.append(probs)

    probs_mat = torch.stack(probs_list, dim=0).numpy()
    return probs_mat


def load_resnet_from_ckpt(ckpt_path: str, model_name: str = "resnet18", num_classes: int = 169) -> torch.nn.Module:
    # Instantiate torchvision model
    if model_name == "resnet18":
        net = torchvision.models.resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        net = torchvision.models.resnet34(num_classes=num_classes)
    elif model_name == "resnet50":
        net = torchvision.models.resnet50(num_classes=num_classes)
    elif model_name == "resnet101":
        net = torchvision.models.resnet101(num_classes=num_classes)
    elif model_name == "resnet152":
        net = torchvision.models.resnet152(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Load checkpoint state_dict; adapt keys from Lightning module if needed
    # Allow full pickle load for our trusted checkpoint
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = sd.get('state_dict', sd)
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('net._orig_mod.'):
            new_k = k[len('net._orig_mod.'):]
        elif k.startswith('net.'):
            new_k = k[len('net.'):]
        else:
            # skip other module metrics like losses, metrics buffers
            continue
        cleaned[new_k] = v
    # If checkpoint was trained with a different number of classes, the
    # final fc layer shape will mismatch. Remove fc weights/bias from the
    # checkpoint in that case so the model keeps its current (correct) fc.
    try:
        ck_fc_w = cleaned.get('fc.weight')
        if ck_fc_w is not None and ck_fc_w.shape != net.fc.weight.shape:
            cleaned.pop('fc.weight', None)
            cleaned.pop('fc.bias', None)
    except Exception:
        # be conservative: if anything unexpected, don't crash—proceed with load
        pass

    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    # It's okay if there are missing batchnorm running stats etc.
    return net


def get_resnet_val_transforms(image_size: int = 384) -> transforms.Compose:
    # Mirror external/ecovision_mushroom/configs/data/mushroom.yaml val_transform
    return transforms.Compose([
        transforms.Resize(440),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4308, 0.4084, 0.3393], std=[0.1063, 0.1023, 0.1123]),
    ])


def compute_resnet_probs(df: pd.DataFrame, labels: List[str], ckpt_path: str, model_name: str = "resnet18",
                         batch_size: int = 32, device: str = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = load_resnet_from_ckpt(ckpt_path, model_name=model_name, num_classes=len(labels))
    net.to(device)
    net.eval()
    tfm = get_resnet_val_transforms()

    probs = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch_paths = df["image_path"].iloc[i:i+batch_size].tolist()
            batch_imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(tfm(img))
                except Exception:
                    # if missing, use zeros to avoid crash
                    batch_imgs.append(torch.zeros((3, 384, 384)))
            batch_t = torch.stack(batch_imgs, dim=0).to(device)
            logits = net(batch_t)
            prob = torch.softmax(logits, dim=1).cpu()
            probs.append(prob)
    probs_mat = torch.cat(probs, dim=0).numpy()
    return probs_mat


def evaluate_metrics(y_true: List[str], y_probs: np.ndarray, labels: List[str]) -> Dict[str, float]:
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[y] for y in y_true])
    top1_idx = y_probs.argmax(axis=1)
    top1_acc = float((top1_idx == y_idx).mean())

    # top-5
    top5_idx = np.argsort(-y_probs, axis=1)[:, :5]
    top5_acc = float(np.mean([y_idx[i] in top5_idx[i] for i in range(len(y_idx))]))

    # balanced acc, macro f1
    y_pred = top1_idx
    bal_acc = balanced_accuracy_score(y_idx, y_pred)
    macro_f1 = f1_score(y_idx, y_pred, average='macro', zero_division=0)

    return {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'balanced_acc': float(bal_acc),
        'macro_f1': float(macro_f1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=DEFAULT_RESNET_CKPT, help='ResNet checkpoint path')
    parser.add_argument('--model', type=str, default='resnet18', help='ResNet model name')
    parser.add_argument('--prompts', type=str, default=DELTA_PROMPTS, help='Prompt JSON for CLIP')
    parser.add_argument('--clip_temp', type=float, default=CLIP_TEMP)
    parser.add_argument('--val_n', type=int, default=NUM_VAL)
    parser.add_argument('--test_n', type=int, default=NUM_TEST)
    parser.add_argument('--restrict-labels-file', type=str, default=None,
                        help='Optional path to a file containing one label per line to restrict evaluation')
    parser.add_argument('--out', type=str, default='fusion_results.json')
    args = parser.parse_args()

    # Load splits
    try:
        df_val = load_split_df('val', limit=args.val_n)
    except FileNotFoundError:
        # Fallback: split test into val/test halves
        df_test_all = load_split_df('test')
        n = min(args.val_n + args.test_n, len(df_test_all))
        df_test_all = df_test_all.head(n)
        df_val = df_test_all.iloc[: args.val_n].reset_index(drop=True)
        df_test = df_test_all.iloc[args.val_n :].reset_index(drop=True)
    else:
        df_test = load_split_df('test', limit=args.test_n)

    labels = load_labels([df_val, df_test])

    # Optional restriction by user-provided label list (one per line)
    if args.restrict_labels_file:
        try:
            with open(args.restrict_labels_file, 'r') as f:
                restricted = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
            # keep only labels that are in the original labels list
            labels = [l for l in labels if l in restricted]
            print(f'Labels restricted to {len(labels)} entries from {args.restrict_labels_file}')
        except Exception as e:
            print(f'Could not read restrict labels file: {e}. Proceeding without restriction.')

    print(f"Loaded val={len(df_val)}, test={len(df_test)}, classes={len(labels)}")

    # Compute CLIP probabilities
    print("Computing CLIP probabilities (val)...")
    clip_val = compute_clip_probs(df_val, labels, args.prompts, temp=args.clip_temp)
    print("Computing CLIP probabilities (test)...")
    clip_test = compute_clip_probs(df_test, labels, args.prompts, temp=args.clip_temp)

    # Compute ResNet probabilities
    if not os.path.exists(args.ckpt):
        print(f"WARNING: checkpoint not found at {args.ckpt}. ResNet probs will be uniform.")
        res_val = np.full_like(clip_val, 1.0 / len(labels))
        res_test = np.full_like(clip_test, 1.0 / len(labels))
    else:
        print("Computing ResNet probabilities (val/test)...")
        res_val = compute_resnet_probs(df_val, labels, args.ckpt, model_name=args.model)
        res_test = compute_resnet_probs(df_test, labels, args.ckpt, model_name=args.model)

    y_val = df_val['label'].tolist()
    y_test = df_test['label'].tolist()

    # Tune alpha on val
    results = {'val': {}, 'test': {}, 'best_alpha': None}
    best_alpha = None
    best_top1 = -1.0
    best_f1 = -1.0
    for a in ALPHAS:
        fused_val = a * res_val + (1 - a) * clip_val
        m = evaluate_metrics(y_val, fused_val, labels)
        results['val'][str(a)] = m
        # choose by top1 accuracy primarily, tie-breaker by macro_f1
        if (m['top1_acc'] > best_top1) or (math.isclose(m['top1_acc'], best_top1) and m['macro_f1'] > best_f1):
            best_top1 = m['top1_acc']
            best_f1 = m['macro_f1']
            best_alpha = a

    # Evaluate on test with best alpha
    fused_test = best_alpha * res_test + (1 - best_alpha) * clip_test
    test_metrics = evaluate_metrics(y_test, fused_test, labels)
    results['test'] = {'alpha': best_alpha, **test_metrics}
    results['labels'] = labels
    results['val_size'] = len(df_val)
    results['test_size'] = len(df_test)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== Fusion Summary ===")
    print(f"Best alpha (val): {best_alpha}")
    print(f"Test metrics: top1={test_metrics['top1_acc']:.3f}, top5={test_metrics['top5_acc']:.3f}, "
          f"bal_acc={test_metrics['balanced_acc']:.3f}, macro_f1={test_metrics['macro_f1']:.3f}")

    # Plot alpha vs metrics on val
    try:
        al = [float(k) for k in results['val'].keys()]
        top1 = [results['val'][k]['top1_acc'] for k in results['val'].keys()]
        top5 = [results['val'][k]['top5_acc'] for k in results['val'].keys()]
        bal = [results['val'][k]['balanced_acc'] for k in results['val'].keys()]
        f1 = [results['val'][k]['macro_f1'] for k in results['val'].keys()]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(al, top1, 'o-', label='Top-1')
        ax.plot(al, top5, 's-', label='Top-5')
        ax.plot(al, bal, '^-', label='Balanced Acc')
        ax.plot(al, f1, 'd-', label='Macro-F1')
        ax.axvline(best_alpha, color='red', linestyle='--', alpha=0.5, label=f'Best alpha={best_alpha}')
        ax.set_xlabel('alpha (weight on ResNet)')
        ax.set_ylabel('Score')
        ax.set_title('Fusion weight sweep on val')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig('fusion_alpha_sweep.png', dpi=300, bbox_inches='tight')
        print('Saved fusion_alpha_sweep.png')
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == '__main__':
    main()
