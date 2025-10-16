#!/usr/bin/env python3
"""Few-shot probing on saved CLIP features.

Creates a reproducible sampler that picks n images per class, then evaluates:
 - Prototype classifier (class centroid)
 - Linear classifier (single linear layer trained with cross-entropy)

Feature files are expected at: features/<backbone>/<split>.npz (X, y, paths)
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score


def load_labels(labels_tsv):
    ids, names = [], []
    with open(labels_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            i, name, *_ = line.split("\t")
            ids.append(int(i)); names.append(name)
    name2id = {n:i for i,n in zip(ids,names)}
    return np.array(names, dtype=object), name2id

def ensure_features(split, backbone, data_root, csv_path, labels_tsv, save_dir="features", pretrained="openai", device=None):
    npz = Path(save_dir)/backbone/f"{split}.npz"
    if npz.exists():
        z=np.load(npz, allow_pickle=True); return z["X"], z["y"], z["paths"]
    # lazy compute via dump_features (copied behaviour from other scripts)
    cmd = f'python scripts/dump_features.py --data-root "{data_root}" --csv "{csv_path}" --labels "{labels_tsv}" --backbone "{backbone}" --pretrained "{pretrained}" --save-dir "{save_dir}"'
    ec = os.system(cmd)
    assert ec==0, "Feature dump failed"
    z=np.load(npz, allow_pickle=True); return z["X"], z["y"], z["paths"]

def topk_acc(y_true, scores, k):
    k = min(k, scores.shape[1])
    topk = np.argsort(-scores, axis=1)[:, :k]
    return (topk == y_true[:,None]).any(axis=1).mean()

def balanced_acc(y_true, y_pred, K):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(K))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = np.where(cm.sum(1)>0, np.diag(cm)/cm.sum(1), 0.0)
    return np.nanmean(per_class)


def sample_few_shot_indices(y_train, K, n_shot, rng):
    """Return arrays of support indices and labels for n_shot per class.

    If a class has fewer than n_shot examples, sampling is done with replacement.
    The order of returned indices corresponds to concatenated classes (but labels are provided).
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
    support_indices = np.concatenate(support_indices, axis=0)
    support_labels = np.concatenate(support_labels, axis=0)
    return support_indices, support_labels


def prototype_classifier(X_support, y_support, K):
    D = X_support.shape[1]
    prototypes = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        inds = np.where(y_support == k)[0]
        p = X_support[inds].mean(axis=0)
        # normalize
        n = np.linalg.norm(p)
        if n > 0:
            p = p / n
        prototypes[k] = p
    return prototypes

class LinearProbe(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.linear = nn.Linear(dim, K)

    def forward(self, x):
        return self.linear(x)

def train_linear_probe(X_support, y_support, X_val=None, y_val=None, epochs=200, lr=1e-2, batch_size=32, device="cpu", seed=42):
    torch.manual_seed(seed)
    X = torch.from_numpy(X_support).float()
    y = torch.from_numpy(y_support).long()
    D = X.shape[1]
    K = int(y.max())+1

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = LinearProbe(D, K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_loss = float('inf')
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        ep_loss = ep_loss / max(1, n)

        # optional val monitoring
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(X_val).float().to(device)
                yv = torch.from_numpy(y_val).long().to(device)
                logits = model(xv)
                val_loss = loss_fn(logits, yv).item()
        else:
            val_loss = ep_loss

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_predictions(y_true, y_pred, K):
    top1 = (y_pred == y_true).mean()
    bal = balanced_acc(y_true, y_pred, K)
    macro = f1_score(y_true, y_pred, average='macro')
    return top1, bal, macro

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1,5,10,20])
    ap.add_argument("--save-dir", default="features")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    label_names, name2id = load_labels(args.labels)
    K = len(label_names)

    # load features
    X_tr, y_tr, _ = ensure_features("train", args.backbone, args.data_root, args.train_csv, args.labels, save_dir=args.save_dir, pretrained=args.pretrained)
    X_te, y_te, _ = ensure_features("test", args.backbone, args.data_root, args.test_csv, args.labels, save_dir=args.save_dir, pretrained=args.pretrained)

    results = []
    rng = np.random.default_rng(args.seed)

    for shot in args.shots:
        # for reproducibility base seed on (global seed, shot)
        shot_rng = np.random.default_rng(args.seed + int(shot))
        sup_idx, sup_labels = sample_few_shot_indices(y_tr, K, shot, shot_rng)
        X_sup = X_tr[sup_idx]
        y_sup = sup_labels

        # Prototype
        prototypes = prototype_classifier(X_sup, y_sup, K)  # [K, D]
        # scores: X_te [N, D] dot prototypes.T [D, K]
        scores_proto = X_te @ prototypes.T
        yhat_proto = np.argmax(scores_proto, axis=1)
        proto_top1, proto_bal, proto_macro = evaluate_predictions(y_te, yhat_proto, K)

        # Linear probe
        device = args.device
        model = train_linear_probe(X_sup, y_sup, X_val=None, y_val=None, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=device, seed=args.seed + int(shot))
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X_te).float().to(device)
            logits = model(xt).cpu().numpy()
            yhat_lin = np.argmax(logits, axis=1)
        lin_top1, lin_bal, lin_macro = evaluate_predictions(y_te, yhat_lin, K)

        print(f"shot={shot}: prototype top1={proto_top1:.4f} bal={proto_bal:.4f} macro={proto_macro:.4f}")
        print(f"shot={shot}: linear    top1={lin_top1:.4f} bal={lin_bal:.4f} macro={lin_macro:.4f}")

        results.append((shot, "prototype", proto_top1, proto_bal, proto_macro))
        results.append((shot, "linear", lin_top1, lin_bal, lin_macro))

    out_csv = os.path.join(args.results_dir, f"few_shot_metrics_{args.backbone.replace(' ','_')}.csv")
    import csv
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shot","model","top1","balanced_acc","macro_f1"])
        for row in results:
            w.writerow(row)
    print(f"Wrote results → {out_csv}")


if __name__ == "__main__":
    main()
