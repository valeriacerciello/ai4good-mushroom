#!/usr/bin/env python3
import argparse, os, csv, numpy as np, torch, open_clip
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

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
    # lazy compute via dump_features
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--backbone", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args=ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)

    label_names, name2id = load_labels(args.labels)
    K = len(label_names)

    # text embeddings
    model, _, _ = open_clip.create_model_and_transforms(args.backbone, pretrained=args.pretrained, device=args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.backbone)
    prompts = [f"a photo of {n}" for n in label_names]
    with torch.no_grad():
        T = tokenizer(prompts).to(args.device)
        text = model.encode_text(T)
        text = text / text.norm(dim=-1, keepdim=True)
        text = text.float().cpu().numpy().T  # [D, K]

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # majority from train
    _, y_tr, _ = ensure_features("train", args.backbone, args.data_root, args.train_csv, args.labels, pretrained=args.pretrained)
    counts = np.bincount(y_tr, minlength=K)
    maj_cls = int(np.argmax(counts))
    top5_freq = np.argsort(-counts)[:min(5, K)]

    rows=[]
    for split, csvp in [("val", args.val_csv), ("test", args.test_csv)]:
        if split not in args.splits: continue

        X, y, _ = ensure_features(split, args.backbone, args.data_root, csvp, args.labels, pretrained=args.pretrained)

        # zero-shot
        scores = X @ text         # cosine since both normalized
        zs_top1 = topk_acc(y, scores, 1)
        zs_top5 = topk_acc(y, scores, 5)
        y_hat   = np.argmax(scores, axis=1)
        zs_bal  = balanced_acc(y, y_hat, K)
        zs_macro_f1 = f1_score(y, y_hat, average="macro")


        # random (expected)
        rng = np.random.default_rng(42)
        y_rand   = rng.integers(0, K, size=len(y))
        rand_top1 = (y_rand == y).mean()
        rand_top5 = min(5, K) / K
        rand_bal  = balanced_acc(y, y_rand, K)
        rand_macro= f1_score(y, y_rand, average="macro")


        # majority
        y_maj   = np.full_like(y, maj_cls)
        maj_top1= (y_maj == y).mean()
        maj_top5= np.isin(y, top5_freq).mean()
        maj_bal = balanced_acc(y, y_maj, K)
        maj_macro = f1_score(y, y_maj, average="macro")


        # plot confusion (normalized)
        cm = confusion_matrix(y, y_hat, labels=np.arange(K), normalize="true")
        try:
            plt.figure(figsize=(6,5))
            plt.imshow(cm, interpolation="nearest")
            plt.title(f"Zero-shot Confusion ({split}, {args.backbone})")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout()
            out_png = os.path.join(args.results_dir, f"confmat_{split}_{args.backbone.replace(' ','_')}.png")
            plt.savefig(out_png, dpi=180); plt.close()
        except Exception as e:
            print("Confusion plot skipped:", e)

        rows += [
            (split,"random",   rand_top1, rand_top5, rand_bal, rand_macro),
            (split,"majority", maj_top1,  maj_top5,  maj_bal,  maj_macro),
            (split,"zero-shot",zs_top1,   zs_top5,   zs_bal,   zs_macro_f1),
        ]


    out_csv = os.path.join(args.results_dir, f"metrics_{args.backbone.replace(' ','_')}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w=csv.writer(f); w.writerow(["split","model","top1","top5","balanced_acc","macro_f1"]); w.writerows(rows)
    print(f"Wrote metrics → {out_csv}")

if __name__ == "__main__":
    main()
