#!/usr/bin/env python3
import argparse, os, csv, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from transformers import BlipProcessor, BlipModel

# helpers 
def load_labels(labels_tsv: str) -> Tuple[np.ndarray, Dict[str,int]]:
    ids, names = [], []
    with open(labels_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            i, name, *_ = line.split("\t")
            ids.append(int(i)); names.append(name)
    name2id = {n:i for i,n in zip(ids,names)}
    return np.array(names, dtype=object), name2id

def read_split(csv_path: str, image_col="filepath", label_col="label"):
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
    if not rows or image_col not in rows[0] or label_col not in rows[0]:
        raise ValueError(f"CSV must have columns '{image_col}' and '{label_col}'.")
    return rows

def balanced_acc(y_true, y_pred, K):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(K))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = np.where(cm.sum(1)>0, np.diag(cm)/cm.sum(1), 0.0)
    return np.nanmean(per_class)

def topk_membership(true_ids, class_scores, k):
    """true in top-k classes chosen by summed neighbor scores."""
    # class_scores: [N, K]
    topk = np.argpartition(-class_scores, kth=min(k, class_scores.shape[1]-1), axis=1)[:, :k]
    return (topk == true_ids[:, None]).any(axis=1).mean()

def main():
    ap = argparse.ArgumentParser(description="BLIP few-shot kNN with vision embeddings")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--image_col", default="filepath")
    ap.add_argument("--label_col", default="label")

    ap.add_argument("--blip_model", default="Salesforce/blip-itm-base-coco")
    ap.add_argument("--cache_dir", default=os.environ.get("HF_HOME",""))
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="fp16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--pooling", choices=["cls","mean"], default="cls",
                    help="How to pool vision tokens when pooler_output is not available")

    ap.add_argument("--gallery_per_class", type=int, default=3,
                    help="# of train images per class to store in gallery")
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_json", default="results/metrics/blip_knn.json")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # load labels (order defines class ids)
    label_names, name2id = load_labels(args.labels); K = len(label_names)

    # processor + model
    cache_kwargs = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    processor = BlipProcessor.from_pretrained(args.blip_model, **cache_kwargs)
    model = BlipModel.from_pretrained(args.blip_model, **cache_kwargs).to(args.device)
    model.eval()

    # dtype & autocast
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    use_autocast = args.device.startswith("cuda")
    amp_dtype = dtype_map[args.dtype]

    def load_images(paths: List[str]) -> List[Image.Image]:
        ims=[]
        for p in paths:
            try:
                ims.append(Image.open(p).convert("RGB"))
            except Exception:
                ims.append(Image.new("RGB",(224,224),(0,0,0)))
        return ims

    @torch.no_grad()
    def encode_images(paths: List[str]) -> np.ndarray:
        feats = []
        for i in range(0, len(paths), args.batch_size):
            batch = paths[i:i+args.batch_size]
            ims = load_images(batch)
            pv = processor(images=ims, return_tensors="pt").pixel_values.to(args.device)
            if use_autocast:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model.vision_model(pixel_values=pv, return_dict=True)
            else:
                out = model.vision_model(pixel_values=pv, return_dict=True)

            # Prefer pooler_output; else CLS or mean pooling
            emb = getattr(out, "pooler_output", None)
            if emb is None:
                hs = out.last_hidden_state  # [B, T, D]
                if args.pooling == "cls":
                    emb = hs[:, 0, :]
                else:
                    emb = hs[:, 1:, :].mean(dim=1)
            # If a projection head exists, use it (aligns with text/image space)
            if hasattr(model, "vision_proj") and model.vision_proj is not None:
                emb = model.vision_proj(emb)

            # L2 normalize
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
            feats.append(emb.float().cpu().numpy())
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 768), dtype=np.float32)

    # build a small gallery from train
    train_rows = read_split(args.train_csv, args.image_col, args.label_col)
    per_class = {c: [] for c in label_names}
    for r in train_rows:
        cls = r[args.label_col]
        if cls in per_class and len(per_class[cls]) < args.gallery_per_class:
            per_class[cls].append(os.path.join(args.data_root, r[args.image_col]))
    gallery_paths, gallery_y = [], []
    for cid, cname in enumerate(label_names):
        for p in per_class[cname]:
            gallery_paths.append(p); gallery_y.append(cid)
    gallery_y = np.array(gallery_y, dtype=int)

    if len(gallery_paths) == 0:
        raise RuntimeError("Empty gallery: increase --gallery_per_class or check paths.")

    print(f"[info] encoding gallery: {len(gallery_paths)} images "
          f"({args.gallery_per_class}/class, K={K})")
    G = encode_images(gallery_paths)              # [G, D]
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)

    results = {"model": args.blip_model, "type": "blip_knn",
               "gallery_per_class": args.gallery_per_class,
               "pooling": args.pooling, "splits": {}}

    # evaluate requested splits
    for split, csvp in [("val", args.val_csv), ("test", args.test_csv)]:
        if split not in args.splits: continue
        rows = read_split(csvp, args.image_col, args.label_col)

        img_paths, y = [], []
        for r in rows:
            cls = r[args.label_col]
            cid = int(np.where(label_names==cls)[0][0])
            img_paths.append(os.path.join(args.data_root, r[args.image_col])); y.append(cid)
        y = np.array(y, dtype=int)
        print(f"[info] encoding {split}: {len(img_paths)} images")
        Q = encode_images(img_paths)              # [N, D]
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)

        # cosine sim with gallery
        S = Q @ G.T                               # [N, G]
        # top-1 prediction: nearest neighbor label
        nn_idx = np.argmax(S, axis=1)
        y_hat1 = gallery_y[nn_idx]

        # build per-class score by summing top-5 neighbor sims
        top_m = 5
        nbr_idx = np.argpartition(-S, kth=min(top_m, S.shape[1]-1), axis=1)[:, :top_m]
        class_scores = np.zeros((S.shape[0], K), dtype=np.float32)
        for i in range(S.shape[0]):
            for j in nbr_idx[i]:
                class_scores[i, gallery_y[j]] += S[i, j]
        # predicted class = argmax aggregated score
        y_hat = np.argmax(class_scores, axis=1)

        top1 = (y_hat1 == y).mean()              # strict 1-NN
        top5 = topk_membership(y, class_scores, k=5)
        bal  = balanced_acc(y, y_hat, K)         # use aggregated for per-class metrics
        macro= f1_score(y, y_hat, average="macro")

        results["splits"][split] = dict(top1=float(top1), top5=float(top5),
                                        balanced_acc=float(bal), macro_f1=float(macro))

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote → {args.out_json}")

if __name__ == "__main__":
    main()
