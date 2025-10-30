#!/usr/bin/env python3
import argparse, os, json, csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from transformers import BlipProcessor, BlipForImageTextRetrieval
import open_clip  # for shortlist via CLIP

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

def read_split_csv(csv_path: str, image_col="filepath", label_col="label") -> List[Dict[str,str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
    if not rows or image_col not in rows[0] or label_col not in rows[0]:
        raise ValueError(f"CSV must have columns '{image_col}' and '{label_col}'.")
    return rows

def topk_acc(y_true, scores, k):
    k = min(k, scores.shape[1])
    topk = np.argsort(-scores, axis=1)[:, :k]
    return (topk == y_true[:,None]).any(axis=1).mean()

def balanced_acc(y_true, y_pred, K):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(K))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = np.where(cm.sum(1)>0, np.diag(cm)/cm.sum(1), 0.0)
    return np.nanmean(per_class)

# CLIP shortlist helpers (uses existing dump_features.py)
def ensure_clip_features(split, backbone, data_root, csv_path, labels_tsv, save_dir="features", pretrained="openai"):
    npz = Path(save_dir)/backbone/f"{split}.npz"
    if npz.exists():
        z=np.load(npz, allow_pickle=True); return z["X"], z["y"], z["paths"]
    cmd = f'python scripts/dump_features.py --data-root "{data_root}" --csv "{csv_path}" --labels "{labels_tsv}" --backbone "{backbone}" --pretrained "{pretrained}" --save-dir "{save_dir}"'
    ec = os.system(cmd)
    assert ec==0, "Feature dump failed"
    z=np.load(npz, allow_pickle=True); return z["X"], z["y"], z["paths"]

def load_prompts_json(path):
    if not path or not os.path.isfile(path): return {}
    obj = json.load(open(path, "r", encoding="utf-8"))
    fixed = {}
    for k, v in obj.items():
        if isinstance(v, list):
            fixed[k] = [str(s) for s in v if isinstance(s,str) and s.strip()]
    return fixed

def build_clip_text_matrix(label_names, prompts_dict, backbone, pretrained, device, fallback="a photo of {}"):
    model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, device=device)
    model.eval()
    tok = open_clip.get_tokenizer(backbone)
    feats = []
    with torch.no_grad():
        for cls in label_names:
            prompts = prompts_dict.get(cls) or [fallback.format(cls)]
            T = tok(prompts).to(device)
            txt = model.encode_text(T)
            txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-12)
            mean = txt.mean(dim=0, keepdim=True)
            mean = mean / (mean.norm(dim=-1, keepdim=True) + 1e-12)
            feats.append(mean.cpu())
    return torch.cat(feats, dim=0).T.contiguous().float().cpu().numpy()  # [D,K]


def main():
    ap = argparse.ArgumentParser(description="BLIP ITM zero-shot baseline (with optional CLIP shortlist)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv",   required=True)
    ap.add_argument("--test-csv",  required=True)
    ap.add_argument("--labels",    required=True)

    ap.add_argument("--prompts", default="data/prompts/class_prompts_enriched.json",
                    help="Per-class prompt list JSON (use enriched by default)")
    ap.add_argument("--max_prompts_per_class", type=int, default=4)
    ap.add_argument("--aggregator", choices=["mean","max"], default="mean")

    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--image_col", default="filepath")
    ap.add_argument("--label_col", default="label")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit_images", type=int, default=0)
    ap.add_argument("--limit_classes", type=int, default=0)

    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_json", default="results/metrics/blip_itm_enriched.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default="Salesforce/blip-itm-base-coco")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="fp16")

    # shortlist config
    ap.add_argument("--shortlist_engine", choices=["none","clip"], default="clip",
                    help="Use CLIP to shortlist classes per image before BLIP-ITM rerank")
    ap.add_argument("--shortlist_topk", type=int, default=20)
    ap.add_argument("--clip_backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--prompts_for_shortlist", default="", help="JSON prompts for shortlist (defaults to --prompts)")

    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # labels & optional class limiting
    label_names, name2id = load_labels(args.labels)
    if args.limit_classes and args.limit_classes < len(label_names):
        label_names = label_names[:args.limit_classes]
    K = len(label_names)

    # prompts per class
    prompts_db = load_prompts_json(args.prompts)
    per_class_prompts: Dict[str, List[str]] = {}
    for cls in label_names:
        lst = prompts_db.get(cls) or [f"a photo of {cls}"]
        per_class_prompts[cls] = lst[:args.max_prompts_per_class]

    # model + processor (prefer safetensors)
    processor = BlipProcessor.from_pretrained(args.model)
    print(f"[info] torch: {torch.__version__}  device: {args.device}  dtype:{args.dtype}")
    try:
        model = BlipForImageTextRetrieval.from_pretrained(args.model, use_safetensors=True).to(args.device)
    except Exception as e:
        print("[warn] safetensors load failed; trying .bin:", e)
        model = BlipForImageTextRetrieval.from_pretrained(args.model, use_safetensors=False).to(args.device)
    model.eval()

    # dtype + autocast
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    use_autocast = (args.device.startswith("cuda"))
    amp_dtype = dtype_map[args.dtype]

    # pre-tokenize prompts once per class (CPU tensors)
    tokenized_texts = {}
    for cls in label_names:
        tok = processor(text=per_class_prompts[cls], padding=True, truncation=True, return_tensors="pt")
        tokenized_texts[cls] = tok  # dict: input_ids, attention_mask

    # CLIP shortlist per image
    shortlist_per_image = None
    if args.shortlist_engine == "clip":
        print("[info] building CLIP shortlist …")
        # choose prompts for shortlist (default = same enriched prompts)
        shortlist_prompts = load_prompts_json(args.prompts_for_shortlist) or prompts_db
        # build CLIP text matrix [D,K]
        T = build_clip_text_matrix(
            label_names, shortlist_prompts,
            backbone=args.clip_backbone, pretrained=args.clip_pretrained,
            device=args.device, fallback="a photo of {}"
        )
        # CLIP image features for splits (uses existing npz or dumps)
        def clip_feats(split, csvp):
            X, y, paths = ensure_clip_features(split, args.clip_backbone, args.data_root, csvp, args.labels,
                                               pretrained=args.clip_pretrained)
            # normalize defensively
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            return X, y, paths
    else:
        print("[info] shortlist disabled (none)")

    # scoring helpers
    def load_batch_images(paths: List[str]) -> List[Image.Image]:
        ims = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224,224), (0,0,0))
            ims.append(img)
        return ims

    def score_batch_vs_class(batch_images: List[Image.Image], cls: str) -> np.ndarray:
        """Aggregate ITM match logit over prompts for a single class -> [B]"""
        B = len(batch_images)
        tok = tokenized_texts[cls]
        P = tok.input_ids.size(0)

        pv = processor(images=batch_images, return_tensors="pt").pixel_values.to(args.device)
        pv_rep = pv.repeat_interleave(P, dim=0)  # image-major [B*P, ...]
        input_ids = tok.input_ids.repeat(B, 1).to(args.device)      # [B*P, L]
        attn_mask = tok.attention_mask.repeat(B, 1).to(args.device) # [B*P, L]

        with torch.no_grad():
            if use_autocast:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model(pixel_values=pv_rep, input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
            else:
                out = model(pixel_values=pv_rep, input_ids=input_ids, attention_mask=attn_mask, return_dict=True)

        logits = None
        for attr in ["itm_score", "logits", "logits_itm"]:
            if hasattr(out, attr):
                logits = getattr(out, attr); break
        if logits is None:
            raise RuntimeError("BLIP ITM output missing itm_score/logits/logits_itm")

        logits = logits.view(B, P, -1)   # [B,P,2]
        match = logits[..., 1]           # [B,P]
        agg = match.mean(dim=1) if args.aggregator=="mean" else match.max(dim=1).values
        return agg.float().cpu().numpy() # [B]

    results = {"model": args.model, "type": "blip_itm", "aggregator": args.aggregator,
               "shortlist": {"engine": args.shortlist_engine, "topk": args.shortlist_topk}, "splits": {}}

    # evaluate split-by-split
    for split, csvp in [("val", args.val_csv), ("test", args.test_csv)]:
        if split not in args.splits:
            continue

        rows = read_split_csv(csvp, image_col=args.image_col, label_col=args.label_col)
        if args.limit_images and args.limit_images < len(rows):
            rows = rows[:args.limit_images]

        img_paths, y = [], []
        for r in rows:
            cls = r[args.label_col]
            if cls not in set(label_names):  # respect --limit_classes
                continue
            y.append(np.where(label_names==cls)[0][0])
            img_paths.append(os.path.join(args.data_root, r[args.image_col]))
        y = np.array(y, dtype=int)
        N = len(img_paths)
        if N == 0:
            print(f"[warn] No images for split {split}")
            continue

        # build per-image shortlist via CLIP 
        perimg_short = None
        if args.shortlist_engine == "clip":
            X, _, _ = clip_feats(split, csvp)
            if args.limit_images: X = X[:N]
            scores_clip = X @ T   # [N,K]
            idx = np.argsort(-scores_clip, axis=1)[:, :min(args.shortlist_topk, K)]
            perimg_short = [set(row.tolist()) for row in idx]

        scores = np.full((N, K), -1e9, dtype=np.float32)  # fill with low score (so non-scored classes never win)

        outer = tqdm(range(0, N, args.batch_size), desc=f"BLIP-ITM {split}", unit="img")
        for i in outer:
            j = min(i + args.batch_size, N)
            batch_files = img_paths[i:j]
            batch_images = load_batch_images(batch_files)

            # choose which classes to score
            if perimg_short is None:
                class_indices = range(K)
            else:
                # union of per-image shortlists in the batch
                u = set()
                for t in range(i, j):
                    u |= perimg_short[t]
                class_indices = sorted(u)

            # inner progress: classes being scored for this batch
            for k in tqdm(class_indices, leave=False, desc="classes"):
                cls = label_names[k]
                s = score_batch_vs_class(batch_images, cls)
                scores[i:j, k] = s

        # metrics
        y_hat = scores.argmax(axis=1)
        top1 = (y_hat == y).mean()
        top5 = topk_acc(y, scores, 5)
        bal  = balanced_acc(y, y_hat, K)
        macro= f1_score(y, y_hat, average="macro")

        results["splits"][split] = dict(top1=float(top1), top5=float(top5),
                                        balanced_acc=float(bal), macro_f1=float(macro))

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote → {args.out_json}")

if __name__ == "__main__":
    main()
