#!/usr/bin/env python3
import argparse, os, csv, json, numpy as np, torch, open_clip
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

# helpers 
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

# helpers for prompt JSONs
def load_prompts_json(path):
    """Returns {class_name: [prompt1, prompt2, ...]} if file exists; else {}."""
    if not path:
        return {}
    if not os.path.isfile(path):
        print(f"[warn] prompts file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # enforce list[str]
    fixed = {}
    for k, v in obj.items():
        if isinstance(v, list):
            fixed[k] = [str(s) for s in v if isinstance(s, str) and s.strip()]
    return fixed

def build_text_matrix_from_prompts(label_names, prompts_dict, tokenizer, model, device, fallback_template="a photo of {}"):
    """
    For each class name in label_names (order matters), build a single text embedding
    by mean-pooling the normalized embeddings of all prompts for that class.
    Returns a numpy array [D, K].
    """
    model.eval()
    feats = []
    with torch.no_grad():
        for cls in label_names:
            prompts = prompts_dict.get(cls, None)
            if not prompts:
                prompts = [fallback_template.format(cls)]
            toks = tokenizer(prompts).to(device)
            txt = model.encode_text(toks)                  # [P, D]
            txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-12)
            mean = txt.mean(dim=0, keepdim=True)           # [1, D]
            mean = mean / (mean.norm(dim=-1, keepdim=True) + 1e-12)
            feats.append(mean.cpu())
    T = torch.cat(feats, dim=0).T.contiguous().float().cpu().numpy()  # [D, K]
    return T

def eval_with_text(X, y, text, K, split, backbone, results_dir, mode_tag):
    """Compute metrics using precomputed image features X and a text matrix [D,K]."""
    # X @ text is cosine if both are normalized; renormalize X defensively
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    scores = Xn @ text                           # [N, K]
    y_hat   = np.argmax(scores, axis=1)

    top1 = topk_acc(y, scores, 1)
    top5 = topk_acc(y, scores, 5)
    bal  = balanced_acc(y, y_hat, K)
    macro= f1_score(y, y_hat, average="macro")

    # plot confusion (normalized)
    cm = confusion_matrix(y, y_hat, labels=np.arange(K), normalize="true")
    try:
        plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Zero-shot Confusion ({split}, {backbone}, {mode_tag})")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        out_png = os.path.join(results_dir, f"confmat_{split}_{backbone.replace(' ','_')}_{mode_tag}.png")
        plt.savefig(out_png, dpi=180); plt.close()
    except Exception as e:
        print("Confusion plot skipped:", e)

    return dict(top1=top1, top5=top5, balanced_acc=bal, macro_f1=macro)

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
    # prompt JSONs
    ap.add_argument("--prompts_plain", default="data/prompts/class_prompts_plain.json")
    ap.add_argument("--prompts_enriched", default="data/prompts/class_prompts_enriched.json")
    # where to drop plain/enriched JSON metrics
    ap.add_argument("--out_plain_json", default="results/metrics/clip_plain.json")
    ap.add_argument("--out_enriched_json", default="results/metrics/clip_enriched.json")
    args=ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.out_plain_json)).mkdir(parents=True, exist_ok=True)

    # labels + model/tokenizer
    label_names, name2id = load_labels(args.labels)   # order defines columns in text matrix
    K = len(label_names)
    model, _, _ = open_clip.create_model_and_transforms(args.backbone, pretrained=args.pretrained, device=args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.backbone)

    # build text matrices 
    # Fallback template keeps backward-compat with your original script
    plain_prompts = load_prompts_json(args.prompts_plain)
    enr_prompts   = load_prompts_json(args.prompts_enriched)

    T_plain = build_text_matrix_from_prompts(label_names, plain_prompts, tokenizer, model, args.device,
                                             fallback_template="a photo of {}")
    # Enriched may be empty for some classes -> fallback to plain template for those
    T_enr   = build_text_matrix_from_prompts(label_names, enr_prompts, tokenizer, model, args.device,
                                             fallback_template="a close-up photo of {} mushroom")

    # majority from train 
    _, y_tr, _ = ensure_features("train", args.backbone, args.data_root, args.train_csv, args.labels, pretrained=args.pretrained)
    counts = np.bincount(y_tr, minlength=K)
    maj_cls = int(np.argmax(counts))
    top5_freq = np.argsort(-counts)[:min(5, K)]

    rows=[]
    json_plain = {"model": f"{args.backbone}/{args.pretrained}", "type": "plain", "splits": {}}
    json_enr   = {"model": f"{args.backbone}/{args.pretrained}", "type": "enriched", "splits": {}}

    for split, csvp in [("val", args.val_csv), ("test", args.test_csv)]:
        if split not in args.splits: 
            continue

        X, y, _ = ensure_features(split, args.backbone, args.data_root, csvp, args.labels, pretrained=args.pretrained)

        # baselines (random/majority)
        rng = np.random.default_rng(42)
        y_rand   = rng.integers(0, K, size=len(y))
        rand_top1 = (y_rand == y).mean()
        rand_top5 = min(5, K) / K
        rand_bal  = balanced_acc(y, y_rand, K)
        rand_macro= f1_score(y, y_rand, average="macro")

        y_maj   = np.full_like(y, maj_cls)
        maj_top1= (y_maj == y).mean()
        maj_top5= np.isin(y, top5_freq).mean()
        maj_bal = balanced_acc(y, y_maj, K)
        maj_macro = f1_score(y, y_maj, average="macro")

        rows += [
            (split,"random",   rand_top1, rand_top5, rand_bal, rand_macro),
            (split,"majority", maj_top1,  maj_top5,  maj_bal,  maj_macro),
        ]

        # zero-shot (plain)
        m_plain = eval_with_text(X, y, T_plain, K, split, args.backbone, args.results_dir, mode_tag="plain")
        rows += [(split,"zero-shot-plain", m_plain["top1"], m_plain["top5"], m_plain["balanced_acc"], m_plain["macro_f1"])]
        json_plain["splits"][split] = m_plain

        # zero-shot (enriched) 
        m_enr = eval_with_text(X, y, T_enr, K, split, args.backbone, args.results_dir, mode_tag="enriched")
        rows += [(split,"zero-shot-enriched", m_enr["top1"], m_enr["top5"], m_enr["balanced_acc"], m_enr["macro_f1"])]
        json_enr["splits"][split] = m_enr

    # write original CSV (now with both zs variants)
    out_csv = os.path.join(args.results_dir, f"metrics_{args.backbone.replace(' ','_')}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w=csv.writer(f); w.writerow(["split","model","top1","top5","balanced_acc","macro_f1"]); w.writerows(rows)
    print(f"Wrote metrics → {out_csv}")

    # write JSONs for quick comparisons
    with open(args.out_plain_json, "w") as f:
        json.dump(json_plain, f, indent=2)
    with open(args.out_enriched_json, "w") as f:
        json.dump(json_enr, f, indent=2)
    print(f"Wrote JSON → {args.out_plain_json}")
    print(f"Wrote JSON → {args.out_enriched_json}")

if __name__ == "__main__":
    main()
