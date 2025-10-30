#!/usr/bin/env python3
import argparse, os, json, math, numpy as np, torch, open_clip
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# helpers
def load_npz_features(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"].astype(np.float32)        # [N, D] already normalized by our pipeline
    y = z["y"].astype(np.int64)          # [N]
    paths = z["paths"]                   # [N] relative paths
    return X, y, paths

def load_labels(labels_tsv: str) -> np.ndarray:
    names = []
    with open(labels_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            _i, name, *_ = line.split("\t")
            names.append(name)
    return np.array(names, dtype=object)

def load_prompts_json(path: str) -> Dict[str, List[str]]:
    if not path or not os.path.isfile(path):
        return {}
    obj = json.load(open(path, "r", encoding="utf-8"))
    out = {}
    for k, v in obj.items():
        if isinstance(v, list):
            out[k] = [s for s in v if isinstance(s, str) and s.strip()]
    return out

@torch.no_grad()
def build_text_matrix_for_classes(
    class_names: np.ndarray,
    prompts_dict: Dict[str, List[str]],
    tokenizer, model, device,
    fallback_template="a photo of {}"
) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
    """
    Returns:
      T: [D, K] mean text embedding per class
      per_class_prompt_embs: dict[class] -> [P, D] normalized text embeddings of that class' prompts
    """
    model.eval()
    T_list = []
    per_class_prompt_embs = {}
    for cls in class_names:
        prompts = prompts_dict.get(cls, None)
        if not prompts:
            prompts = [fallback_template.format(cls)]
        toks = tokenizer(prompts).to(device)
        txt = model.encode_text(toks)                 # [P, D]
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-12)
        per_class_prompt_embs[cls] = txt              # cache full prompt set for “winning prompt”
        mean = txt.mean(dim=0, keepdim=True)          # [1, D]
        mean = mean / (mean.norm(dim=-1, keepdim=True) + 1e-12)
        T_list.append(mean.cpu())
    T = torch.cat(T_list, dim=0).T.contiguous().float().cpu().numpy()  # [D, K]
    return T, per_class_prompt_embs

def topk(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
    # sort those top-k
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    top_val = np.take_along_axis(scores, top_idx, axis=1)
    return top_idx, top_val

def format_top5(class_names, idx_row, val_row) -> List[str]:
    lines = []
    for r in range(len(idx_row)):
        c = class_names[idx_row[r]]
        s = float(val_row[r])
        lines.append(f"{r+1}. {c}  ({s:.3f})")
    return lines

def pick_winning_prompt(img_feat: np.ndarray, prompt_embs: torch.Tensor) -> int:
    # img_feat: [D], prompt_embs: [P, D] (both L2-normalized)
    t = prompt_embs.cpu().numpy().T  # [D, P]
    scores = img_feat @ t            # [P]
    return int(np.argmax(scores))

def main():
    ap = argparse.ArgumentParser(description="Make qualitative grid: image + top-5 (plain vs enriched) + winning prompts")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--features_dir", default="features")
    ap.add_argument("--backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--pretrained", default="openai")

    ap.add_argument("--split", choices=["val","test"], default="val")
    ap.add_argument("--labels", default="labels.tsv")

    ap.add_argument("--prompts_plain", default="data/prompts/class_prompts_plain.json")
    ap.add_argument("--prompts_enriched", default="data/prompts/class_prompts_enriched.json")

    ap.add_argument("--n_examples", type=int, default=12)
    ap.add_argument("--out", default="results/examples/qualitative.png")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    # 1) load CLIP features for the requested split
    npz_path = Path(args.features_dir) / args.backbone / f"{args.split}.npz"
    X, y, rel_paths = load_npz_features(npz_path)   # X normalized
    class_names = load_labels(args.labels)
    K = len(class_names)

    # 2) build text matrices for plain & enriched (and cache per-class prompt embeddings)
    model, _, _ = open_clip.create_model_and_transforms(args.backbone, pretrained=args.pretrained, device=args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.backbone)

    plain = load_prompts_json(args.prompts_plain)
    enr   = load_prompts_json(args.prompts_enriched)

    T_plain, plain_prompts_embs = build_text_matrix_for_classes(
        class_names, plain, tokenizer, model, args.device, fallback_template="a photo of {}"
    )
    T_enr, enr_prompts_embs = build_text_matrix_for_classes(
        class_names, enr, tokenizer, model, args.device, fallback_template="a close-up photo of {} mushroom"
    )

    # 3) compute scores and predictions
    S_plain = X @ T_plain      # [N, K]
    S_enr   = X @ T_enr

    yhat_plain = np.argmax(S_plain, axis=1)
    yhat_enr   = np.argmax(S_enr,   axis=1)
    correct_plain = (yhat_plain == y)
    correct_enr   = (yhat_enr   == y)

    # 4) select examples: mix of cases
    idx_A = np.where( (correct_enr==1) & (correct_plain==0) )[0]   # enriched fixes plain
    idx_B = np.where( (correct_enr==0) & (correct_plain==0) )[0]   # both wrong
    idx_C = np.where( (correct_enr==1) & (correct_plain==1) )[0]   # both correct

    sel = []
    def take_some(src, n):
        nonlocal sel
        for i in list(src)[:n]:
            sel.append(int(i))

    n_total = min(args.n_examples, len(X))
    per_bucket = max(1, n_total // 3)
    take_some(idx_A, per_bucket)
    take_some(idx_B, per_bucket)
    take_some(idx_C, n_total - len(sel))
    sel = sel[:n_total]

    # 5) build the figure (rows = examples, 3 cols: Image | PLAIN | ENRICHED)
    rows = len(sel); cols = 3
    plt.figure(figsize=(cols*7.2, rows*3.6), dpi=150)

    for r, i in enumerate(sel):
        # load image
        img_path = os.path.join(args.data_root, rel_paths[i])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224,224), (30,30,30))

        # top-5 for both modes
        t5p_idx, t5p_val = topk(S_plain[i:i+1], 5)
        t5e_idx, t5e_val = topk(S_enr[i:i+1],   5)
        t5p_idx, t5p_val = t5p_idx[0], t5p_val[0]
        t5e_idx, t5e_val = t5e_idx[0], t5e_val[0]

        # winning prompts for the top-1 prediction of each mode
        img_feat = X[i]   # [D], already L2 normalized
        top1_plain_cls = class_names[yhat_plain[i]]
        top1_enr_cls   = class_names[yhat_enr[i]]

        # get per-class prompt embeddings; if missing, fallback handled at build time
        prm_plain = plain_prompts_embs.get(top1_plain_cls)
        prm_enr   = enr_prompts_embs.get(top1_enr_cls)

        w_plain = pick_winning_prompt(img_feat, prm_plain) if prm_plain is not None else -1
        w_enr   = pick_winning_prompt(img_feat, prm_enr)   if prm_enr   is not None else -1

        # get raw prompt strings (fall back if dict empty)
        best_plain_prompt = (plain.get(top1_plain_cls) or [f"a photo of {top1_plain_cls}"])[w_plain if w_plain>=0 else 0]
        best_enr_prompt   = (enr.get(top1_enr_cls)   or [f"a close-up photo of {top1_enr_cls} mushroom"])[w_enr if w_enr>=0 else 0]

        # plot: image
        ax_img = plt.subplot(rows, cols, r*cols + 1)
        ax_img.imshow(img); ax_img.axis("off")
        gt = class_names[y[i]]
        title_color = "#2ca02c" if yhat_enr[i]==y[i] else "#d62728"
        ax_img.set_title(f"{os.path.basename(rel_paths[i])}\nGT: {gt}", fontsize=10, color=title_color)

        # plot: PLAIN text block
        ax_p = plt.subplot(rows, cols, r*cols + 2)
        ax_p.axis("off")
        lines_p = [f"PLAIN  (pred: {top1_plain_cls})", ""]
        lines_p += format_top5(class_names, t5p_idx, t5p_val)
        lines_p += ["", f"Winning prompt:", f"“{best_plain_prompt}”"]
        ax_p.text(0.01, 0.98, "\n".join(lines_p), va="top", ha="left", fontsize=9, family="monospace")

        # plot: ENRICHED text block
        ax_e = plt.subplot(rows, cols, r*cols + 3)
        ax_e.axis("off")
        lines_e = [f"ENRICHED  (pred: {top1_enr_cls})", ""]
        lines_e += format_top5(class_names, t5e_idx, t5e_val)
        lines_e += ["", f"Winning prompt:", f"“{best_enr_prompt}”"]
        ax_e.text(0.01, 0.98, "\n".join(lines_e), va="top", ha="left", fontsize=9, family="monospace")

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"[done] saved → {args.out}")

if __name__ == "__main__":
    main()
