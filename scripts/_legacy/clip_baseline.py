#!/usr/bin/env python
import os, argparse, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

ROOT_DEFAULT = "/zfs/ai4good/datasets/mushroom"
USER_NAME=input("input your user name: ")
USER = os.environ.get("USER", USER_NAME)          
CACHE_DEFAULT = f"/zfs/ai4good/student/{USER}/cache/open_clip"

CAND_IMAGE_COLS = ["image_path","filepath","path","image","img_path","file"]
CAND_LABEL_COLS = ["label","species","class","target","y"]

def guess_cols(df):
    img_col = next((c for c in CAND_IMAGE_COLS if c in df.columns), None)
    lab_col = next((c for c in CAND_LABEL_COLS if c in df.columns), None)
    if not img_col or not lab_col:
        raise ValueError(f"CSV needs image+label columns. Found {list(df.columns)}")
    return img_col, lab_col

def resolve_path(p: str, root: str, kaggle_prefix="/kaggle/working") -> str:
    """Map Kaggle-style absolute paths to your ZFS dataset root."""
    p = str(p)
    if os.path.exists(p):
        return p
    if p.startswith(kaggle_prefix):
        cand = p.replace(kaggle_prefix, root, 1)
        if os.path.exists(cand):
            return cand
    # robust fallback: stitch from 'merged_dataset/...'
    key = "merged_dataset/"
    if key in p:
        rel = p.split(key, 1)[1]
        cand = os.path.join(root, key, rel)
        if os.path.exists(cand):
            return cand
    # final attempt: join root + stripped path
    cand = os.path.join(root, p.lstrip("/"))
    return cand if os.path.exists(cand) else p  # will error later if missing

class CSVDataset(Dataset):
    def __init__(self, csv_path, root=None, transform=None, label2idx=None):
        self.df = pd.read_csv(csv_path)
        self.img_col, self.lab_col = guess_cols(self.df)
        self.root = root
        self.transform = transform
        if label2idx is None:
            labels = sorted(self.df[self.lab_col].unique())
            self.label2idx = {l:i for i,l in enumerate(labels)}
        else:
            self.label2idx = label2idx
        self.idx2label = {v:k for k,v in self.label2idx.items()}

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = resolve_path(r[self.img_col], self.root)
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        y = self.label2idx[r[self.lab_col]]
        return img, y

def subset_by_k_per_class(df, lab_col, k, seed=0):
    if k <= 0: return df.iloc[0:0]
    rng = np.random.RandomState(seed)
    parts = []
    for _, sub in df.groupby(lab_col, sort=False):
        parts.append(sub.sample(n=min(k, len(sub)), random_state=rng))
    return pd.concat(parts, ignore_index=True)

def limit_per_class(df, lab_col, m):
    if m is None or m <= 0: return df
    outs = []
    for _, sub in df.groupby(lab_col, sort=False):
        outs.append(sub.iloc[:m])
    return pd.concat(outs, ignore_index=True)

def build_prompts(classnames):
    templates = [
        "a photo of a {} mushroom.",
        "a close-up photo of a {}.",
        "a natural photo of {} in the wild.",
        "a macro shot of a {} cap.",
        "a field guide photo of {}.",
    ]
    return [[t.format(c.replace("_"," ").replace("-", " ")) for t in templates] for c in classnames]

@torch.no_grad()
def encode_texts(model, tokenizer, prompts, device):
    class_feats = []
    for plist in prompts:
        toks = tokenizer(plist)
        feats = model.encode_text(toks.to(device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        class_feats.append(feats.mean(dim=0))
    return torch.stack(class_feats, dim=0)  # [C, D]

@torch.no_grad()
def encode_images(model, loader, device):
    feats, ys = [], []
    for x, y in tqdm(loader, desc="encode_images"):
        x = x.to(device, non_blocking=True)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu()); ys.append(y)
    return torch.cat(feats, 0).numpy(), torch.cat(ys, 0).numpy()

def accuracy(logits, y):
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--root", default=ROOT_DEFAULT, help="ZFS dataset root with merged_dataset/")
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")  # e.g., laion2b_s34b_b79k
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=max(2, os.cpu_count()//2))
    ap.add_argument("--shots", type=int, default=0, help="K shots/class for linear probe (0 = zero-shot only)")
    ap.add_argument("--limit_per_class", type=int, default=50, help="limit val per class for speed; <=0 = all")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=CACHE_DEFAULT, help="where to cache CLIP weights (on ZFS)")
    args = ap.parse_args()

    # put caches on ZFS to avoid home-quota
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ.setdefault("OPENCLIP_CACHE_DIR", args.cache_dir)
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device, cache_dir=args.cache_dir
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    # load CSVs (with Kaggle-style paths)
    df_tr = pd.read_csv(args.train_csv)
    img_col, lab_col = guess_cols(df_tr)

    df_va = pd.read_csv(args.val_csv)
    df_va = limit_per_class(df_va, lab_col, args.limit_per_class)

    # label mapping from train
    labels = sorted(df_tr[lab_col].unique())
    label2idx = {l:i for i,l in enumerate(labels)}

    # few-shot subset (optional)
    df_shots = subset_by_k_per_class(df_tr, lab_col, k=args.shots, seed=args.seed) if args.shots>0 else df_tr.iloc[0:0]

    # build datasets without rewriting CSV files
    ds_val = CSVDataset.__new__(CSVDataset)
    ds_val.df = df_va.reset_index(drop=True)
    ds_val.img_col, ds_val.lab_col = img_col, lab_col
    ds_val.root, ds_val.transform = args.root, preprocess
    ds_val.label2idx = label2idx
    ds_val.idx2label = {v:k for k,v in label2idx.items()}

    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Zero-shot
    prompts = build_prompts(labels)
    text_feats = encode_texts(model, tokenizer, prompts, device)            # [C,D]
    img_feats, y_val = encode_images(model, val_loader, device)             # [N,D],[N]
    zsl_logits = (img_feats @ text_feats.cpu().numpy().T) * 100.0           # cosine ~ logits
    zsl_acc = accuracy(zsl_logits, y_val)
    print(f"[Zero-shot] top-1 acc: {zsl_acc:.4f}  (limit_per_class={args.limit_per_class})")

    # Few-shot linear probe
    if args.shots > 0 and len(df_shots) > 0:
        ds_shots = CSVDataset.__new__(CSVDataset)
        ds_shots.df = df_shots.reset_index(drop=True)
        ds_shots.img_col, ds_shots.lab_col = img_col, lab_col
        ds_shots.root, ds_shots.transform = args.root, preprocess
        ds_shots.label2idx = label2idx
        ds_shots.idx2label = {v:k for k,v in label2idx.items()}

        shot_loader = DataLoader(ds_shots, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        x_shot, y_shot = encode_images(model, shot_loader, device)
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs")
        clf.fit(x_shot, y_shot)
        lp_acc = clf.score(img_feats, y_val)
        print(f"[Few-shot {args.shots}-shot] val acc: {lp_acc:.4f}")

if __name__ == "__main__":
    main()
