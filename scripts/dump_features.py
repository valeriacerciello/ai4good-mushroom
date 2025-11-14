#!/usr/bin/env python3
import argparse, os, csv, numpy as np, torch, open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def read_csv(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        assert {"filepath", "label"}.issubset(r.fieldnames), \
            f"CSV must have columns: filepath,label"
        for row in r:
            rows.append((row["filepath"], row["label"]))
    return rows

def load_label_map(labels_tsv):
    ids, names = [], []
    with open(labels_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            ids.append(int(parts[0]))
            names.append(parts[1])
    return {n: i for i, n in zip(ids, names)}, names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--csv", required=True)                 # splits/train.csv | val.csv | test.csv
    ap.add_argument("--labels", required=True)              # labels.tsv
    ap.add_argument("--backbone", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--save-dir", default="features")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    # model + preprocess
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.backbone,
        pretrained=args.pretrained,
        device=args.device
    )
    model.eval()

    name2id, _ = load_label_map(args.labels)
    rows = read_csv(args.csv)

    feats = []
    labels = []
    rel_paths = []

    with torch.no_grad():
        for i in tqdm(range(0, len(rows), args.batch_size), ncols=80):
            batch_rows = rows[i:i + args.batch_size]

            ims = []
            batch_labels = []
            batch_rel = []

            for rel_path, lbl_name in batch_rows:
                full_path = os.path.join(args.data_root, rel_path)

                # 1) If file is missing, skip
                if not os.path.exists(full_path):
                    print(f"[WARN] missing image: {full_path} — skipping")
                    continue

                try:
                    img = Image.open(full_path).convert("RGB")
                except FileNotFoundError:
                    print(f"[WARN] FileNotFoundError when opening {full_path} — skipping")
                    continue

                ims.append(preprocess(img))
                batch_labels.append(name2id[lbl_name])
                batch_rel.append(rel_path)

            if not ims:
                # whole batch was missing → continue
                continue

            ims_tensor = torch.stack(ims).to(args.device)
            f = model.encode_image(ims_tensor)
            f = f / f.norm(dim=-1, keepdim=True)

            feats.append(f.float().cpu().numpy())
            labels.extend(batch_labels)
            rel_paths.extend(batch_rel)

    if not feats:
        raise RuntimeError("No features extracted — all images seem to be missing or failed to load.")

    X = np.concatenate(feats, axis=0)
    y = np.array(labels, dtype=np.int64)
    rel = np.array(rel_paths, dtype=object)

    out_dir = Path(args.save-dir) / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    split = Path(args.csv).stem  # train/val/test

    np.savez_compressed(out_dir / f"{split}.npz", X=X, y=y, paths=rel)
    print(f"Saved {X.shape} → {out_dir / f'{split}.npz'}")

if __name__ == "__main__":
    main()
