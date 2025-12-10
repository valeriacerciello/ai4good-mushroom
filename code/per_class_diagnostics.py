#!/usr/bin/env python3
"""Per-class diagnostics: train linear probe from cached features and evaluate per-class accuracies
Outputs:
- diagnostics/per_class_fusion.csv
- diagnostics/summary.txt
- prints top-10 helped/harmed species
"""
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import argparse

DATA_DIR = '/zfs/ai4good/datasets/mushroom'
FEATURES_DIR = '/zfs/ai4good/student/hgupta/clip_features'
PROMPTS_FILE = 'data/delta_prompts.json'
TRAIN_FEAT = os.path.join(FEATURES_DIR, 'train.pt')


def load_resnet(checkpoint_path, num_classes, device):
    from torchvision import models
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    cleaned = {}
    for k, v in state_dict.items():
        newk = k.replace('model.', '').replace('net.', '')
        cleaned[newk] = v
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.eval()
    return model


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, data_dir, preprocess, species_to_idx):
        import pandas as pd
        df = pd.read_csv(csv_path)
        # detect image/path columns
        if 'image_path' in df.columns:
            img_col = 'image_path'
        elif 'path' in df.columns:
            img_col = 'path'
        else:
            img_col = df.columns[0]
        if 'label' in df.columns:
            label_col = 'label'
        elif 'species' in df.columns:
            label_col = 'species'
        else:
            label_col = df.columns[1] if len(df.columns) > 1 else img_col
        self.rows = []
        for _, r in df.iterrows():
            p = r.get(img_col)
            if not isinstance(p, str):
                p = str(p)
            p = p if os.path.isabs(p) else os.path.join(data_dir, p)
            if not os.path.exists(p):
                continue
            label = r.get(label_col)
            idx = species_to_idx.get(label, None)
            if idx is None:
                try:
                    idx = int(label)
                except Exception:
                    idx = -1
            self.rows.append((p, idx, label))
        self.preprocess = preprocess
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        p, idx, lname = self.rows[i]
        img = Image.open(p).convert('RGB')
        img_t = self.preprocess(img)
        return img_t, idx, lname


def train_linear_probe_from_features(train_feats, train_labels, num_classes, device, epochs=50, lr=0.03, wd=1e-4, batch_size=64):
    ds = TensorDataset(train_feats, train_labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    it = iter(dl)
    feats0, labels0 = next(it)
    feat_dim = feats0.shape[1]
    feat_dtype = feats0.dtype
    clf = torch.nn.Linear(feat_dim, num_classes).to(device, dtype=feat_dtype)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        clf.train()
        pbar = tqdm(dl, desc=f'LP Epoch {epoch+1}/{epochs}')
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = clf(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({'loss': float(loss.detach().cpu())})
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='PE-Core-bigG-14-448')
    parser.add_argument('--features_dir', type=str, default=FEATURES_DIR)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device', device)

    with open(PROMPTS_FILE, 'r') as f:
        all_prompts = json.load(f)
    species_list = sorted(all_prompts.keys())
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes = len(species_list)
    print('Num classes', num_classes)

    # Load CLIP model (open_clip if needed)
    try:
        import clip
        clip_model, preprocess = clip.load(args.backbone, device=device)
        print('Loaded clip via clip')
    except Exception:
        import open_clip
        # try fallback
        clip_model, preprocess = None, None
        for name in [args.backbone, args.backbone.replace('-', '_'), args.backbone.lower()]:
            try:
                m, _, p = open_clip.create_model_and_transforms(name, pretrained='meta')
                clip_model = m.to(device); preprocess = p
                print('Loaded open_clip', name)
                break
            except Exception:
                continue
        if clip_model is None:
            raise RuntimeError('Could not load clip model')
    clip_model.eval()

    # Load ResNet
    # find checkpoint
    import glob
    ckpt_pattern = "/zfs/ai4good/student/hgupta/logs/mushroom/train/runs/*/checkpoints/*.ckpt"
    cands = sorted(glob.glob(ckpt_pattern), key=os.path.getmtime, reverse=True)
    if not cands:
        raise RuntimeError('No resnet checkpoint')
    ckpt = cands[0]
    resnet = load_resnet(ckpt, num_classes=num_classes, device=device)

    # Load cached train features
    train_pt = os.path.join(args.features_dir, 'train.pt')
    if not os.path.exists(train_pt):
        raise RuntimeError('train.pt not found in features_dir')
    cached = torch.load(train_pt, map_location='cpu')
    train_feats = cached['features']
    train_labels = cached['labels']
    # ensure long
    train_labels = train_labels.long()
    print('Loaded train feats', train_feats.shape)

    # Train linear probe from features
    clf = train_linear_probe_from_features(train_feats, train_labels, num_classes, device, epochs=args.epochs, lr=0.03, wd=1e-4, batch_size=args.batch_size)
    clf.eval()

    # Create val loader
    val_ds = ValDataset(os.path.join(DATA_DIR, 'val.csv'), DATA_DIR, preprocess, species_to_idx)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    print('Val samples', len(val_ds))

    # storage
    records = []
    with torch.no_grad():
        for imgs, labels_idx, species_name in tqdm(val_dl, desc='Eval'):
            imgs = imgs.to(device)
            # CLIP features
            feats = clip_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits_clip = clf(feats)
            probs_clip = torch.softmax(logits_clip, dim=1).cpu()
            # ResNet
            res_logits = resnet(imgs).cpu()
            probs_res = torch.softmax(res_logits, dim=1)
            for i in range(len(labels_idx)):
                rec = {
                    'species': species_name[i],
                    'label_idx': int(labels_idx[i]),
                    'clip_probs': probs_clip[i].numpy(),
                    'res_probs': probs_res[i].numpy()
                }
                records.append(rec)

    # aggregate per-class
    per = {}
    for r in records:
        sp = r['species']
        idx = r['label_idx']
        if idx < 0 or idx >= num_classes:
            continue
        if sp not in per:
            per[sp] = {'idx': idx, 'clip_correct':0, 'res_correct':0, 'fusion_correct':0, 'n':0}
        # predictions
        clip_pred = int(np.argmax(r['clip_probs']))
        res_pred = int(np.argmax(r['res_probs']))
        # fusion using provided per-class alpha (ResNet weight)
        alpha = float(all_prompts.get(sp, [])[0]) if False else None
        # get alpha from class_alphas.json
    # load class alphas
    ca = None
    ca_path = os.path.join('data','class_alphas.json')
    if os.path.exists(ca_path):
        with open(ca_path,'r') as f:
            ca_raw = json.load(f)
        ca = [ca_raw.get(s, 0.0) for s in species_list]
        ca = np.array(ca)
    else:
        ca = np.zeros(num_classes)

    for r in records:
        sp = r['species']
        idx = r['label_idx']
        if idx < 0 or idx >= num_classes:
            continue
        clip_pred = int(np.argmax(r['clip_probs']))
        res_pred = int(np.argmax(r['res_probs']))
        alpha = float(ca[idx])
        # fused probs
        fused = r['res_probs'] * alpha + r['clip_probs'] * (1.0 - alpha)
        fused_pred = int(np.argmax(fused))
        entry = per[sp]
        entry['n'] += 1
        if clip_pred == idx:
            entry['clip_correct'] += 1
        if res_pred == idx:
            entry['res_correct'] += 1
        if fused_pred == idx:
            entry['fusion_correct'] += 1

    rows = []
    for sp, v in per.items():
        n = v['n']
        rows.append({
            'species': sp,
            'idx': v['idx'],
            'alpha': float(ca[v['idx']]),
            'clip_acc': v['clip_correct']/n if n>0 else 0.0,
            'res_acc': v['res_correct']/n if n>0 else 0.0,
            'fusion_acc': v['fusion_correct']/n if n>0 else 0.0,
            'n': n,
            'delta_vs_clip': (v['fusion_correct']/n - v['clip_correct']/n) if n>0 else 0.0
        })
    df = pd.DataFrame(rows).sort_values('delta_vs_clip')
    os.makedirs('diagnostics', exist_ok=True)
    out_csv = 'diagnostics/per_class_fusion.csv'
    df.to_csv(out_csv, index=False)

    # summary
    top_harmed = df.head(10)
    top_helped = df.tail(10).iloc[::-1]
    with open('diagnostics/summary.txt','w') as f:
        f.write('Top 10 harmed (by delta_vs_clip):\n')
        f.write(top_harmed.to_string(index=False))
        f.write('\n\nTop 10 helped:\n')
        f.write(top_helped.to_string(index=False))
    print('Wrote', out_csv, 'and diagnostics/summary.txt')

    # Test inverted semantics quickly: use alpha as CLIP weight
    total = len(records)
    correct_inv = 0
    for r in records:
        idx = r['label_idx']
        alpha = float(ca[idx])
        fused_inv = r['res_probs'] * (1.0 - alpha) + r['clip_probs'] * alpha
        if int(np.argmax(fused_inv)) == idx:
            correct_inv += 1
    print('Inverted semantics top1:', correct_inv/total)

if __name__ == '__main__':
    main()
