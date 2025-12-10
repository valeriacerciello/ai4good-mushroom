"""Train ResNet on a subset of labels and evaluate on the held-out label complement.

Workflow:
- Choose `n_train_labels` labels from the dataset labels (deterministic: first N sorted labels)
- Filter `train.csv` to only those labels and build half-splits per class (train/held)
- Train ResNet on that restricted training set
- After training, evaluate CLIP and the trained ResNet on the held split restricted to the *complement* labels
- Save checkpoint, metadata, held-subset CSV, and evaluation plots/JSON in `experiment_results/`.

Assumptions made (reasonable defaults):
- We pick the first `n_train_labels` from `fusion_clip_resnet.load_labels()` in sorted order for determinism.
  If you want a random selection, use --seed and set --randomize_labels.

"""
import argparse
import json
import random
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

import fusion_clip_resnet as fcr

ROOT = Path(__file__).resolve().parent
ER = ROOT / 'experiment_results'
FIGS = ER / 'figs'
SPLITS = ER / 'splits'
FIGS.mkdir(parents=True, exist_ok=True)
SPLITS.mkdir(parents=True, exist_ok=True)


class CsvImageDataset(Dataset):
    def __init__(self, df, label_to_idx, transform=None):
        self.rows = df[['image_path', 'label']].values.tolist()
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        p, lab = self.rows[idx]
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            img = Image.new('RGB', (384, 384), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        y = self.label_to_idx[lab]
        return img, y


def save_csv(rows, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path', 'label'])
        for p, lab in rows:
            w.writerow([p, lab])


def build_half_splits(df_train):
    grp = {}
    for i, row in df_train.reset_index(drop=True).iterrows():
        lab = row['label']
        grp.setdefault(lab, []).append(row['image_path'])

    train_rows = []
    held_rows = []
    for lab, paths in grp.items():
        random.shuffle(paths)
        n = len(paths)
        k = n // 2
        train_samples = paths[:k]
        held_samples = paths[k:]
        for p in train_samples:
            train_rows.append((p, lab))
        for p in held_samples:
            held_rows.append((p, lab))

    return train_rows, held_rows


def get_train_transforms(image_size=384):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4308, 0.4084, 0.3393], std=[0.1063, 0.1023, 0.1123]),
    ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return running_loss / total, correct / total


def run_evaluation_on_complement(ckpt_path, train_labels, prompts, clip_temp):
    # Determine complement labels from full held split
    held_csv = SPLITS / 'held_half.csv'
    if not held_csv.exists():
        raise FileNotFoundError(f'Held split not found: {held_csv}')
    df_held = pd.read_csv(held_csv)
    all_labels = fcr.load_labels([df_held])
    comp_labels = [l for l in all_labels if l not in set(train_labels)]
    print(f'Evaluating on complement labels: {len(comp_labels)} classes, filtering held to those labels')

    df_test = df_held[df_held['label'].isin(comp_labels)].reset_index(drop=True)
    save_csv(df_test.values.tolist(), SPLITS / 'held_complement_test.csv')

    # CLIP
    print('Computing CLIP probabilities on complement held set...')
    clip_probs = fcr.compute_clip_probs(df_test, comp_labels, prompts, temp=clip_temp)
    clip_metrics = fcr.evaluate_metrics(df_test['label'].tolist(), clip_probs, comp_labels)
    print('CLIP metrics on complement:', clip_metrics)

    # ResNet (load checkpoint; final fc will be reinitialized if shapes mismatch)
    print('Computing ResNet probabilities on complement held set from', ckpt_path)
    if not Path(ckpt_path).exists():
        print('Checkpoint not found; resnet probs will be uniform')
        res_probs = np.full_like(clip_probs, 1.0 / len(comp_labels))
    else:
        res_probs = fcr.compute_resnet_probs(df_test, comp_labels, ckpt_path)
    res_metrics = fcr.evaluate_metrics(df_test['label'].tolist(), res_probs, comp_labels)
    print('ResNet metrics on complement:', res_metrics)

    out = {
        'n_test': len(df_test),
        'train_labels': train_labels,
        'complement_labels_count': len(comp_labels),
        'clip': clip_metrics,
        'resnet': res_metrics,
    }
    json.dump(out, open(FIGS / 'train84_complement_eval.json', 'w'), indent=2)
    print('Saved complement evaluation to', FIGS / 'train84_complement_eval.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train_labels', type=int, default=84)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--out_ckpt', type=str, default=str(ER / 'ckpt_resnet_train84.pth'))
    parser.add_argument('--prompts', type=str, default=fcr.DELTA_PROMPTS)
    parser.add_argument('--clip_temp', type=float, default=fcr.CLIP_TEMP)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load full train split
    print('Loading full train split...')
    df_train_full = fcr.load_split_df('train')
    print(f'Full train size: {len(df_train_full)}')

    # Build label list and choose training labels (deterministic)
    all_labels = fcr.load_labels([df_train_full])
    if len(all_labels) < args.n_train_labels:
        raise ValueError('Not enough labels in dataset')
    train_labels = sorted(all_labels)[:args.n_train_labels]
    # Save train labels file
    labels_file = SPLITS / f'train_labels_{args.n_train_labels}.txt'
    with open(labels_file, 'w') as f:
        for l in train_labels:
            f.write(l + '\n')
    print(f'Selected {len(train_labels)} train labels; saved to {labels_file}')

    # Filter train_full to only these labels
    df_train_subset = df_train_full[df_train_full['label'].isin(train_labels)].reset_index(drop=True)
    print(f'Filtered train size: {len(df_train_subset)}')

    # Build half splits per class on the subset
    train_rows, held_rows = build_half_splits(df_train_subset)
    train_csv = SPLITS / 'train_train84_half.csv'
    held_csv = SPLITS / 'held_train84_half.csv'
    save_csv(train_rows, train_csv)
    save_csv(held_rows, held_csv)
    print(f'Saved splits: {train_csv} ({len(train_rows)}), {held_csv} ({len(held_rows)})')

    # Build label mapping
    labels = sorted(train_labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Datasets and loaders
    train_df = pd.read_csv(train_csv)
    val_df = train_df.sample(frac=0.05, random_state=args.seed).reset_index(drop=True)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)

    train_ds = CsvImageDataset(train_df, label_to_idx, transform=get_train_transforms())
    val_ds = CsvImageDataset(val_df, label_to_idx, transform=fcr.get_resnet_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device(args.device)
    num_classes = len(labels)
    print(f'Number of train classes: {num_classes}')

    # Create model
    if args.model == 'resnet18':
        model = models.resnet18(num_classes=num_classes)
    elif args.model == 'resnet50':
        model = models.resnet50(num_classes=num_classes)
    else:
        raise ValueError('Unsupported model')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    best_val_acc = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        t1 = time.time()
        print(f'Epoch {ep}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  time={t1-t0:.1f}s')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'labels': labels}, args.out_ckpt)
            print('Saved best checkpoint to', args.out_ckpt)

    # Save final metadata
    meta = {'args': vars(args), 'best_val_acc': best_val_acc, 'labels': labels}
    with open(ER / 'ckpt_resnet_train84_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Run evaluation on complement labels using the saved checkpoint
    run_evaluation_on_complement(args.out_ckpt, train_labels, args.prompts, args.clip_temp)

    print('All done.')


if __name__ == '__main__':
    main()
