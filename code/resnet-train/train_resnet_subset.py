"""Train a ResNet on half the images per class and save a checkpoint.

Workflow:
- Build per-class half split from DATA_ROOT/train.csv (train half) and remainder (held-out)
- Save split CSVs to experiment_results/splits/
- Train a ResNet (configurable) on the train-half
- Save final checkpoint to experiment_results/ckpt_resnet_half.pth

This script is intentionally conservative and configurable. It will use CUDA if available
in the active Python environment.
"""
import argparse
import os
import random
from pathlib import Path
import csv
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

import fusion_clip_resnet as fcr


ROOT = Path(__file__).resolve().parent
ER = ROOT / "experiment_results"
SPLITS_DIR = ER / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
ER.mkdir(parents=True, exist_ok=True)


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


def build_half_splits(df_train):
    # df_train is a pandas DataFrame with columns ['image_path','label']
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


def save_csv(rows, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path', 'label'])
        for p, lab in rows:
            w.writerow([p, lab])


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_ckpt', type=str, default=str(ER / 'ckpt_resnet_half.pth'))
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    import pandas as pd

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Loading full train split...')
    df_train_full = fcr.load_split_df('train')
    print(f'Full train size: {len(df_train_full)}')

    print('Building half-splits per class...')
    train_rows, held_rows = build_half_splits(df_train_full)
    train_csv = SPLITS_DIR / 'train_half.csv'
    held_csv = SPLITS_DIR / 'held_half.csv'
    save_csv(train_rows, train_csv)
    save_csv(held_rows, held_csv)
    print(f'Saved splits: {train_csv} ({len(train_rows)}), {held_csv} ({len(held_rows)})')

    # Build labels and mapping from union of splits
    labels = fcr.load_labels([pd.DataFrame(train_rows, columns=['image_path','label']),
                              pd.DataFrame(held_rows, columns=['image_path','label'])])
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Datasets
    train_df = pd.read_csv(train_csv)
    val_df = train_df.sample(frac=0.05, random_state=args.seed).reset_index(drop=True)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    held_df = pd.read_csv(held_csv)

    train_ds = CsvImageDataset(train_df, label_to_idx, transform=get_train_transforms())
    val_ds = CsvImageDataset(val_df, label_to_idx, transform=fcr.get_resnet_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    num_classes = len(labels)
    print(f'Number of classes: {num_classes}')

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
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        t1 = time.time()
        print(f'Epoch {ep}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  time={t1-t0:.1f}s')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'labels': labels}, args.out_ckpt)
            print('Saved best checkpoint to', args.out_ckpt)

    # Save final artifacts
    meta = {'args': vars(args), 'best_val_acc': best_val_acc, 'labels': labels}
    with open(ER / 'ckpt_resnet_half_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    with open(ER / 'ckpt_resnet_half_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print('Done. Checkpoint and metadata written to experiment_results/')


if __name__ == '__main__':
    main()
