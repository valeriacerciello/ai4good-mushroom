"""Few-shot linear-probe ResNet on the complement (unseen) labels.

For each requested shot count k, this script:
- samples up to k training images per complement class from the full `train.csv`
- builds a support set (few-shot) and trains a linear classifier (fc) on top of ResNet18
  initialized from the existing `experiment_results/ckpt_resnet_train84.pth` backbone when available
- evaluates the fine-tuned model on `experiment_results/splits/held_complement_test.csv`
- saves per-shot metrics and (optionally) the trained fc checkpoint.

Saves results to `experiment_results/figs/resnet_fewshot_results.json` and per-shot checkpoints.
"""
import argparse
import random
import json
from pathlib import Path
import time

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
SPLITS = ER / 'splits'
FIGS = ER / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)


class CsvImageDataset(Dataset):
    def __init__(self, rows, label_to_idx, transform=None):
        # rows: list of (image_path, label)
        self.rows = rows
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


def sample_support(full_train_df, labels, shots, seed=42):
    # full_train_df has columns image_path,label
    rows = []
    rng = random.Random(seed)
    for lab in labels:
        dfc = full_train_df[full_train_df['label'] == lab]
        paths = dfc['image_path'].tolist()
        if len(paths) == 0:
            continue
        rng.shuffle(paths)
        k = min(shots, len(paths))
        for p in paths[:k]:
            rows.append((p, lab))
    return rows


def evaluate_model(model, device, test_rows, labels, batch_size=64):
    model.eval()
    tfm = fcr.get_resnet_val_transforms()
    ds = CsvImageDataset(test_rows, {l: i for i, l in enumerate(labels)}, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    probs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(prob)
    probs = np.vstack(probs)
    y_true = [r[1] for r in test_rows]
    metrics = fcr.evaluate_metrics(y_true, probs, labels)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', type=int, nargs='+', default=[1,2,5,10], help='Shot counts to run')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backbone_ckpt', type=str, default=str(ER / 'ckpt_resnet_train84.pth'))
    parser.add_argument('--save_ckpt', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load train_labels_84 to determine complement labels
    labels_file = SPLITS / 'train_labels_84.txt'
    if not labels_file.exists():
        raise FileNotFoundError(f'Missing {labels_file}; please run train_resnet_on_label_subset.py first')
    train_labels = [ln.strip() for ln in open(labels_file).read().splitlines() if ln.strip()]

    # Build complement labels from full dataset
    full_train_df = fcr.load_split_df('train')
    all_labels = fcr.load_labels([full_train_df])
    comp_labels = [l for l in all_labels if l not in set(train_labels)]
    print(f'Complement labels: {len(comp_labels)}')

    # Load held complement test rows
    held_comp_csv = SPLITS / 'held_complement_test.csv'
    if not held_comp_csv.exists():
        raise FileNotFoundError(f'Missing {held_comp_csv}; please run train_resnet_on_label_subset.py first')
    df_test = pd.read_csv(held_comp_csv)
    test_rows = df_test[['image_path','label']].values.tolist()

    results = {}

    device = torch.device(args.device)

    for shots in args.shots:
        print(f'Running {shots}-shot...')
        support_rows = sample_support(full_train_df, comp_labels, shots, seed=args.seed)
        print(f'Collected support size: {len(support_rows)} (up to {shots} per class)')

        # Build label->idx for complement labels
        label_to_idx = {l: i for i, l in enumerate(comp_labels)}

        # Datasets
        tfm_train = get_fewshot_train_transform = transforms.Compose([
            transforms.RandomResizedCrop(384, scale=(0.7,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4308,0.4084,0.3393], std=[0.1063,0.1023,0.1123])
        ])
        train_ds = CsvImageDataset(support_rows, label_to_idx, transform=tfm_train)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # Initialize model from backbone ckpt if possible
        num_classes = len(comp_labels)
        model = models.resnet18(num_classes=num_classes)
        # Try load backbone weights and ignore fc if shape mismatch
        if Path(args.backbone_ckpt).exists():
            sd = torch.load(args.backbone_ckpt, map_location='cpu')
            state = sd.get('model_state_dict', sd.get('model_state', sd.get('state_dict', sd)))
            # Clean keys similar to fusion_clip_resnet.load_resnet_from_ckpt
            cleaned = {}
            for k, v in state.items():
                if k.startswith('net._orig_mod.'):
                    new_k = k[len('net._orig_mod.'):]
                elif k.startswith('net.'):
                    new_k = k[len('net.'):]
                else:
                    new_k = k
                cleaned[new_k] = v
            # Remove fc if size mismatch
            ck_fc_w = cleaned.get('fc.weight')
            if ck_fc_w is not None and ck_fc_w.shape != model.fc.weight.shape:
                cleaned.pop('fc.weight', None)
                cleaned.pop('fc.bias', None)
            model.load_state_dict(cleaned, strict=False)

        model = model.to(device)

        # Freeze backbone except fc
        for name, p in model.named_parameters():
            if not name.startswith('fc'):
                p.requires_grad = False

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Train linear head
        for ep in range(1, args.epochs+1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            t0 = time.time()
            for xb, yb in train_loader:
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
            train_loss = running_loss / max(1, total)
            train_acc = correct / max(1, total)
            if ep % 5 == 0 or ep == args.epochs:
                print(f'Shot {shots} Epoch {ep}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} time={time.time()-t0:.1f}s')

        # Optionally save the fc-only checkpoint
        if args.save_ckpt:
            ck = ER / f'ckpt_resnet_fewshot_{shots}shot.pth'
            torch.save({'model_state_dict': model.state_dict(), 'labels': comp_labels}, ck)
            print('Saved few-shot checkpoint to', ck)

        # Evaluate on held complement
        metrics = evaluate_model(model, device, test_rows, comp_labels, batch_size=64)
        print(f'{shots}-shot eval metrics:', metrics)
        results[shots] = metrics

    # Save results
    outp = FIGS / 'resnet_fewshot_results.json'
    json.dump(results, open(outp, 'w'), indent=2)
    print('Saved few-shot results to', outp)


if __name__ == '__main__':
    main()
