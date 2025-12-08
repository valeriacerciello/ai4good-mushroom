#!/usr/bin/env python3
"""Precompute CLIP image features for train/val/test datasets.

Saves tensors as `{split}.pt` with keys `features` (FloatTensor NxD) and `labels` (LongTensor N)
into a directory specified by `--output_dir` (default: `/zfs/ai4good/student/hgupta/clip_features`).
"""

import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import pandas as pd
import sys

def build_datasets(data_dir, split_csv, transform, shots=-1, species_list=None, species_to_idx=None, fewshot=False):
    df = pd.read_csv(split_csv)
    # resolve image paths
    def _resolve(p):
        if pd.isna(p):
            return ''
        p = str(p)
        return p if os.path.isabs(p) else os.path.join(data_dir, p)

    if fewshot:
        # few-shot sampling per species
        if 'species' in df.columns:
            label_col = 'species'
        elif 'label' in df.columns:
            label_col = 'label'
        else:
            label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        if species_list is None:
            species_list = sorted(df[label_col].unique())

        rows = []
        for sp in species_list:
            sp_rows = df[df[label_col] == sp]
            if len(sp_rows) == 0:
                continue
            if shots <= 0:
                sampled = sp_rows
            else:
                sampled = sp_rows.sample(n=min(shots, len(sp_rows)), random_state=42)
            rows.append(sampled)

        if len(rows) == 0:
            df2 = pd.DataFrame(columns=df.columns)
        else:
            df2 = pd.concat(rows).reset_index(drop=True)
        df = df2

    df['__resolved_path'] = df[df.columns[0]].apply(_resolve) if 'image_path' not in df.columns and 'path' not in df.columns else df.get('image_path', df.get('path')).apply(_resolve)
    exists_mask = df['__resolved_path'].apply(lambda p: os.path.exists(p))
    if exists_mask.sum() < len(df):
        print(f"Warning: {len(df)-int(exists_mask.sum())} missing images in {split_csv}; they will be ignored.")
    df = df[exists_mask].reset_index(drop=True)

    return df


def load_clip_model(backbone, device):
    try:
        import clip
        model, preprocess = clip.load(backbone, device=device)
        return model, preprocess
    except Exception:
        import open_clip
        # try open_clip variants as in final_fusion
        requested = backbone
        oc_variants = [requested, requested.replace('-', '_'), requested.lower(), requested.replace('/', '_'), requested.replace('/', '_').lower()]
        oc_variants += ['PE-Core-bigG-14-448', 'PE-Core-bigG-14-224']
        pretrained_candidates = ['meta', 'openai', 'laion2b_s13b_b90k', 'laion400m_e32']
        oc_model = None
        oc_preprocess = None
        for name in oc_variants:
            for src in pretrained_candidates:
                try:
                    oc_model, _, oc_preprocess = open_clip.create_model_and_transforms(name, pretrained=src)
                    break
                except Exception:
                    continue
            if oc_model is not None:
                break
        if oc_model is None:
            raise RuntimeError(f"Could not load CLIP backbone {backbone}")
        oc_model = oc_model.to(device)
        oc_model.eval()
        return oc_model, oc_preprocess


def encode_split(df, data_dir, model, preprocess, device, batch_size, num_workers, species_list=None):
    # Create a simple DataLoader that returns images and labels
    class ImgDataset(Dataset):
        def __init__(self, df, transform=None, label_col=None, species_to_idx=None):
            self.df = df
            self.transform = transform
            self.species_to_idx = species_to_idx or {}
            if label_col is None:
                if 'species' in df.columns:
                    self._label_col = 'species'
                elif 'label' in df.columns:
                    self._label_col = 'label'
                else:
                    self._label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            else:
                self._label_col = label_col

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = row['__resolved_path']
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)

            # Map species name to index when possible
            raw_label = row.get(self._label_col, None)
            label_idx = -1
            if raw_label is None:
                label_idx = -1
            else:
                # If the CSV label is already numeric index, try to coerce
                try:
                    label_idx = int(raw_label)
                except Exception:
                    # Treat as species name and map
                    label_idx = int(self.species_to_idx.get(str(raw_label), -1))

            return img, label_idx

    # Build species->index mapping if species_list provided
    species_to_idx = None
    if species_list is not None:
        species_to_idx = {s: i for i, s in enumerate(species_list)}

    ds = ImgDataset(df, transform=preprocess, species_to_idx=species_to_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    features_list = []
    labels_list = []
    with torch.no_grad():
        total_batches = len(dl)
        for i, (images, labels) in enumerate(tqdm(dl, desc="Encoding", unit="batch")):
            images = images.to(device)
            # Use AMP autocast for faster, lower-memory forward pass
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            features_list.append(feats.cpu())
            labels_list.append(torch.tensor(labels))
            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                print(f"Encoded batch {i+1}/{total_batches}", flush=True)

    if features_list:
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0).long()
    else:
        features = torch.empty(0)
        labels = torch.empty(0, dtype=torch.long)

    return features, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_backbone', type=str, default='PE-Core-bigG-14-448')
    parser.add_argument('--data_dir', type=str, default='/zfs/ai4good/datasets/mushroom')
    parser.add_argument('--shots', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='/zfs/ai4good/student/hgupta/clip_features')
    parser.add_argument('--splits', nargs='+', default=['train'], help='Which splits to encode: train val test')
    parser.add_argument('--prompts_file', type=str, default='data/delta_prompts.json')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    model, preprocess = load_clip_model(args.clip_backbone, device)

    # Load species order for few-shot sampling
    with open(args.prompts_file, 'r') as f:
        all_prompts = json.load(f)
    species_list = sorted(all_prompts.keys())

    for split in args.splits:
        csv_path = os.path.join(args.data_dir, f'{split}.csv')
        if not os.path.exists(csv_path):
            print(f'Warning: {csv_path} not found; skipping')
            continue
        print(f'Preparing {split} dataframe...')
        df = build_datasets(args.data_dir, csv_path, preprocess, shots=args.shots if split=='train' else -1, species_list=species_list, fewshot=(split=='train'))
        print(f'Encoding {split}: {len(df)} samples')
        feats, labels = encode_split(df, args.data_dir, model, preprocess, device, args.batch_size, args.num_workers, species_list=species_list)
        out_path = os.path.join(args.output_dir, f'{split}.pt')
        torch.save({'features': feats, 'labels': labels, 'species_list': species_list}, out_path)
        print(f'Wrote {out_path} ({feats.shape}, labels {labels.shape})')


if __name__ == '__main__':
    main()
