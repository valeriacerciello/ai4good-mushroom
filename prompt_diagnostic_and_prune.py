#!/usr/bin/env python3
"""Compute per-prompt discriminativeness using CLIP and prune prompts.

Outputs:
- prompt_discriminativeness.csv
- delta_prompts.pruned.json (top K prompts per class)
- fusion_results_full_pruned.json (fusion run on full test set, using pruned prompts)
"""
import json
import os
from collections import defaultdict
import argparse

import torch
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np
import pandas as pd

# Project paths (match fusion script)
DATA_ROOT = "/zfs/ai4good/datasets/mushroom"
DELTA_PROMPTS = "./delta_prompts.json"
PRUNED_PROMPTS = "./delta_prompts.pruned.json"
CSV_OUT = "prompt_discriminativeness.csv"

# Load splits

def load_split(split):
    csv_path = os.path.join(DATA_ROOT, f"{split}.csv")
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['image_path'] = df.iloc[:, 0]
    df['label'] = df.iloc[:, 1]
    return df[['image_path','label']]


def get_label_images(df, label, max_per_label=50):
    sub = df[df['label']==label]
    return sub['image_path'].tolist()[:max_per_label]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune_k', type=int, default=8)
    parser.add_argument('--val_max_per_label', type=int, default=50)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)

    # load prompts
    with open(DELTA_PROMPTS, 'r') as f:
        prompts = json.load(f)

    # build labels list
    labels = sorted(list(prompts.keys()))

    # load val images
    df_val = load_split('val')

    # For each label, gather up to val_max_per_label images
    label_imgs = {l: get_label_images(df_val, l, max_per_label=args.val_max_per_label) for l in labels}

    # Precompute image embeddings (per-label average)
    print('Computing image embeddings...')
    img_embs = {}
    for l, imgs in label_imgs.items():
        embs = []
        for p in imgs:
            try:
                img = Image.open(p).convert('RGB')
                it = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    e = model.encode_image(it)
                    e = F.normalize(e, dim=-1)
                    embs.append(e.cpu())
            except Exception:
                continue
        if embs:
            img_embs[l] = torch.cat(embs, dim=0).mean(dim=0)
        else:
            img_embs[l] = None

    # For each prompt, compute text embedding and score: mean_sim_same - max_mean_sim_other
    rows = []
    pruned = {}
    for l in labels:
        candidate_prompts = prompts.get(l, [])
        scores = []
        for p in candidate_prompts:
            # encode text
            try:
                text_tok = clip.tokenize([p]).to(device)
                with torch.no_grad():
                    t_emb = model.encode_text(text_tok)
                    t_emb = F.normalize(t_emb, dim=-1).squeeze(0).cpu()
            except Exception:
                continue
            sims_same = []
            sims_others = defaultdict(list)
            for l2 in labels:
                if img_embs.get(l2) is None:
                    continue
                sim = float((t_emb @ img_embs[l2].cpu()).item())
                if l2 == l:
                    sims_same.append(sim)
                else:
                    sims_others[l2].append(sim)
            if not sims_same:
                mean_same = float('-inf')
            else:
                mean_same = float(np.mean(sims_same))
            # max mean similarity to any other class
            max_other = float('-inf')
            for l2, s in sims_others.items():
                if s:
                    mm = float(np.mean(s))
                    if mm > max_other:
                        max_other = mm
            if max_other == float('-inf'):
                max_other = float('-inf')
            score = mean_same - max_other
            rows.append({'label': l, 'prompt': p, 'mean_same': mean_same, 'max_other': max_other, 'score': score})
            scores.append((p, score))
        # prune top-k
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        keep = [p for p, s in ranked[:args.prune_k]]
        pruned[l] = keep

    # write CSV
    df_out = pd.DataFrame(rows).sort_values(['label','score'], ascending=[True, False])
    df_out.to_csv(CSV_OUT, index=False)
    print('Wrote', CSV_OUT)

    # write pruned json
    with open(PRUNED_PROMPTS, 'w') as f:
        json.dump(pruned, f, indent=2)
    print('Wrote', PRUNED_PROMPTS)

    # Now run fusion on full test set with pruned prompts using existing fusion script
    print('Running fusion on full test set with pruned prompts...')
    # call fusion script via subprocess
    import subprocess
    fusion_cmd = [
        '/zfs/ai4good/student/hgupta/mushroom_env/bin/python3',
        'fusion_clip_resnet.py',
        '--ckpt', './logs/mushroom/train/runs/2025-10-22_11-03-32/checkpoints/last.ckpt',
        '--prompts', PRUNED_PROMPTS,
        '--clip_temp', '0.05',
        '--val_n', '200',
        '--test_n', str(1000000),  # large number to force full test
        '--out', 'fusion_results_full_pruned.json'
    ]
    subprocess.check_call(fusion_cmd)
    print('Fusion complete. Results in fusion_results_full_pruned.json')


if __name__ == '__main__':
    main()
