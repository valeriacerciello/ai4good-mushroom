"""Quick comparison plot: CLIP vs ResNet on a sampled subset of the held split.
Saves results to experiment_results/figs/clip_vs_resnet_quick.png and JSON.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fusion_clip_resnet as fcr

ROOT = Path(__file__).resolve().parent
ER = ROOT / 'experiment_results'
FIGS = ER / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)


def main(n_samples: int = 1000, seed: int = 42, ckpt: str = None, prompts: str = fcr.DELTA_PROMPTS, clip_temp: float = fcr.CLIP_TEMP):
    held_csv = ER / 'splits' / 'held_half.csv'
    if not held_csv.exists():
        raise FileNotFoundError(f'Held split not found: {held_csv}')
    df = pd.read_csv(held_csv).sample(frac=1, random_state=seed).reset_index(drop=True)
    n = min(n_samples, len(df))
    df_sub = df.iloc[:n].reset_index(drop=True)

    # labels
    meta_json = ER / 'ckpt_resnet_half_meta.json'
    if meta_json.exists():
        try:
            meta = json.load(open(meta_json))
            labels = meta.get('labels', None)
            if labels is None:
                labels = fcr.load_labels([df_sub])
        except Exception:
            labels = fcr.load_labels([df_sub])
    else:
        labels = fcr.load_labels([df_sub])

    print(f'Using subset of {len(df_sub)} held images, labels={len(labels)}')

    print('Computing CLIP probs...')
    clip_probs = fcr.compute_clip_probs(df_sub, labels, prompts, temp=clip_temp)
    print('Computing ResNet probs...')
    if ckpt is None:
        ckpt = str(ER / 'ckpt_resnet_half.fixed.pth')
    if not Path(ckpt).exists():
        print('Checkpoint not found, using uniform probs')
        res_probs = np.full_like(clip_probs, 1.0/len(labels))
    else:
        res_probs = fcr.compute_resnet_probs(df_sub, labels, ckpt)

    y = df_sub['label'].tolist()
    clip_metrics = fcr.evaluate_metrics(y, clip_probs, labels)
    res_metrics = fcr.evaluate_metrics(y, res_probs, labels)

    summary = {'n_samples': len(df_sub), 'clip': clip_metrics, 'resnet': res_metrics}
    out_json = FIGS / 'clip_vs_resnet_quick_metrics.json'
    json.dump(summary, open(out_json, 'w'), indent=2)
    print('Saved metrics to', out_json)

    # Plot comparison
    metrics = ['top1_acc', 'top5_acc', 'balanced_acc', 'macro_f1']
    clip_vals = [clip_metrics[m] for m in metrics]
    res_vals = [res_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x - width/2, clip_vals, width, label='CLIP')
    ax.bar(x + width/2, res_vals, width, label='ResNet')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title(f'CLIP vs ResNet on held subset (n={len(df_sub)})')
    ax.legend()
    plt.tight_layout()
    out_png = FIGS / 'clip_vs_resnet_quick.png'
    fig.savefig(out_png, dpi=200)
    print('Saved plot to', out_png)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--prompts', type=str, default=fcr.DELTA_PROMPTS)
    p.add_argument('--clip_temp', type=float, default=fcr.CLIP_TEMP)
    args = p.parse_args()
    main(n_samples=args.n, seed=args.seed, ckpt=args.ckpt, prompts=args.prompts, clip_temp=args.clip_temp)
