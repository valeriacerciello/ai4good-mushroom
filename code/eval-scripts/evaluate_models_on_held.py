"""Evaluate ResNet-only and CLIP-only on held_half split and plot per-class accuracies.

Saves:
- experiment_results/figs/resnet_per_class_acc.csv
- experiment_results/figs/clip_per_class_acc.csv
- experiment_results/figs/resnet_per_class_acc.png
- experiment_results/figs/clip_per_class_acc.png
- experiment_results/figs/overall_metrics.json

Uses existing fusion utilities in `fusion_clip_resnet.py` for loading models and computing probs.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fusion_clip_resnet as fcr

ROOT = Path(__file__).resolve().parent
ER = ROOT / 'experiment_results'
FIGS = ER / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)


def per_class_accuracy(y_true, top1_idx, labels):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[y] for y in y_true])
    per_acc = {}
    for i, lab in enumerate(labels):
        mask = (y_idx == i)
        if mask.sum() == 0:
            per_acc[lab] = None
            continue
        per_acc[lab] = float((top1_idx[mask] == i).mean())
    return per_acc


def evaluate(probs, y_true, labels):
    # probs: N x C
    top1_idx = probs.argmax(axis=1)
    top5_idx = np.argsort(-probs, axis=1)[:, :5]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[y] for y in y_true])
    top1 = float((top1_idx == y_idx).mean())
    top5 = float(np.mean([y_idx[i] in top5_idx[i] for i in range(len(y_idx))]))
    # balanced and macro f1 via fusion_clip_resnet utilities
    metrics = fcr.evaluate_metrics(y_true, probs, labels)
    metrics['top1_acc'] = top1
    metrics['top5_acc'] = top5
    return metrics, top1_idx


def plot_per_class(per_acc, out_png, title, topk=1):
    # per_acc: dict label->acc (None allowed)
    items = [(k, v if v is not None else 0.0) for k, v in per_acc.items()]
    items_sorted = sorted(items, key=lambda x: x[1])
    labels = [i[0] for i in items_sorted]
    vals = [i[1] for i in items_sorted]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=90, fontsize=6)
    plt.ylim(0, 1)
    plt.ylabel('Top-1 Accuracy')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main(ckpt_path: str = None, prompts: str = fcr.DELTA_PROMPTS, clip_temp: float = fcr.CLIP_TEMP):
    held_csv = ER / 'splits' / 'held_half.csv'
    if not held_csv.exists():
        raise FileNotFoundError(f'Held split not found: {held_csv}')
    df = pd.read_csv(held_csv).reset_index(drop=True)

    # Use labels from ckpt meta if available for consistent ordering
    meta_json = ER / 'ckpt_resnet_half_meta.json'
    if meta_json.exists():
        try:
            meta = json.load(open(meta_json))
            labels = meta.get('labels', None)
            if labels is None:
                labels = fcr.load_labels([df])
        except Exception:
            labels = fcr.load_labels([df])
    else:
        labels = fcr.load_labels([df])

    print(f'Evaluating on held split: {len(df)} samples, {len(labels)} classes')

    # Compute CLIP probs
    print('Computing CLIP probabilities...')
    clip_probs = fcr.compute_clip_probs(df, labels, prompts, temp=clip_temp)
    clip_metrics, clip_top1_idx = evaluate(clip_probs, df['label'].tolist(), labels)
    print('CLIP overall metrics:', clip_metrics)

    # Compute ResNet probs
    if ckpt_path is None:
        ckpt_path = str(ER / 'ckpt_resnet_half.fixed.pth')
    print('Computing ResNet probabilities from', ckpt_path)
    res_probs = fcr.compute_resnet_probs(df, labels, ckpt_path)
    res_metrics, res_top1_idx = evaluate(res_probs, df['label'].tolist(), labels)
    print('ResNet overall metrics:', res_metrics)

    # Per-class accuracy
    res_per = per_class_accuracy(df['label'].tolist(), res_top1_idx, labels)
    clip_per = per_class_accuracy(df['label'].tolist(), clip_top1_idx, labels)

    # Save CSVs
    res_df = pd.DataFrame([{'label': k, 'top1_acc': v} for k, v in res_per.items()])
    clip_df = pd.DataFrame([{'label': k, 'top1_acc': v} for k, v in clip_per.items()])
    res_df.to_csv(FIGS / 'resnet_per_class_acc.csv', index=False)
    clip_df.to_csv(FIGS / 'clip_per_class_acc.csv', index=False)

    # Plot per-class sorted
    plot_per_class(res_per, FIGS / 'resnet_per_class_acc.png', 'ResNet per-class Top-1 Accuracy (held split)')
    plot_per_class(clip_per, FIGS / 'clip_per_class_acc.png', 'CLIP per-class Top-1 Accuracy (held split)')

    # Save overall metrics JSON
    overall = {'resnet': res_metrics, 'clip': clip_metrics}
    json.dump(overall, open(FIGS / 'overall_metrics.json', 'w'), indent=2)

    print('Saved per-class CSVs and plots to', FIGS)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=str(ER / 'ckpt_resnet_half.fixed.pth'))
    p.add_argument('--prompts', type=str, default=fcr.DELTA_PROMPTS)
    p.add_argument('--clip_temp', type=float, default=fcr.CLIP_TEMP)
    args = p.parse_args()
    main(ckpt_path=args.ckpt, prompts=args.prompts, clip_temp=args.clip_temp)
