"""Run CLIP+ResNet alpha sweep on held-out images produced by train_resnet_subset.py.

Loads `experiment_results/splits/held_half.csv`, splits it into val/test (configurable sizes),
computes CLIP prompt-pooled probs and ResNet probs using provided checkpoint, sweeps alpha,
saves JSON and plot under `experiment_results/`.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

import fusion_clip_resnet as fcr


ROOT = Path(__file__).resolve().parent
ER = ROOT / 'experiment_results'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--val_n', type=int, default=5000)
    parser.add_argument('--test_n', type=int, default=15614)
    parser.add_argument('--prompts', type=str, default=fcr.DELTA_PROMPTS)
    parser.add_argument('--clip_temp', type=float, default=fcr.CLIP_TEMP)
    parser.add_argument('--out', type=str, default=str(ER / 'fusion_resnet_half_alpha_sweep.json'))
    args = parser.parse_args()

    held_csv = ER / 'splits' / 'held_half.csv'
    if not held_csv.exists():
        raise FileNotFoundError(f'Held split not found: {held_csv}')

    df = pd.read_csv(held_csv).sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = min(args.val_n, len(df)//10)
    n_test = min(args.test_n, len(df)-n_val)
    df_val = df.iloc[:n_val].reset_index(drop=True)
    df_test = df.iloc[n_val:n_val+n_test].reset_index(drop=True)

    # Prefer labels saved in checkpoint metadata if available (keeps ordering consistent with trained model)
    meta_json = ER / 'ckpt_resnet_half_meta.json'
    if meta_json.exists():
        try:
            import json as _json
            meta = _json.load(open(meta_json))
            labels = meta.get('labels', None)
            if labels is None:
                labels = fcr.load_labels([df_val, df_test])
        except Exception:
            labels = fcr.load_labels([df_val, df_test])
    else:
        labels = fcr.load_labels([df_val, df_test])
    print(f'Using held split: val={len(df_val)}, test={len(df_test)}, classes={len(labels)}')

    print('Computing CLIP probs (val/test)')
    clip_val = fcr.compute_clip_probs(df_val, labels, args.prompts, temp=args.clip_temp)
    clip_test = fcr.compute_clip_probs(df_test, labels, args.prompts, temp=args.clip_temp)

    print('Computing ResNet probs (val/test)')
    res_val = fcr.compute_resnet_probs(df_val, labels, args.ckpt)
    res_test = fcr.compute_resnet_probs(df_test, labels, args.ckpt)

    y_val = df_val['label'].tolist()
    y_test = df_test['label'].tolist()

    results = {'val': {}, 'test': {}, 'labels': labels, 'val_size': len(df_val), 'test_size': len(df_test)}
    best_alpha = None
    best_top1 = -1
    for a in fcr.ALPHAS:
        fused_val = a * res_val + (1 - a) * clip_val
        m = fcr.evaluate_metrics(y_val, fused_val, labels)
        results['val'][str(a)] = m
        if m['top1_acc'] > best_top1:
            best_top1 = m['top1_acc']
            best_alpha = a

    fused_test = best_alpha * res_test + (1 - best_alpha) * clip_test
    test_metrics = fcr.evaluate_metrics(y_test, fused_test, labels)
    results['test'] = {'alpha': best_alpha, **test_metrics}

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved results to', args.out)

    # plot
    al = [float(k) for k in results['val'].keys()]
    top1 = [results['val'][k]['top1_acc'] for k in results['val'].keys()]
    top5 = [results['val'][k]['top5_acc'] for k in results['val'].keys()]
    bal = [results['val'][k]['balanced_acc'] for k in results['val'].keys()]
    f1 = [results['val'][k]['macro_f1'] for k in results['val'].keys()]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(al, top1, 'o-', label='Top-1')
    ax.plot(al, top5, 's-', label='Top-5')
    ax.plot(al, bal, '^-', label='Balanced')
    ax.plot(al, f1, 'd-', label='Macro-F1')
    ax.axvline(best_alpha, color='red', linestyle='--', label=f'best alpha={best_alpha}')
    ax.set_xlabel('alpha (weight on ResNet)')
    ax.set_ylabel('metric')
    ax.set_title('Alpha sweep on held-out split')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    p = ER / 'fusion_resnet_half_alpha_sweep.png'
    fig.savefig(p, dpi=200)
    print('Saved plot to', p)


if __name__ == '__main__':
    main()
