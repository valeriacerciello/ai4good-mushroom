"""Run fusion restricted to labels present in sampled val/test splits.

This imports helper functions from `fusion_clip_resnet.py` and performs an alpha
sweep over the sampled val split (random_state=42, default sizes). Results are
written to `fusion_restricted_sample.json` and a plot `fusion_alpha_sweep_restricted.png`.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from fusion_clip_resnet import (
    load_split_df,
    compute_clip_probs,
    compute_resnet_probs,
    evaluate_metrics,
    ALPHAS,
)

DATA_ROOT = "/zfs/ai4good/datasets/mushroom"
DELTA_PROMPTS = "/zfs/ai4good/student/hgupta/delta_prompts.json"
CKPT = "/zfs/ai4good/student/hgupta/logs/mushroom/train/runs/2025-10-21_22-27-28/checkpoints/epoch_000.ckpt"
CLIP_TEMP = 0.05

def main():
    # load sampled val/test (same sampling logic as other scripts)
    df_val = load_split_df('val', limit=200)
    df_test = load_split_df('test', limit=200)

    # Restrict labels to those present in the sampled splits
    labels_subset = sorted(list(set(df_val['label'].tolist() + df_test['label'].tolist())))
    print(f"Restricted label set size: {len(labels_subset)}")

    print('Computing CLIP probabilities (val restricted)...')
    clip_val = compute_clip_probs(df_val, labels_subset, DELTA_PROMPTS, temp=CLIP_TEMP)
    print('Computing CLIP probabilities (test restricted)...')
    clip_test = compute_clip_probs(df_test, labels_subset, DELTA_PROMPTS, temp=CLIP_TEMP)

    if os.path.exists(CKPT):
        print('Computing ResNet probabilities (val/test restricted)...')
        res_val = compute_resnet_probs(df_val, labels_subset, CKPT)
        res_test = compute_resnet_probs(df_test, labels_subset, CKPT)
    else:
        print('ResNet checkpoint not found; using uniform probs fallback')
        res_val = np.full_like(clip_val, 1.0 / len(labels_subset))
        res_test = np.full_like(clip_test, 1.0 / len(labels_subset))

    y_val = df_val['label'].tolist()
    y_test = df_test['label'].tolist()

    results = {'val': {}, 'test': {}, 'labels': labels_subset}
    best_alpha = None
    best_top1 = -1.0
    best_f1 = -1.0
    for a in ALPHAS:
        fused_val = a * res_val + (1 - a) * clip_val
        m = evaluate_metrics(y_val, fused_val, labels_subset)
        results['val'][str(a)] = m
        if (m['top1_acc'] > best_top1) or (np.isclose(m['top1_acc'], best_top1) and m['macro_f1'] > best_f1):
            best_top1 = m['top1_acc']
            best_f1 = m['macro_f1']
            best_alpha = a

    fused_test = best_alpha * res_test + (1 - best_alpha) * clip_test
    test_metrics = evaluate_metrics(y_test, fused_test, labels_subset)
    results['test'] = {'alpha': best_alpha, **test_metrics}

    out = 'fusion_restricted_sample.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote {out}')

    # quick plot
    try:
        al = [float(k) for k in results['val'].keys()]
        top1 = [results['val'][k]['top1_acc'] for k in results['val'].keys()]
        top5 = [results['val'][k]['top5_acc'] for k in results['val'].keys()]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(al, top1, 'o-', label='Top-1')
        ax.plot(al, top5, 's-', label='Top-5')
        ax.axvline(best_alpha, color='red', linestyle='--', alpha=0.5, label=f'Best alpha={best_alpha}')
        ax.set_xlabel('alpha (weight on ResNet)')
        ax.set_ylabel('Score')
        ax.set_title('Restricted fusion alpha sweep (sample labels)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig('fusion_alpha_sweep_restricted.png', dpi=300, bbox_inches='tight')
        print('Saved fusion_alpha_sweep_restricted.png')
    except Exception as e:
        print('Plotting failed:', e)

if __name__ == '__main__':
    main()
