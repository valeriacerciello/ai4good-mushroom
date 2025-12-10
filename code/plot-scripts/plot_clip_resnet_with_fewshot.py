"""Combine the given CLIP vs ResNet plot with few-shot ResNet results.

Reads:
- experiment_results/figs/train84_complement_eval.json (contains CLIP and ResNet zero-shot on complement)
- experiment_results/figs/resnet_fewshot_results.json (per-shot ResNet few-shot results)

Writes:
- experiment_results/figs/given_clip_resnet_with_fewshot.png
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / 'experiment_results' / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)

def load_json(p):
    if not p.exists():
        return None
    return json.load(open(p))

def main():
    eval_json = FIGS / 'train84_complement_eval.json'
    fewshot_json = FIGS / 'resnet_fewshot_results.json'
    evald = load_json(eval_json)
    few = load_json(fewshot_json)
    if evald is None:
        print('Missing', eval_json)
        return
    if few is None:
        print('Missing', fewshot_json)
        return

    metrics = ['top1_acc', 'top5_acc', 'balanced_acc', 'macro_f1']
    # baseline values
    clip_vals = [evald['clip'][m] for m in metrics]
    resnet_zero_vals = [evald['resnet'][m] for m in metrics]

    # few-shot shots sorted
    shots = sorted([int(k) for k in few.keys()])
    few_vals_by_shot = {s: [few[str(s)][m] for m in metrics] for s in shots}

    # For each metric, we will plot grouped bars: CLIP, ResNet-zero, ResNet-1, ResNet-2, ResNet-5, ResNet-10
    labels = ['CLIP (ZS)', 'ResNet (ZS)'] + [f'ResNet {s}shot' for s in shots]
    n_groups = len(metrics)
    n_bars = len(labels)
    x = np.arange(n_groups)
    total_width = 0.8
    width = total_width / n_bars

    fig, ax = plt.subplots(figsize=(12,6))
    offsets = np.linspace(-total_width/2 + width/2, total_width/2 - width/2, n_bars)

    # prepare values matrix: shape (n_bars, n_metrics)
    values = []
    values.append(clip_vals)
    values.append(resnet_zero_vals)
    for s in shots:
        values.append(few_vals_by_shot[s])

    colors = plt.get_cmap('tab10').colors
    for i, (label, off) in enumerate(zip(labels, offsets)):
        vals = values[i]
        bars = ax.bar(x + off, vals, width, label=label, color=colors[i % len(colors)])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title('CLIP (zero-shot) vs ResNet (zero-shot) and ResNet few-shot on 85 unseen classes')
    ax.legend(ncol=2)
    ax.set_ylim(0, max(max(clip_vals), max(resnet_zero_vals), max(max(v) for v in values)) * 1.2)
    plt.tight_layout()
    out = FIGS / 'given_clip_resnet_with_fewshot.png'
    fig.savefig(out, dpi=200)
    plt.close()
    print('Saved', out)

if __name__ == '__main__':
    main()
