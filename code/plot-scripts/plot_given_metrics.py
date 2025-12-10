"""Plot a grouped bar chart for provided CLIP vs ResNet metrics.

Saves: experiment_results/figs/given_clip_resnet_comparison.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / 'experiment_results' / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)

def main():
    metrics = ['top1_acc', 'top5_acc', 'balanced_acc', 'macro_f1']
    clip_vals = [0.18979, 0.44546, 0.18979, 0.14317]
    res_vals = [0.01176, 0.05727, 0.01176, 0.00027]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,5))
    bars_clip = ax.bar(x - width/2, clip_vals, width, label='CLIP (zero-shot)', color='#1f77b4')
    bars_res = ax.bar(x + width/2, res_vals, width, label='ResNet (trained on 84)', color='#ff7f0e')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, max(max(clip_vals), max(res_vals)) * 1.3)
    ax.set_ylabel('Score')
    ax.set_title('CLIP vs ResNet on unseen 85 classes')
    ax.legend()

    # Annotate
    for bar in bars_clip:
        h = bar.get_height()
        ax.annotate(f"{h:.5f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    for bar in bars_res:
        h = bar.get_height()
        ax.annotate(f"{h:.5f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = FIGS / 'given_clip_resnet_comparison.png'
    fig.savefig(out, dpi=200)
    plt.close()
    print('Saved plot to', out)

if __name__ == '__main__':
    main()
