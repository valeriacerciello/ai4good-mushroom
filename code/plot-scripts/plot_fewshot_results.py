"""Plot few-shot ResNet results saved in experiment_results/figs/resnet_fewshot_results.json

Produces:
- experiment_results/figs/resnet_fewshot_bar_top1_top5.png
- experiment_results/figs/resnet_fewshot_bar_all_metrics.png

If the JSON is not present the script will print a message and exit.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / 'experiment_results' / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)
RES_JSON = FIGS / 'resnet_fewshot_results.json'


def load_results():
    if not RES_JSON.exists():
        print('Few-shot results JSON not found at', RES_JSON)
        return None
    return json.load(open(RES_JSON))


def plot_top1_top5(results, out_png):
    shots = sorted([int(k) for k in results.keys()])
    top1 = [results[str(s)]['top1_acc'] for s in shots]
    top5 = [results[str(s)]['top5_acc'] for s in shots]

    x = np.arange(len(shots))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,5))
    bars1 = ax.bar(x - width/2, top1, width, label='Top-1')
    bars5 = ax.bar(x + width/2, top5, width, label='Top-5')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in shots])
    ax.set_xlabel('Shots')
    ax.set_ylabel('Accuracy')
    ax.set_title('ResNet few-shot: Top-1 and Top-5 vs Shots')
    ax.legend()
    for bar in list(bars1) + list(bars5):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close()
    print('Saved', out_png)


def plot_all_metrics(results, out_png):
    shots = sorted([int(k) for k in results.keys()])
    metrics = ['top1_acc', 'top5_acc', 'balanced_acc', 'macro_f1']
    data = {m: [results[str(s)][m] for s in shots] for m in metrics}

    x = np.arange(len(shots))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10,5))
    for i, m in enumerate(metrics):
        bars = ax.bar(x + (i-1.5)*width, data[m], width, label=m)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in shots])
    ax.set_xlabel('Shots')
    ax.set_ylabel('Score')
    ax.set_title('ResNet few-shot metrics vs Shots')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close()
    print('Saved', out_png)


def main():
    results = load_results()
    if results is None:
        return
    plot_top1_top5(results, FIGS / 'resnet_fewshot_bar_top1_top5.png')
    plot_all_metrics(results, FIGS / 'resnet_fewshot_bar_all_metrics.png')


if __name__ == '__main__':
    main()
