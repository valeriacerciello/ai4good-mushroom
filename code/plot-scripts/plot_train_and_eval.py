"""Plot training curves and CLIP vs ResNet complement evaluation.

Produces:
- experiment_results/figs/train_subset_curve.png
- experiment_results/figs/train84_complement_comparison.png

Reads:
- experiment_results/train_subset.log (for epoch-by-epoch metrics)
- experiment_results/figs/train84_complement_eval.json
"""
from pathlib import Path
import re
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ER = ROOT / 'experiment_results'
FIGS = ER / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)


def parse_train_log(log_path):
    # Example line: "Epoch 1/10  train_loss=3.3461 train_acc=0.1853  val_acc=0.2781  time=347.0s"
    epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\s*train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+val_acc=([0-9.]+)")
    epochs = []
    train_loss = []
    train_acc = []
    val_acc = []
    if not log_path.exists():
        return None
    with open(log_path, 'r') as f:
        for ln in f:
            m = epoch_re.search(ln)
            if m:
                e = int(m.group(1))
                epochs.append(e)
                train_loss.append(float(m.group(3)))
                train_acc.append(float(m.group(4)))
                val_acc.append(float(m.group(5)))
    if not epochs:
        return None
    return {'epochs': epochs, 'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc}


def plot_training_curve(data, out_png):
    if data is None:
        print('No training log found or no epoch lines parsed')
        return
    epochs = data['epochs']
    tr_loss = data['train_loss']
    tr_acc = data['train_acc']
    val_acc = data['val_acc']

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, tr_loss, 'o-', color='tab:blue', label='train_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(epochs, tr_acc, 's--', color='tab:green', label='train_acc')
    ax2.plot(epochs, val_acc, 'd--', color='tab:red', label='val_acc')
    ax2.set_ylabel('Accuracy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower right')
    plt.title('Training loss and accuracy')
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close()
    print('Saved training curve to', out_png)


def plot_complement_comparison(eval_json, out_png):
    if not eval_json.exists():
        print('Complement eval JSON not found:', eval_json)
        return
    d = json.load(open(eval_json))
    metrics = ['top1_acc', 'top5_acc', 'balanced_acc', 'macro_f1']
    clip_vals = [d['clip'][m] for m in metrics]
    res_vals = [d['resnet'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_clip = ax.bar(x - width/2, clip_vals, width, label='CLIP')
    bars_res = ax.bar(x + width/2, res_vals, width, label='ResNet')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, max(max(clip_vals), max(res_vals)) * 1.2)
    ax.set_ylabel('Score')
    ax.set_title('CLIP vs ResNet on unseen complement labels')
    # labels
    for bar in bars_clip:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    for bar in bars_res:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close()
    print('Saved complement comparison to', out_png)


def main():
    logp = ER / 'train_subset.log'
    train_data = parse_train_log(logp)
    plot_training_curve(train_data, FIGS / 'train_subset_curve.png')
    plot_complement_comparison(FIGS / 'train84_complement_eval.json', FIGS / 'train84_complement_comparison.png')


if __name__ == '__main__':
    main()
