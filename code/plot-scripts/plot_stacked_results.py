import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
ER = ROOT / "experiment_results"
FIGS = ER / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

shots = [0, 1, 2, 5, 10]

def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def main():
    rows = []

    # zero-shot CLIP baseline
    proto_zero = load_json(ER / 'prototype_knn_results.json')

    for s in shots:
        stacked_file = ER / f'stacked_fusion_shots{s}.json'
        proto_file = ER / f'prototype_knn_fewshot_shots{s}.json'

        stacked = load_json(stacked_file)
        # LR stacked metrics
        lr = stacked.get('stacked', {})

        # ResNet-only metrics: when best_alpha is 1.0 many runs used ResNet-only;
        # but stacked files' best_alpha.metrics with alpha=1.0 contains ResNet-only test numbers
        best_alpha = stacked.get('best_alpha', {})
        resnet_metrics = {}
        if best_alpha and best_alpha.get('alpha', None) == 1.0:
            resnet_metrics = best_alpha.get('metrics', {})
        else:
            # fallback: if not present, copy lr (best effort)
            resnet_metrics = best_alpha.get('metrics', {})

        # CLIP few-shot best-alpha metrics
        if s == 0:
            clip_metrics = proto_zero.get('test', {})
            # proto_zero stores test as simple numbers; ensure keys match
            # expected keys: top1_acc, top5_acc, balanced_acc, macro_f1
            clip_top1 = clip_metrics.get('top1_acc', None)
            clip_top5 = clip_metrics.get('top5_acc', None)
            clip_bal = clip_metrics.get('balanced_acc', None)
            clip_f1 = clip_metrics.get('macro_f1', None)
            clip_metrics = {
                'top1_acc': clip_top1,
                'top5_acc': clip_top5,
                'balanced_acc': clip_bal,
                'macro_f1': clip_f1,
            }
        else:
            proto = load_json(proto_file)
            clip_metrics = proto.get('best_alpha', {}).get('metrics', {})

        rows.append({
            'shot': s,
            'resnet_top1': resnet_metrics.get('top1_acc'),
            'resnet_top5': resnet_metrics.get('top5_acc'),
            'resnet_balanced': resnet_metrics.get('balanced_acc'),
            'resnet_f1': resnet_metrics.get('macro_f1'),
            'clip_top1': clip_metrics.get('top1_acc'),
            'clip_top5': clip_metrics.get('top5_acc'),
            'clip_balanced': clip_metrics.get('balanced_acc'),
            'clip_f1': clip_metrics.get('macro_f1'),
            'lr_top1': lr.get('top1_acc'),
            'lr_top5': lr.get('top5_acc'),
            'lr_balanced': lr.get('balanced_acc'),
            'lr_f1': lr.get('macro_f1'),
        })

    df = pd.DataFrame(rows).sort_values('shot')
    csv_out = FIGS / 'stacked_shots_metrics.csv'
    df.to_csv(csv_out, index=False)

    # Plot grouped bars for Top1 and Top5
    def plot_metric(metric, ylabel, outname):
        x = [str(s) for s in df['shot']]
        width = 0.2
        fig, ax = plt.subplots(figsize=(9,5))
        ax.bar([i - width for i in range(len(x))], df[f'clip_{metric}'], width, label='CLIP (best-alpha)')
        ax.bar([i for i in range(len(x))], df[f'resnet_{metric}'], width, label='ResNet-only')
        ax.bar([i + width for i in range(len(x))], df[f'lr_{metric}'], width, label='LR-stacked')
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x)
        ax.set_xlabel('shots')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} across shots (CLIP vs ResNet vs LR-stacked)')
        ax.legend()
        for i, v in enumerate(df[f'clip_{metric}']):
            ax.text(i - width, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
        for i, v in enumerate(df[f'resnet_{metric}']):
            ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
        for i, v in enumerate(df[f'lr_{metric}']):
            ax.text(i + width, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
        plt.tight_layout()
        fig.savefig(FIGS / outname)
        plt.close(fig)

    plot_metric('top1', 'Top-1 Accuracy', 'stacked_comparison_top1.png')
    plot_metric('top5', 'Top-5 Accuracy', 'stacked_comparison_top5.png')

    # Also create a combined line plot (shots vs Top1) for quick view
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(df['shot'], df['clip_top1'], marker='o', label='CLIP (best-alpha)')
    ax.plot(df['shot'], df['resnet_top1'], marker='o', label='ResNet-only')
    ax.plot(df['shot'], df['lr_top1'], marker='o', label='LR-stacked')
    ax.set_xlabel('shots')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title('Top-1 vs shots')
    ax.legend()
    for x_, y_ in zip(df['shot'], df['clip_top1']):
        ax.text(x_, y_ + 0.003, f'{y_:.3f}', fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGS / 'shots_vs_top1_lines.png')
    plt.close(fig)

    # New: grouped bar comparing LR-stacked vs alpha-fusion (few-shot) across shots
    def plot_lr_vs_alpha(shots_list, metric, ylabel, outname):
        sub = df[df['shot'].isin(shots_list)].set_index('shot')
        labels = [str(s) for s in sub.index]
        x = range(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8,5))
        alpha_vals = sub[f'clip_{metric}'].values
        lr_vals = sub[f'lr_{metric}'].values
        ax.bar([i - width/2 for i in x], lr_vals, width, label='LR-stacked', color='#1f77b4')
        ax.bar([i + width/2 for i in x], alpha_vals, width, label='Alpha-fusion (CLIP few-shot)', color='#ff7f0e')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('shots')
        ax.set_ylabel(ylabel)
        ax.set_title(f'LR-stacked vs Alpha-fusion across shots ({ylabel})')
        ax.legend()
        for i, v in enumerate(lr_vals):
            ax.text(i - width/2, v + 0.003, f'{v:.3f}', ha='center', fontsize=9)
        for i, v in enumerate(alpha_vals):
            ax.text(i + width/2, v + 0.003, f'{v:.3f}', ha='center', fontsize=9)
        plt.tight_layout()
        fig.savefig(FIGS / outname)
        plt.close(fig)

    fewshot_shots = [1,2,5,10]
    plot_lr_vs_alpha(fewshot_shots, 'top1', 'Top-1 Accuracy', 'lr_vs_alpha_fewshot_top1.png')
    plot_lr_vs_alpha(fewshot_shots, 'top5', 'Top-5 Accuracy', 'lr_vs_alpha_fewshot_top5.png')

    # New: LR-only multi-metric grouped bar chart (Top1, Top5, Balanced, Macro-F1) across shots
    def plot_lr_multi_metric(shots_list, outname):
        sub = df[df['shot'].isin(shots_list)].set_index('shot')
        metrics = ['lr_top1', 'lr_top5', 'lr_balanced', 'lr_f1']
        metric_labels = ['Top-1', 'Top-5', 'Balanced Acc', 'Macro-F1']
        n_metrics = len(metrics)
        x = range(len(sub.index))
        width = 0.18
        fig, ax = plt.subplots(figsize=(12,6))
        colors = ['#2E86AB', '#7BC043', '#E4572E', '#8F59A7']
        for i, m in enumerate(metrics):
            vals = sub[m].values
            positions = [p + (i - (n_metrics-1)/2) * width for p in x]
            ax.bar(positions, vals, width, label=metric_labels[i], color=colors[i], alpha=0.95, edgecolor='k', linewidth=0.3)
            for pos, v in zip(positions, vals):
                if v is None:
                    continue
                ax.text(pos, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sub.index])
        ax.set_xlabel('shots')
        ax.set_ylabel('score')
        ax.set_ylim(0, 1.0)
        ax.set_title('Few-shot: Metrics by shots (LR-stacked)')
        ax.legend(loc='lower center')
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        fig.savefig(FIGS / outname)
        plt.close(fig)

    plot_lr_multi_metric(fewshot_shots, 'lr_metrics_by_shots.png')

    print('Wrote:', csv_out)
    print('Wrote plots to', FIGS)


if __name__ == '__main__':
    main()
