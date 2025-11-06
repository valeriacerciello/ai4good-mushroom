import json
import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def plot_alpha_sweep(val_dict, out_path, title='Alpha sweep (val)'):
    # val_dict: {alpha_str: {metrics}}
    alphas = sorted([float(k) for k in val_dict.keys()])
    top1 = [val_dict[str(a)]['top1_acc'] for a in alphas]
    top5 = [val_dict[str(a)]['top5_acc'] for a in alphas]
    bal = [val_dict[str(a)]['balanced_acc'] for a in alphas]
    f1 = [val_dict[str(a)]['macro_f1'] for a in alphas]

    plt.figure(figsize=(8,5))
    plt.plot(alphas, top1, 'o-', label='Top-1')
    plt.plot(alphas, top5, 's-', label='Top-5')
    plt.plot(alphas, bal, '^-', label='Balanced')
    plt.plot(alphas, f1, 'd-', label='Macro-F1')
    plt.xlabel('alpha (weight on ResNet)')
    plt.ylabel('score')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    root = Path('experiment_results')
    figdir = root / 'figs'
    ensure_dir(figdir)

    # collect few-shot files
    shots_files = sorted(root.glob('prototype_knn_fewshot_shots*.json'))
    shots_vals = []
    shots_tests = []
    shots_list = []
    # collect CLIP-only (alpha=0.0) val metrics per shot
    cliponly_vals = []

    for p in shots_files:
        data = load_json(p)
        shots = data.get('shots', None)
        shots_list.append(shots)
        # alpha sweep plot for val
        if 'val' in data and isinstance(data['val'], dict) and data['val']:
            out = figdir / f'alpha_sweep_shots{shots}.png'
            plot_alpha_sweep(data['val'], out, title=f'Alpha sweep (val) — shots={shots}')
        # capture CLIP-only (alpha=0.0) val metrics if present
        clip_val = None
        try:
            if 'val' in data and '0.0' in data['val']:
                clip_val = data['val']['0.0']
        except Exception:
            clip_val = None
        cliponly_vals.append((shots, clip_val))
        # record best_alpha test metrics if present
        best = data.get('best_alpha', {})
        if isinstance(best, dict) and 'metrics' in best:
            shots_tests.append((shots, best['metrics']))
        else:
            shots_tests.append((shots, {}))

    # also include baseline fusion and prototype files
    baseline = root / 'fusion_results_delta_full.json'
    proto = root / 'prototype_knn_results.json'
    calib = root / 'calibration_tta_results.json'

    if baseline.exists():
        b = load_json(baseline)
        # baseline contains 'val' dict per alpha
        if 'val' in b and isinstance(b['val'], dict):
            plot_alpha_sweep(b['val'], figdir / 'alpha_sweep_baseline.png', title='Alpha sweep (val) — baseline')

    # Plot shots vs test Top-1/Top-5
    if shots_tests:
        shots_tests = sorted(shots_tests, key=lambda x: (math.inf if x[0] is None else x[0]))
        s = [st[0] for st in shots_tests]
        top1 = [st[1].get('top1_acc', np.nan) for st in shots_tests]
        top5 = [st[1].get('top5_acc', np.nan) for st in shots_tests]

        plt.figure(figsize=(6,4))
        plt.plot(s, top1, 'o-', label='Test Top-1')
        plt.plot(s, top5, 's-', label='Test Top-5')
        plt.xlabel('shots')
        plt.ylabel('score')
        plt.title('Few-shot: Test Top-1 / Top-5 vs shots')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / 'shots_vs_test_top1_top5.png')
        plt.close()

        # Also produce a combined plot for Top-1, Top-5, Balanced Acc, Macro-F1 vs shots
        bal = [st[1].get('balanced_acc', np.nan) for st in shots_tests]
        f1 = [st[1].get('macro_f1', np.nan) for st in shots_tests]

        plt.figure(figsize=(8,5))
        plt.plot(s, top1, 'o-', label='Test Top-1')
        plt.plot(s, top5, 's-', label='Test Top-5')
        plt.plot(s, bal, '^-', label='Balanced Acc')
        plt.plot(s, f1, 'd-', label='Macro-F1')
        plt.xlabel('shots')
        plt.ylabel('score')
        plt.title('Few-shot: Test metrics vs shots')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / 'shots_vs_all_metrics.png')
        plt.close()

        # Save CSV summary
        import csv
        csv_path = figdir / 'shots_vs_metrics.csv'
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['shots', 'top1_acc', 'top5_acc', 'balanced_acc', 'macro_f1'])
            for i, shot in enumerate(s):
                writer.writerow([shot, top1[i], top5[i], bal[i], f1[i]])

    # Plot CLIP-only (kNN) val metrics across shots (use alpha=0.0 entries)
    # cliponly_vals: list of (shots, metrics_dict)
    cliponly_vals = [c for c in cliponly_vals if c[1] is not None]
    if cliponly_vals:
        cliponly_vals = sorted(cliponly_vals, key=lambda x: (math.inf if x[0] is None else x[0]))
        cs = [c[0] for c in cliponly_vals]
        c_top1 = [c[1].get('top1_acc', np.nan) for c in cliponly_vals]
        c_top5 = [c[1].get('top5_acc', np.nan) for c in cliponly_vals]
        c_bal = [c[1].get('balanced_acc', np.nan) for c in cliponly_vals]
        c_f1 = [c[1].get('macro_f1', np.nan) for c in cliponly_vals]

        plt.figure(figsize=(10,5))
        plt.plot(cs, c_top1, 'o-', label='Val Top-1')
        plt.plot(cs, c_top5, 's-', label='Val Top-5')
        plt.plot(cs, c_bal, '^-', label='Val Balanced')
        plt.plot(cs, c_f1, 'd-', label='Val Macro-F1')
        plt.xlabel('shots')
        plt.ylabel('score')
        plt.title('CLIP-only (kNN) — validation metrics vs shots (alpha=0.0)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / 'shots_vs_knn_val_metrics.png')
        plt.close()

        # grouped bar chart for CLIP-only val metrics
        try:
            x = np.arange(len(cs))
            width = 0.18
            plt.figure(figsize=(12,6))
            colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
            b1 = plt.bar(x - 1.5*width, c_top1, width, label='Top-1', color=colors[0], edgecolor='black', linewidth=0.8)
            b2 = plt.bar(x - 0.5*width, c_top5, width, label='Top-5', color=colors[1], edgecolor='black', linewidth=0.8)
            b3 = plt.bar(x + 0.5*width, c_bal, width, label='Balanced Acc', color=colors[2], edgecolor='black', linewidth=0.8)
            b4 = plt.bar(x + 1.5*width, c_f1, width, label='Macro-F1', color=colors[3], edgecolor='black', linewidth=0.8)
            plt.xlabel('shots')
            plt.ylabel('score')
            plt.title('CLIP-only (kNN) — validation metrics by shots')
            plt.xticks(x, [str(int(v)) for v in cs])
            ymax = max([v for v in (c_top1+c_top5+c_bal+c_f1) if not math.isnan(v)]) if len(cs) > 0 else 1.0
            plt.ylim(0, min(1.0, ymax + 0.08))
            plt.grid(axis='y', alpha=0.2)
            plt.legend()

            def annotate_bars(bars):
                for bar in bars:
                    h = bar.get_height()
                    label = f'{h:.3f}' if not math.isnan(h) else 'n/a'
                    plt.text(bar.get_x() + bar.get_width()/2, h + 0.01, label, ha='center', va='bottom', fontsize=9, color='black')

            annotate_bars(b1)
            annotate_bars(b2)
            annotate_bars(b3)
            annotate_bars(b4)

            plt.tight_layout()
            plt.savefig(figdir / 'shots_vs_knn_val_bar.png', dpi=300)
            plt.close()
        except Exception as e:
            print('CLIP-only bar chart generation failed:', e)

        # save CSV for CLIP-only
        csv_k = figdir / 'shots_vs_knn_val_metrics.csv'
        import csv
        with open(csv_k, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['shots', 'val_top1', 'val_top5', 'val_balanced', 'val_macro_f1'])
            for i, shot in enumerate(cs):
                writer.writerow([shot, c_top1[i], c_top5[i], c_bal[i], c_f1[i]])

        # Also create a grouped bar chart for the 4 metrics across shots
        try:
            x = np.arange(len(s))
            width = 0.18
            plt.figure(figsize=(12,6))
            # define colors
            colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
            bars1 = plt.bar(x - 1.5*width, top1, width, label='Top-1', color=colors[0], edgecolor='black', linewidth=0.8)
            bars2 = plt.bar(x - 0.5*width, top5, width, label='Top-5', color=colors[1], edgecolor='black', linewidth=0.8)
            bars3 = plt.bar(x + 0.5*width, bal, width, label='Balanced Acc', color=colors[2], edgecolor='black', linewidth=0.8)
            bars4 = plt.bar(x + 1.5*width, f1, width, label='Macro-F1', color=colors[3], edgecolor='black', linewidth=0.8)
            plt.xlabel('shots')
            plt.ylabel('score')
            plt.title('Few-shot: Metrics by shots (bar chart)')
            plt.xticks(x, [str(int(v)) for v in s])
            # set y limit slightly above max
            ymax = max([v for v in (top1+top5+bal+f1) if not math.isnan(v)]) if len(s) > 0 else 1.0
            plt.ylim(0, min(1.0, ymax + 0.08))
            plt.grid(axis='y', alpha=0.2)
            plt.legend()

            # annotate bars with numeric labels
            def annotate(bars):
                for bar in bars:
                    h = bar.get_height()
                    if math.isnan(h):
                        label = 'n/a'
                    else:
                        label = f'{h:.3f}'
                    plt.text(bar.get_x() + bar.get_width()/2, h + 0.01, label, ha='center', va='bottom', fontsize=9, color='black')

            annotate(bars1)
            annotate(bars2)
            annotate(bars3)
            annotate(bars4)

            plt.tight_layout()
            plt.savefig(figdir / 'shots_vs_all_metrics_bar.png', dpi=300)
            plt.close()
        except Exception as e:
            print('Bar chart generation failed:', e)

    # Add a small summary text file listing all images written
    with open(figdir / 'README.txt', 'w') as f:
        f.write('Figures generated from experiment_results JSONs\n')
        f.write('Files:\n')
        for p in sorted(figdir.iterdir()):
            f.write(f'- {p.name}\n')

    print('Saved figures to', figdir)


if __name__ == '__main__':
    main()
