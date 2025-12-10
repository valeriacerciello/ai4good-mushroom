"""
Create comprehensive visualization graphs for CLIP+ResNet fusion analysis.

Generates:
1. Alpha sweep plot (all 4 metrics vs fusion weight)
2. Model comparison bar chart (CLIP-only vs ResNet-only vs Best-fusion)
3. Metric heatmap showing per-alpha performance
4. Delta improvement plot (fusion gain over individual models)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load fusion results
FUSION_RESULTS = "fusion_results.json"

def load_results():
    with open(FUSION_RESULTS, 'r') as f:
        return json.load(f)

def create_alpha_sweep_plot(results):
    """Create detailed alpha sweep with all 4 metrics"""
    alphas = sorted([float(k) for k in results['val'].keys()])
    
    metrics = {
        'Top-1 Accuracy': [results['val'][str(a)]['top1_acc'] for a in alphas],
        'Top-5 Accuracy': [results['val'][str(a)]['top5_acc'] for a in alphas],
        'Balanced Accuracy': [results['val'][str(a)]['balanced_acc'] for a in alphas],
        'Macro F1-Score': [results['val'][str(a)]['macro_f1'] for a in alphas],
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CLIP+ResNet Fusion: Alpha Weight Sweep on Validation Set', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
    markers = ['o', 's', '^', 'd']
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        ax.plot(alphas, values, marker=markers[idx], linewidth=2.5, 
                markersize=8, color=colors[idx], label=metric_name, alpha=0.8)
        
        # Mark best alpha
        best_idx = np.argmax(values)
        best_alpha = alphas[best_idx]
        best_val = values[best_idx]
        ax.axvline(best_alpha, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.scatter([best_alpha], [best_val], color='red', s=150, zorder=5, 
                   marker='*', edgecolors='darkred', linewidths=1.5)
        
        ax.set_xlabel('Alpha (ResNet weight)', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}\nBest α={best_alpha:.2f}, Score={best_val:.3f}', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlim(-0.05, 1.05)
        
        # Add value labels at key points
        for i in [0, best_idx, len(alphas)-1]:
            if i < len(alphas):
                ax.annotate(f'{values[i]:.3f}', 
                           xy=(alphas[i], values[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('fusion_alpha_sweep_detailed.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: fusion_alpha_sweep_detailed.png')
    plt.close()

def create_model_comparison_chart(results):
    """Bar chart comparing CLIP-only, ResNet-only, and Best-fusion"""
    best_alpha = results['test']['alpha']
    
    # Extract metrics for each approach
    clip_only = results['val']['0.0']  # alpha=0 means CLIP only
    resnet_only = results['val']['1.0']  # alpha=1 means ResNet only
    best_fusion = results['test']
    
    metrics = ['top1_acc', 'top5_acc', 'balanced_acc', 'macro_f1']
    metric_names = ['Top-1 Acc', 'Top-5 Acc', 'Balanced Acc', 'Macro F1']
    
    clip_vals = [clip_only[m] for m in metrics]
    resnet_vals = [resnet_only[m] for m in metrics]
    fusion_vals = [best_fusion[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width, clip_vals, width, label='CLIP Only (α=0.0)', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, resnet_vals, width, label='ResNet Only (α=1.0)', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + width, fusion_vals, width, label=f'Best Fusion (α={best_alpha:.2f})', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    ax.set_xlabel('Metrics', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison: CLIP vs ResNet vs Fusion (Test Set)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(clip_vals), max(resnet_vals), max(fusion_vals)) * 1.15)
    
    plt.tight_layout()
    plt.savefig('fusion_model_comparison.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: fusion_model_comparison.png')
    plt.close()

def create_heatmap(results):
    """Heatmap showing all metrics across all alpha values"""
    alphas = sorted([float(k) for k in results['val'].keys()])
    metrics = ['top1_acc', 'top5_acc', 'balanced_acc', 'macro_f1']
    metric_labels = ['Top-1', 'Top-5', 'Balanced', 'Macro-F1']
    
    data = np.array([[results['val'][str(a)][m] for m in metrics] for a in alphas])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=data.max())
    
    ax.set_xticks(np.arange(len(alphas)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([f'{a:.1f}' for a in alphas], fontsize=10)
    ax.set_yticklabels(metric_labels, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Alpha (ResNet Weight)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_title('Fusion Performance Heatmap\n(Validation Set)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(alphas)):
            text = ax.text(j, i, f'{data[j, i]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fusion_heatmap.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: fusion_heatmap.png')
    plt.close()

def create_improvement_plot(results):
    """Show improvement of fusion over individual models"""
    best_alpha = results['test']['alpha']
    
    clip_only = results['val']['0.0']
    resnet_only = results['val']['1.0']
    best_fusion = results['test']
    
    metrics = ['top1_acc', 'top5_acc', 'balanced_acc', 'macro_f1']
    metric_names = ['Top-1\nAccuracy', 'Top-5\nAccuracy', 'Balanced\nAccuracy', 'Macro\nF1-Score']
    
    # Calculate improvements
    clip_vals = [clip_only[m] for m in metrics]
    resnet_vals = [resnet_only[m] for m in metrics]
    fusion_vals = [best_fusion[m] for m in metrics]
    
    clip_improve = [(fusion_vals[i] - clip_vals[i]) * 100 for i in range(len(metrics))]
    resnet_improve = [(fusion_vals[i] - resnet_vals[i]) * 100 for i in range(len(metrics))]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, clip_improve, width, label='vs CLIP-only', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, resnet_improve, width, label='vs ResNet-only', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:+.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9, fontweight='bold')
    
    autolabel(bars1)
    autolabel(bars2)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Metrics', fontsize=13, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Fusion Improvement over Individual Models (α={best_alpha:.2f})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fusion_improvement.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: fusion_improvement.png')
    plt.close()

def create_combined_overview(results):
    """Single comprehensive figure with all key insights"""
    best_alpha = results['test']['alpha']
    alphas = sorted([float(k) for k in results['val'].keys()])
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Top row: Alpha sweeps for all 4 metrics
    metrics_data = {
        'Top-1': [results['val'][str(a)]['top1_acc'] for a in alphas],
        'Top-5': [results['val'][str(a)]['top5_acc'] for a in alphas],
        'Bal-Acc': [results['val'][str(a)]['balanced_acc'] for a in alphas],
        'Macro-F1': [results['val'][str(a)]['macro_f1'] for a in alphas],
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
    ax_sweep = fig.add_subplot(gs[0, :])
    
    for idx, (name, values) in enumerate(metrics_data.items()):
        ax_sweep.plot(alphas, values, marker='o', linewidth=2, 
                     markersize=6, color=colors[idx], label=name, alpha=0.8)
    
    ax_sweep.axvline(best_alpha, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax_sweep.set_xlabel('Alpha (ResNet Weight)', fontsize=12, fontweight='bold')
    ax_sweep.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax_sweep.set_title(f'Fusion Weight Sweep (Best α={best_alpha:.2f})', 
                      fontsize=13, fontweight='bold')
    ax_sweep.legend(fontsize=10, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax_sweep.grid(True, alpha=0.3)
    
    # Bottom left: Model comparison
    ax_comp = fig.add_subplot(gs[1, 0])
    models = ['CLIP', 'ResNet', 'Fusion']
    top1_scores = [
        results['val']['0.0']['top1_acc'],
        results['val']['1.0']['top1_acc'],
        results['test']['top1_acc']
    ]
    bars = ax_comp.bar(models, top1_scores, color=['#3498db', '#e74c3c', '#2ecc71'], 
                       alpha=0.8, edgecolor='black', linewidth=1.5)
    ax_comp.set_ylabel('Top-1 Accuracy', fontsize=11, fontweight='bold')
    ax_comp.set_title('Model Comparison', fontsize=12, fontweight='bold')
    ax_comp.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax_comp.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Bottom middle: Metric breakdown
    ax_metrics = fig.add_subplot(gs[1, 1])
    metric_names = ['Top-1', 'Top-5', 'Bal-Acc', 'F1']
    fusion_scores = [
        results['test']['top1_acc'],
        results['test']['top5_acc'],
        results['test']['balanced_acc'],
        results['test']['macro_f1']
    ]
    bars = ax_metrics.barh(metric_names, fusion_scores, color=colors, 
                           alpha=0.8, edgecolor='black', linewidth=1.5)
    ax_metrics.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax_metrics.set_title('Fusion Test Metrics', fontsize=12, fontweight='bold')
    ax_metrics.grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax_metrics.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.3f}', ha='left', va='center', 
                       fontweight='bold', fontsize=9)
    
    # Bottom right: Summary stats
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.axis('off')
    
    # Use fallback sample sizes if not present in results
    val_size = results.get('val_size', 'N/A')
    test_size = results.get('test_size', 'N/A')

    summary_text = f"""
    FUSION SUMMARY
    ══════════════════════════
    
    Best Alpha: {best_alpha:.3f}
    (ResNet weight in ensemble)
    
    Test Performance:
    • Top-1 Accuracy: {results['test']['top1_acc']:.3f}
    • Top-5 Accuracy: {results['test']['top5_acc']:.3f}
    • Balanced Accuracy: {results['test']['balanced_acc']:.3f}
    • Macro F1-Score: {results['test']['macro_f1']:.3f}
    
    Dataset:
    • Validation: {val_size} samples
    • Test: {test_size} samples
    • Classes: {len(results['labels'])}
    
    Improvement vs CLIP-only:
    • Top-1: {(results['test']['top1_acc'] - results['val']['0.0']['top1_acc'])*100:+.2f}%
    
    Improvement vs ResNet-only:
    • Top-1: {(results['test']['top1_acc'] - results['val']['1.0']['top1_acc'])*100:+.2f}%
    """
    
    ax_summary.text(0.1, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('CLIP + ResNet Fusion: Complete Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('fusion_complete_overview.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: fusion_complete_overview.png')
    plt.close()

def main():
    print("Creating comprehensive fusion visualization graphs...")
    print(f"Loading results from: {FUSION_RESULTS}\n")
    
    results = load_results()
    
    print("Generating graphs:")
    create_alpha_sweep_plot(results)
    create_model_comparison_chart(results)
    create_heatmap(results)
    create_improvement_plot(results)
    create_combined_overview(results)
    
    print("\n" + "="*60)
    print("✓ All graphs created successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. fusion_alpha_sweep_detailed.png - Detailed alpha sweep (4 panels)")
    print("  2. fusion_model_comparison.png - Bar chart comparison")
    print("  3. fusion_heatmap.png - Performance heatmap")
    print("  4. fusion_improvement.png - Improvement over individual models")
    print("  5. fusion_complete_overview.png - Comprehensive single-page summary")

if __name__ == '__main__':
    main()
