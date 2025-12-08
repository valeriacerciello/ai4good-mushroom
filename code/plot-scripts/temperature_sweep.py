"""Temperature sweep for per-prompt softmax pooling on delta_prompts.json"""
import json
import os
import torch
import torch.nn.functional as F
import clip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score
from clip_utils import get_text_embeddings

TEST_CSV = "/zfs/ai4good/datasets/mushroom/test.csv"
DELTA_PROMPTS = "delta_prompts.json"
NUM_SAMPLES = 200
TEMPS = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5]


def load_test_subset(csv_file, n=200):
    df = pd.read_csv(csv_file)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df.head(n)


def encode_images(model, preprocess, image_paths, device):
    images = []
    for p in image_paths:
        local = p.replace('/kaggle/working/', '/zfs/ai4good/datasets/mushroom/')
        if os.path.exists(local):
            from PIL import Image
            im = Image.open(local).convert('RGB')
            img_t = preprocess(im).unsqueeze(0).to(device)
            images.append(img_t)
        else:
            images.append(None)
    embeddings = []
    for img in images:
        if img is None:
            embeddings.append(None)
            continue
        with torch.no_grad():
            emb = model.encode_image(img)
            emb = F.normalize(emb, dim=-1)
            embeddings.append(emb.cpu().squeeze(0))
    return embeddings


def predict_with_prompt_pooling(image_emb, embeddings_bundle, temp=0.12, top_k=1):
    scores = {}
    for label, info in embeddings_bundle.items():
        pe = info['prompt_embeddings']
        if pe.numel() == 0:
            scores[label] = -1e9
            continue
        sims = torch.cosine_similarity(image_emb.unsqueeze(0), pe.to(image_emb.dtype), dim=1)
        w = torch.softmax(sims / temp, dim=0)
        pooled = float((w * sims).sum())
        scores[label] = pooled
    sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_k == 1:
        return sorted_labels[0][0]
    else:
        return [lbl for lbl, _ in sorted_labels[:top_k]]


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    df = load_test_subset(TEST_CSV, NUM_SAMPLES)
    image_paths = df['image_path'].tolist()
    true_labels = df['label'].tolist()

    print("Encoding images...")
    image_embeddings = encode_images(model, preprocess, image_paths, device)

    labels = sorted(list(set(true_labels)))

    print(f'Computing text embeddings for delta_prompts.json...')
    delta_embs = get_text_embeddings(labels, prompt_set='ensemble', model_name='ViT-B/32', device=device,
                                      include_common_names=True, common_prompts_path=DELTA_PROMPTS)

    results = []
    
    for temp in TEMPS:
        print(f'\nTesting temperature: {temp}')
        preds_top1 = []
        preds_top5 = []
        
        for img_emb in image_embeddings:
            if img_emb is None:
                preds_top1.append(None)
                preds_top5.append(None)
                continue
            pred1 = predict_with_prompt_pooling(img_emb, delta_embs, temp=temp, top_k=1)
            pred5 = predict_with_prompt_pooling(img_emb, delta_embs, temp=temp, top_k=5)
            preds_top1.append(pred1)
            preds_top5.append(pred5)
        
        # compute metrics
        mask = [p is not None for p in preds_top1]
        trues_m = [t for t, m in zip(true_labels, mask) if m]
        preds1_m = [p for p, m in zip(preds_top1, mask) if m]
        preds5_m = [p for p, m in zip(preds_top5, mask) if m]
        
        # Top-1 accuracy
        acc_top1 = sum(1 for p, t in zip(preds1_m, trues_m) if p == t) / len(trues_m)
        
        # Top-5 accuracy
        acc_top5 = sum(1 for p, t in zip(preds5_m, trues_m) if t in p) / len(trues_m)
        
        # Balanced accuracy (using top-1 predictions)
        bal_acc = balanced_accuracy_score(trues_m, preds1_m)
        
        # Macro F1-score (using top-1 predictions)
        macro_f1 = f1_score(trues_m, preds1_m, average='macro', zero_division=0)
        
        results.append({
            'temperature': temp,
            'top1_acc': acc_top1,
            'top5_acc': acc_top5,
            'balanced_acc': bal_acc,
            'macro_f1': macro_f1
        })
        
        print(f'  Top-1: {acc_top1:.4f}, Top-5: {acc_top5:.4f}')
        print(f'  Balanced Acc: {bal_acc:.4f}, Macro-F1: {macro_f1:.4f}')
    
    # save results
    with open('temperature_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # find best for each metric
    best_top1 = max(results, key=lambda x: x['top1_acc'])
    best_top5 = max(results, key=lambda x: x['top5_acc'])
    best_bal = max(results, key=lambda x: x['balanced_acc'])
    best_f1 = max(results, key=lambda x: x['macro_f1'])
    
    print('\n=== Temperature Sweep Summary ===')
    print(f'Best Top-1: T={best_top1["temperature"]}, Acc={best_top1["top1_acc"]:.4f}')
    print(f'Best Top-5: T={best_top5["temperature"]}, Acc={best_top5["top5_acc"]:.4f}')
    print(f'Best Balanced Acc: T={best_bal["temperature"]}, Acc={best_bal["balanced_acc"]:.4f}')
    print(f'Best Macro-F1: T={best_f1["temperature"]}, F1={best_f1["macro_f1"]:.4f}')
    print('\nFull results saved to temperature_sweep_results.json')
    
    # Generate plots
    print('\nGenerating plots...')
    temps_list = [r['temperature'] for r in results]
    top1_list = [r['top1_acc'] for r in results]
    top5_list = [r['top5_acc'] for r in results]
    bal_list = [r['balanced_acc'] for r in results]
    f1_list = [r['macro_f1'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temperature Sweep: Impact on Classification Metrics', fontsize=16, fontweight='bold')
    
    # Top-1 Accuracy
    axes[0, 0].plot(temps_list, top1_list, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Temperature', fontsize=12)
    axes[0, 0].set_ylabel('Top-1 Accuracy', fontsize=12)
    axes[0, 0].set_title('Top-1 Accuracy vs Temperature', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    best_idx = top1_list.index(max(top1_list))
    axes[0, 0].axvline(temps_list[best_idx], color='red', linestyle='--', alpha=0.5, label=f'Best T={temps_list[best_idx]}')
    axes[0, 0].legend()
    
    # Top-5 Accuracy
    axes[0, 1].plot(temps_list, top5_list, 'o-', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel('Temperature', fontsize=12)
    axes[0, 1].set_ylabel('Top-5 Accuracy', fontsize=12)
    axes[0, 1].set_title('Top-5 Accuracy vs Temperature', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    best_idx = top5_list.index(max(top5_list))
    axes[0, 1].axvline(temps_list[best_idx], color='red', linestyle='--', alpha=0.5, label=f'Best T={temps_list[best_idx]}')
    axes[0, 1].legend()
    
    # Balanced Accuracy
    axes[1, 0].plot(temps_list, bal_list, 'o-', linewidth=2, markersize=8, color='#F18F01')
    axes[1, 0].set_xlabel('Temperature', fontsize=12)
    axes[1, 0].set_ylabel('Balanced Accuracy', fontsize=12)
    axes[1, 0].set_title('Balanced Accuracy vs Temperature', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    best_idx = bal_list.index(max(bal_list))
    axes[1, 0].axvline(temps_list[best_idx], color='red', linestyle='--', alpha=0.5, label=f'Best T={temps_list[best_idx]}')
    axes[1, 0].legend()
    
    # Macro F1
    axes[1, 1].plot(temps_list, f1_list, 'o-', linewidth=2, markersize=8, color='#6A994E')
    axes[1, 1].set_xlabel('Temperature', fontsize=12)
    axes[1, 1].set_ylabel('Macro F1-Score', fontsize=12)
    axes[1, 1].set_title('Macro F1-Score vs Temperature', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    best_idx = f1_list.index(max(f1_list))
    axes[1, 1].axvline(temps_list[best_idx], color='red', linestyle='--', alpha=0.5, label=f'Best T={temps_list[best_idx]}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('temperature_sweep_plots.png', dpi=300, bbox_inches='tight')
    print('Plots saved to temperature_sweep_plots.png')
    
    # Combined plot
    fig2, ax = plt.subplots(figsize=(12, 7))
    ax.plot(temps_list, top1_list, 'o-', linewidth=2, markersize=8, label='Top-1 Accuracy', color='#2E86AB')
    ax.plot(temps_list, top5_list, 's-', linewidth=2, markersize=8, label='Top-5 Accuracy', color='#A23B72')
    ax.plot(temps_list, bal_list, '^-', linewidth=2, markersize=8, label='Balanced Accuracy', color='#F18F01')
    ax.plot(temps_list, f1_list, 'd-', linewidth=2, markersize=8, label='Macro F1-Score', color='#6A994E')
    ax.set_xlabel('Temperature', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('All Metrics vs Temperature (Delta Prompts)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('temperature_sweep_combined.png', dpi=300, bbox_inches='tight')
    print('Combined plot saved to temperature_sweep_combined.png')


if __name__ == '__main__':
    run()
