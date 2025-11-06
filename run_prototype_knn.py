"""
Compute CLIP image embeddings, build per-class prototypes (centroids) from `train` split,
classify `val` and `test` by cosine similarity to centroids (optionally softmax with temperature),
evaluate Top-1/Top-5/BalancedAcc/MacroF1, and save results.

Usage example:
python3 run_prototype_knn.py --train_limit 5000 --val_n 200 --test_n 2000 --out prototype_knn_results.json
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
import pandas as pd
import os

from fusion_clip_resnet import load_split_df, load_labels, fix_path, compute_resnet_probs, evaluate_metrics


def build_clip_image_encoder(device):
    model, preprocess = clip.load('ViT-B/32', device=device)
    return model, preprocess


def compute_image_embeddings(df, preprocess, model, batch_size=32, device='cpu'):
    model.to(device)
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            paths = df['image_path'].iloc[i:i+batch_size].tolist()
            tensors = []
            for p in paths:
                try:
                    img = Image.open(p).convert('RGB')
                    t = preprocess(img).to(device)
                except Exception:
                    t = torch.zeros((3, 224, 224), device=device)
                tensors.append(t)
            batch = torch.stack(tensors, dim=0)
            b_emb = model.encode_image(batch).float()
            b_emb = F.normalize(b_emb, dim=1)
            embs.append(b_emb.cpu())
    embs = torch.cat(embs, dim=0).numpy()
    return embs


def evaluate(y_true, probs, labels):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[y] for y in y_true])
    top1_idx = probs.argmax(axis=1)
    top1_acc = float((top1_idx == y_idx).mean())
    top5_idx = np.argsort(-probs, axis=1)[:, :5]
    top5_acc = float(np.mean([y_idx[i] in top5_idx[i] for i in range(len(y_idx))]))
    y_pred = top1_idx
    bal_acc = balanced_accuracy_score(y_idx, y_pred)
    macro_f1 = f1_score(y_idx, y_pred, average='macro', zero_division=0)
    return {'top1_acc': top1_acc, 'top5_acc': top5_acc, 'balanced_acc': float(bal_acc), 'macro_f1': float(macro_f1)}


ALPHAS = [round(a, 2) for a in np.linspace(0.0, 1.0, 11)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_limit', type=int, default=5000)
    parser.add_argument('--val_n', type=int, default=200)
    parser.add_argument('--test_n', type=int, default=2000)
    parser.add_argument('--temp', type=float, default=0.01)
    parser.add_argument('--shots', type=int, default=0,
                        help='Number of shots per class to form prototypes. 0 means use all available examples.')
    parser.add_argument('--ckpt', type=str, default=None, help='Optional ResNet checkpoint to fuse with')
    parser.add_argument('--model', type=str, default='resnet18', help='ResNet model name')
    parser.add_argument('--clip_temp', type=float, default=0.05, help='Temperature used when pooling CLIP prompts (if used)')
    parser.add_argument('--out', type=str, default='prototype_knn_results.json')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device
    # Load splits
    df_train = load_split_df('train', limit=args.train_limit)
    df_val = load_split_df('val', limit=args.val_n)
    df_test = load_split_df('test', limit=args.test_n)

    labels = load_labels([df_train, df_val, df_test])
    print(f'Loaded train={len(df_train)}, val={len(df_val)}, test={len(df_test)}, classes={len(labels)}')

    model, preprocess = build_clip_image_encoder(device)
    print('Computing train image embeddings...')
    emb_train = compute_image_embeddings(df_train, preprocess, model, device=device)
    print('Computing val image embeddings...')
    emb_val = compute_image_embeddings(df_val, preprocess, model, device=device)
    print('Computing test image embeddings...')
    emb_test = compute_image_embeddings(df_test, preprocess, model, device=device)

    # Build centroids (few-shot if requested)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    class_sums = np.zeros((len(labels), emb_train.shape[1]), dtype=np.float32)
    class_counts = np.zeros((len(labels),), dtype=np.int32)
    if args.shots and args.shots > 0:
        print(f'Using up to {args.shots} shots per class to build prototypes')
        # group indices by label
        grp = {}
        for i, row in df_train.reset_index(drop=True).iterrows():
            lab = row['label']
            if lab not in label_to_idx:
                continue
            grp.setdefault(lab, []).append(i)
        # deterministic sampling
        for lab, idxs in grp.items():
            sel = idxs[:args.shots] if len(idxs) >= args.shots else idxs
            idx = label_to_idx[lab]
            for ii in sel:
                class_sums[idx] += emb_train[ii]
                class_counts[idx] += 1
    else:
        for i, row in df_train.reset_index(drop=True).iterrows():
            lab = row['label']
            if lab not in label_to_idx:
                continue
            idx = label_to_idx[lab]
            class_sums[idx] += emb_train[i]
            class_counts[idx] += 1
    # avoid zero division
    class_counts = np.maximum(class_counts, 1)
    centroids = class_sums / class_counts[:, None]
    # normalize
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # compute similarity and softmax
    def sims_to_probs(embs):
        sims = embs @ centroids.T  # (N, C)
        sims = sims / args.temp
        # clip for numerical stability
        sims = sims - sims.max(axis=1, keepdims=True)
        exp = np.exp(sims)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

    probs_val = sims_to_probs(emb_val)
    probs_test = sims_to_probs(emb_test)

    results = {'val': {}, 'test': {}, 'shots': args.shots, 'train_limit': args.train_limit,
               'val_n': args.val_n, 'test_n': args.test_n}

    # If user provided a ResNet checkpoint, compute ResNet probs; otherwise treat ResNet as uniform
    if args.ckpt and os.path.exists(args.ckpt):
        print('Computing ResNet probabilities (val/test)...')
        res_val = compute_resnet_probs(df_val, labels, args.ckpt, model_name=args.model, device=args.device)
        res_test = compute_resnet_probs(df_test, labels, args.ckpt, model_name=args.model, device=args.device)
    else:
        if args.ckpt:
            print(f'WARNING: provided checkpoint {args.ckpt} not found. Using uniform ResNet probs.')
        else:
            print('No ResNet checkpoint provided. Using uniform ResNet probs.')
        res_val = np.full_like(probs_val, 1.0 / len(labels))
        res_test = np.full_like(probs_test, 1.0 / len(labels))

    # Sweep alpha and evaluate on val to pick best fusion weight
    best_alpha = None
    best_top1 = -1.0
    best_f1 = -1.0
    y_val = df_val['label'].tolist()
    y_test = df_test['label'].tolist()
    for a in ALPHAS:
        fused_val = a * res_val + (1 - a) * probs_val
        m = evaluate_metrics(y_val, fused_val, labels)
        results['val'][str(a)] = m
        if (m['top1_acc'] > best_top1) or (np.isclose(m['top1_acc'], best_top1) and m['macro_f1'] > best_f1):
            best_top1 = m['top1_acc']
            best_f1 = m['macro_f1']
            best_alpha = a

    # Evaluate test with best alpha
    fused_test = best_alpha * res_test + (1 - best_alpha) * probs_test
    test_metrics = evaluate_metrics(y_test, fused_test, labels)
    results['best_alpha'] = {'alpha': best_alpha, 'metrics': test_metrics}
    results['labels_count'] = len(labels)
    results['labels'] = labels

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print('Saved prototype KNN + fusion results to', args.out)

if __name__ == '__main__':
    main()
