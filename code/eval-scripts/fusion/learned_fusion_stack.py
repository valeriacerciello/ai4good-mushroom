"""
Train a small stacking classifier (multinomial logistic regression) on top of [CLIP_probs, ResNet_probs].
Uses functions from `fusion_clip_resnet.py`.
Saves results to JSON and prints a short summary.
"""
import argparse
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
import fusion_clip_resnet as fcr
import clip
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os


def build_clip_image_encoder(device):
    model, preprocess = clip.load('ViT-B/32', device=device)
    return model, preprocess
 
from fewshot_utils import compute_image_embeddings, build_prototypes, sims_to_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--prompts', type=str, default=fcr.DELTA_PROMPTS)
    parser.add_argument('--clip_temp', type=float, default=fcr.CLIP_TEMP)
    parser.add_argument('--val_n', type=int, default=200)
    parser.add_argument('--test_n', type=int, default=15614)
    parser.add_argument('--train_n', type=int, default=5000, help='number of train examples to use for stacker training')
    parser.add_argument('--shots', type=int, default=0, help='few-shot: number of shots per class for CLIP prototypes (0 = prompt pooling)')
    parser.add_argument('--device', type=str, default='cpu', help='device for CLIP/ResNet computation (cpu or cuda)')
    parser.add_argument('--out', type=str, default='stacked_fusion_results.json')
    args = parser.parse_args()

    # Load splits
    df_train = fcr.load_split_df('train', limit=args.train_n)
    df_val = fcr.load_split_df('val', limit=args.val_n)
    df_test = fcr.load_split_df('test', limit=args.test_n)
    labels = fcr.load_labels([df_train, df_val, df_test])
    print(f'Loaded train={len(df_train)}, val={len(df_val)}, test={len(df_test)}, classes={len(labels)}')

    # Compute CLIP probabilities. If shots>0 use prototype centroids (few-shot); else use prompt pooling
    if args.shots and args.shots > 0:
        device = args.device
        print(f'Computing CLIP embeddings (device={device}) for few-shot prototypes...')
        model, preprocess = build_clip_image_encoder(device)
        emb_train = compute_image_embeddings(df_train, preprocess, model, device=device)
        emb_val = compute_image_embeddings(df_val, preprocess, model, device=device)
        emb_test = compute_image_embeddings(df_test, preprocess, model, device=device)

        # Build centroids (few-shot) using shared helper
        centroids = build_prototypes(df_train, emb_train, labels, shots=args.shots)
        print('Computing CLIP prototype probs...')
        clip_train = sims_to_probs(emb_train, centroids, temp=args.clip_temp)
        clip_val = sims_to_probs(emb_val, centroids, temp=args.clip_temp)
        clip_test = sims_to_probs(emb_test, centroids, temp=args.clip_temp)
    else:
        print('Computing CLIP probs (prompt pooling)')
        clip_train = fcr.compute_clip_probs(df_train, labels, args.prompts, temp=args.clip_temp)
        clip_val = fcr.compute_clip_probs(df_val, labels, args.prompts, temp=args.clip_temp)
        clip_test = fcr.compute_clip_probs(df_test, labels, args.prompts, temp=args.clip_temp)

    if not fcr.DEFAULT_RESNET_CKPT:
        print('No default resnet ckpt configured; using provided ckpt')

    if not args.ckpt:
        print('No ckpt provided; will use uniform ResNet probs')
        res_train = np.full_like(clip_train, 1.0 / len(labels))
        res_val = np.full_like(clip_val, 1.0 / len(labels))
        res_test = np.full_like(clip_test, 1.0 / len(labels))
    else:
        print('Computing ResNet probs (train/val/test)')
        res_train = fcr.compute_resnet_probs(df_train, labels, args.ckpt, model_name='resnet18', device=args.device)
        res_val = fcr.compute_resnet_probs(df_val, labels, args.ckpt, model_name='resnet18', device=args.device)
        res_test = fcr.compute_resnet_probs(df_test, labels, args.ckpt, model_name='resnet18', device=args.device)

    # Prepare stacked features
    X_train = np.concatenate([clip_train, res_train], axis=1)
    X_val = np.concatenate([clip_val, res_val], axis=1)
    X_test = np.concatenate([clip_test, res_test], axis=1)

    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_train = np.array([label_to_idx[l] for l in df_train['label'].tolist()])
    y_val = np.array([label_to_idx[l] for l in df_val['label'].tolist()])
    y_test = np.array([label_to_idx[l] for l in df_test['label'].tolist()])

    print('Training logistic regression stacker (multinomial) on train predictions...')
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000, class_weight='balanced')
    clf.fit(X_train, y_train)

    print('Predicting on test...')
    probs_test_partial = clf.predict_proba(X_test)
    # Map partial class probs (clf.classes_) into full label space
    full_C = len(labels)
    probs_test = np.zeros((probs_test_partial.shape[0], full_C), dtype=float)
    for i, cls in enumerate(clf.classes_):
        # cls is an integer label index used during training
        probs_test[:, int(cls)] = probs_test_partial[:, i]

    # Evaluate stacked model
    stacked_metrics = fcr.evaluate_metrics(df_test['label'].tolist(), probs_test, labels)

    # Also compute best alpha sweep (use Top-5 selection)
    print('Computing alpha sweep baseline (val)')
    alphas = fcr.ALPHAS
    best_alpha = None
    best_top5 = -1.0
    for a in alphas:
        fused_val = a * res_val + (1 - a) * clip_val
        m = fcr.evaluate_metrics(df_val['label'].tolist(), fused_val, labels)
        if m['top5_acc'] > best_top5:
            best_top5 = m['top5_acc']
            best_alpha = a

    fused_test_bestalpha = best_alpha * res_test + (1 - best_alpha) * clip_test
    bestalpha_metrics = fcr.evaluate_metrics(df_test['label'].tolist(), fused_test_bestalpha, labels)

    results = {
        'stacked': stacked_metrics,
        'best_alpha': {'alpha': best_alpha, 'metrics': bestalpha_metrics},
        'labels_count': len(labels),
        'val_size': len(df_val),
        'test_size': len(df_test),
    }

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== Results Summary ===')
    print(f"Stacked (LR) Test: top1={stacked_metrics['top1_acc']:.3f}, top5={stacked_metrics['top5_acc']:.3f}, "
          f"bal_acc={stacked_metrics['balanced_acc']:.3f}, macro_f1={stacked_metrics['macro_f1']:.3f}")
    print(f"Best-alpha (val->test) alpha={best_alpha} Test: top1={bestalpha_metrics['top1_acc']:.3f}, top5={bestalpha_metrics['top5_acc']:.3f}")
    print(f"Saved results to {args.out}")


if __name__ == '__main__':
    main()
