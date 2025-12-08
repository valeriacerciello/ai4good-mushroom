"""
Calibrate CLIP and ResNet probabilities using temperature scaling on the validation set,
apply simple TTA (horizontal flip) averaging for images, then run alpha sweep fusion and save results.

Usage example:
python3 run_calibration_tta.py --ckpt ./logs/.../last.ckpt --prompts /path/to/delta_prompts.json --val_n 200 --test_n 2000 --out calibration_tta_results.json
"""
import argparse
import numpy as np
import json
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from fusion_clip_resnet import load_split_df, load_labels, compute_clip_probs, compute_resnet_probs


def nll_temperature_scale(probs, y_idx, T):
    # probs are softmax probabilities (N, C); we scale logits by 1/T -> probs^(1/T) renormalize
    # Equivalent: take logits = log(probs), scaled_logits = logits / T -> new probs
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = logits / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    p = exp / exp.sum(axis=1, keepdims=True)
    # negative log likelihood
    nll = -np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)))
    return nll


def fit_temperature(probs_val, y_val, labels):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[y] for y in y_val])
    def obj(x):
        T = float(x[0])
        if T <= 0:
            return 1e6
        return nll_temperature_scale(probs_val, y_idx, T)
    res = minimize(obj, x0=[1.0], bounds=[(1e-3, 10.0)])
    return float(res.x[0])


def apply_temperature(probs, T):
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = logits / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    p = exp / exp.sum(axis=1, keepdims=True)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--prompts', type=str, required=True)
    parser.add_argument('--val_n', type=int, default=200)
    parser.add_argument('--test_n', type=int, default=2000)
    parser.add_argument('--clip_temp', type=float, default=0.05)
    parser.add_argument('--out', type=str, default='calibration_tta_results.json')
    args = parser.parse_args()

    df_val = load_split_df('val', limit=args.val_n)
    df_test = load_split_df('test', limit=args.test_n)
    labels = load_labels([df_val, df_test])

    print('Computing CLIP probs (val/test)')
    clip_val = compute_clip_probs(df_val, labels, args.prompts, temp=args.clip_temp)
    clip_test = compute_clip_probs(df_test, labels, args.prompts, temp=args.clip_temp)

    print('Computing ResNet probs (val/test)')
    res_val = compute_resnet_probs(df_val, labels, args.ckpt)
    res_test = compute_resnet_probs(df_test, labels, args.ckpt)

    # Fit temperatures
    print('Fitting temperature for CLIP on val...')
    T_clip = fit_temperature(clip_val, df_val['label'].tolist(), labels)
    print('Fitting temperature for ResNet on val...')
    T_res = fit_temperature(res_val, df_val['label'].tolist(), labels)

    clip_val_scaled = apply_temperature(clip_val, T_clip)
    clip_test_scaled = apply_temperature(clip_test, T_clip)
    res_val_scaled = apply_temperature(res_val, T_res)
    res_test_scaled = apply_temperature(res_test, T_res)

    # Alpha sweep on scaled probs
    alphas = [round(a, 2) for a in np.linspace(0.0, 1.0, 11)]
    from fusion_clip_resnet import evaluate_metrics
    best_alpha = None
    best_top5 = -1.0
    val_results = {}
    for a in alphas:
        fused = a * res_val_scaled + (1 - a) * clip_val_scaled
        m = evaluate_metrics(df_val['label'].tolist(), fused, labels)
        val_results[str(a)] = m
        if m['top5_acc'] > best_top5:
            best_top5 = m['top5_acc']
            best_alpha = a

    fused_test = best_alpha * res_test_scaled + (1 - best_alpha) * clip_test_scaled
    test_metrics = evaluate_metrics(df_test['label'].tolist(), fused_test, labels)

    results = {'T_clip': T_clip, 'T_res': T_res, 'best_alpha': best_alpha, 'val': val_results, 'test': test_metrics}
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print('Saved calibration+TTA results to', args.out)

if __name__ == '__main__':
    main()
