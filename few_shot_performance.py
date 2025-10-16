#!/usr/bin/env python3
"""Run few-shot performance experiments and plot accuracy vs shots.

This script expects cached CLIP features in the same format used by
`scripts/dump_features.py`: features/<backbone>/<split>.npz containing X, y, paths.

It runs:
 - Zero-shot using text prompts (requires open_clip)
 - Prototype classifier using k-shot supports
 - Linear probe trained on k-shot supports

Outputs:
 - CSV table with metrics
 - PNG plot accuracy vs shots comparing zero-shot, prototype, and linear
"""
import os
from pathlib import Path
import sys
import numpy as np
import torch
import open_clip
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scripts.few_shot import ensure_features, load_labels, prototype_classifier, train_linear_probe, topk_acc, balanced_acc


def zero_shot_scores(label_names, backbone, pretrained, device):
	model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, device=device)
	model.eval()
	tokenizer = open_clip.get_tokenizer(backbone)
	prompts = [f"a photo of {n}" for n in label_names]
	with torch.no_grad():
		T = tokenizer(prompts).to(device)
		text = model.encode_text(T)
		text = text / text.norm(dim=-1, keepdim=True)
		text = text.float().cpu().numpy().T  # [D, K]
	return text

class Args:
    data_root = "/home/c/dkorot/AI4GOOD/provided_dir/datasets/mushroom/merged_dataset"
    train_csv = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/train.csv"
    val_csv = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/val.csv"
    test_csv = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/test.csv"
    labels = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/labels.tsv"
    backbone = "ViT-B-32-quickgelu"
    pretrained = "openai"
    shots = [1, 5]
    splits = ["val", "test"]
    save_dir = "features"
    results_dir = "results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    epochs = 100
    lr = 1e-2
    batch_size = 32


def main():
	# ap = argparse.ArgumentParser()
	# ap.add_argument("--data-root", required=True)
	# ap.add_argument("--train-csv", required=True)
	# ap.add_argument("--test-csv", required=True)
	# ap.add_argument("--labels", required=True)
	# ap.add_argument("--backbone", default="ViT-B-32")
	# ap.add_argument("--pretrained", default="openai")
	# ap.add_argument("--shots", nargs="+", type=int, default=[0,1,5,10,20])
	# ap.add_argument("--save-dir", default="features")
	# ap.add_argument("--results-dir", default="results")
	# ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	# ap.add_argument("--seed", type=int, default=42)
	# ap.add_argument("--epochs", type=int, default=200)
	# ap.add_argument("--lr", type=float, default=1e-2)
	# ap.add_argument("--batch-size", type=int, default=32)
	# args = ap.parse_args()
	print("~~~~~~~~~~~~~set up~~~~~~~~~~~~~~~~")
	args = Args()

	torch.manual_seed(args.seed); np.random.seed(args.seed)

	Path(args.results_dir).mkdir(parents=True, exist_ok=True)

	label_names, name2id = load_labels(args.labels)
	K = len(label_names)
	print("~~~~~~~~~~~~~load features~~~~~~~~~~~~~~~~")
	# load features
	X_tr, y_tr, _ = ensure_features("train", args.backbone, args.data_root, args.train_csv, args.labels, save_dir=args.save_dir, pretrained=args.pretrained)
	X_te, y_te, _ = ensure_features("test", args.backbone, args.data_root, args.test_csv, args.labels, save_dir=args.save_dir, pretrained=args.pretrained)

	# zero-shot text embeddings
	print("~~~~~~~~~~~~~0 shot~~~~~~~~~~~~~~~~")
	text_emb = zero_shot_scores(label_names, args.backbone, args.pretrained, args.device)

	records = []

	# if 0 in shots then compute zero-shot baseline
	if 0 in args.shots:
		scores_zs = X_te @ text_emb
		yhat_zs = np.argmax(scores_zs, axis=1)
		zs_top1 = (yhat_zs == y_te).mean()
		zs_bal = balanced_acc(y_te, yhat_zs, K)
		records.append((0, "zero-shot", zs_top1, zs_bal, None))
		print(f"shot=0: zs top1={zs_top1:.4f}")

	print("~~~~~~~~~~~~~few shot~~~~~~~~~~~~~~~~")
	for shot in sorted([s for s in args.shots if s>0]):
		print("~~~~~~~~~~~~~shot number: " +str(shot)+" ~~~~~~~~~~~~~~~~")
		# sample supports
		print("~~~~~~~~~~~~~sample supports~~~~~~~~~~~~~~~~")
		rng = np.random.default_rng(args.seed + int(shot))
		from scripts.few_shot import sample_few_shot_indices
		sup_idx, sup_labels = sample_few_shot_indices(y_tr, K, shot, rng)
		X_sup = X_tr[sup_idx]
		y_sup = sup_labels

		# prototype
		print("~~~~~~~~~~~~~prototype~~~~~~~~~~~~~~~~")
		prototypes = prototype_classifier(X_sup, y_sup, K)
		scores_proto = X_te @ prototypes.T
		yhat_proto = np.argmax(scores_proto, axis=1)
		proto_top1 = (yhat_proto == y_te).mean()
		proto_bal = balanced_acc(y_te, yhat_proto, K)
		records.append((shot, "prototype", proto_top1, proto_bal, None))

		# linear probe
		print("~~~~~~~~~~~~~linear probe~~~~~~~~~~~~~~~~")
		model = train_linear_probe(X_sup, y_sup, X_val=None, y_val=None, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device, seed=args.seed + int(shot))
		model.eval()
		with torch.no_grad():
			xt = torch.from_numpy(X_te).float().to(args.device)
			logits = model(xt).cpu().numpy()
			yhat_lin = np.argmax(logits, axis=1)
		lin_top1 = (yhat_lin == y_te).mean()
		lin_bal = balanced_acc(y_te, yhat_lin, K)
		records.append((shot, "linear", lin_top1, lin_bal, None))

		print(f"shot={shot}: proto top1={proto_top1:.4f}, lin top1={lin_top1:.4f}")

	# Save CSV table
	print("~~~~~~~~~~~~~save csv tab;e~~~~~~~~~~~~~~~~")
	out_csv = os.path.join(args.results_dir, f"few_shot_table_{args.backbone.replace(' ','_')}.csv")
	with open(out_csv, "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["shot","model","top1","balanced_acc","notes"])
		for r in records:
			w.writerow(r)
	print(f"Wrote table -> {out_csv}")

	# Plot accuracy vs shots (top1)
	# gather plot data
	print("~~~~~~~~~~~~~splot~~~~~~~~~~~~~~~~")
	import pandas as pd
	df = pd.DataFrame(records, columns=["shot","model","top1","balanced_acc","notes"])
	pivot = df.pivot(index="shot", columns="model", values="top1")
	pivot = pivot.reindex(sorted(pivot.index))

	plt.figure(figsize=(6,4))
	for col in pivot.columns:
		plt.plot(pivot.index, pivot[col], marker='o', label=col)
	plt.xlabel("shots")
	plt.ylabel("Top-1 accuracy")
	plt.xscale('log' if max(pivot.index) / min([s for s in pivot.index if s>0]) > 8 else 'linear')
	plt.xticks(pivot.index)
	plt.grid(alpha=0.3)
	plt.legend()
	out_png = os.path.join(args.results_dir, f"few_shot_plot_{args.backbone.replace(' ','_')}.png")
	plt.tight_layout()
	plt.savefig(out_png, dpi=180)
	print(f"Wrote plot -> {out_png}")


if __name__ == "__main__":
	main()

