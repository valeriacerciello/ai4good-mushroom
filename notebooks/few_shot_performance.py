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
from sklearn.metrics import f1_score
from scripts.few_shot import ensure_features, load_labels, prototype_classifier, train_linear_probe, topk_acc, balanced_acc


def zero_shot_scores(label_names, backbone, pretrained, device):
	print("creating model")
	model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, device=device)
	print("evaluatemodel")
	model.eval()
	print("get tocken")
	tokenizer = open_clip.get_tokenizer(backbone)
	prompts = [f"a photo of {n}" for n in label_names]
	print("start encoding text")
	with torch.no_grad():
		T = tokenizer(prompts).to(device)
		print("encode text")
		text = model.encode_text(T)
		print("normalize")
		text = text / text.norm(dim=-1, keepdim=True)
		print("transopse")
		text = text.float().cpu().numpy().T  # [D, K]
	return text

class Args:
    data_root = "/home/c/dkorot/AI4GOOD/provided_dir/datasets/mushroom/merged_dataset"
    train_csv = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/train.csv"
    val_csv = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/val.csv"
    test_csv = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/test.csv"
    labels = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/labels.tsv"
    backbones = ["ViT-B-32-quickgelu", "ViT-L-14", "ViT-B-16"]  # List of backbones to evaluate
    pretrained = "openai"
    shots = [0, 1, 5, 10, 20]
    splits = ["val", "test"]
    save_dir = "features"
    results_dir = "results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    epochs = 100
    lr = 1e-2
    batch_size = 32


def evaluate_backbone(backbone, args, label_names, K):
	print(f"\n~~~~~~~~~~~~~Evaluating backbone: {backbone}~~~~~~~~~~~~~~~~")
	
	print("~~~~~~~~~~~~~load features~~~~~~~~~~~~~~~~")
	# load features for this backbone
	X_tr, y_tr, _ = ensure_features("train", backbone, args.data_root, args.train_csv, args.labels, save_dir=args.save_dir, pretrained=args.pretrained)
	X_vl, y_vl, _ = ensure_features("val",   backbone, args.data_root, args.val_csv,   args.labels, save_dir=args.save_dir, pretrained=args.pretrained)
	X_te, y_te, _ = ensure_features("test",  backbone, args.data_root, args.test_csv,  args.labels, save_dir=args.save_dir, pretrained=args.pretrained)

	# zero-shot text embeddings for this backbone
	print("~~~~~~~~~~~~~0 shot~~~~~~~~~~~~~~~~")
	text_emb = zero_shot_scores(label_names, backbone, args.pretrained, args.device)

	backbone_records = []
	
	def evaluate_split(split_name, X, y):
		split_records = []
		print(f"~~~~~~~~~~~~~Evaluating {split_name} for {backbone}~~~~~~~~~~~~~~~~")
		
		# if 0 in shots then compute zero-shot baseline
		if 0 in args.shots:
			scores_zs = X @ text_emb
			yhat_zs = np.argmax(scores_zs, axis=1)
			zs_top1 = (yhat_zs == y).mean()
			zs_bal = balanced_acc(y, yhat_zs, K)
			zs_top5 = topk_acc(y, scores_zs, 5)
			zs_macro = f1_score(y, yhat_zs, average='macro')
			split_records.append((0, "zero-shot", split_name, backbone, zs_top1, zs_top5, zs_bal, zs_macro))
			print(f"{split_name} shot=0: zs top1={zs_top1:.4f}")
		
		print(f"~~~~~~~~~~~~~{split_name} few shot~~~~~~~~~~~~~~~~")
		
		for shot in sorted([s for s in args.shots if s>0]):
			print(f"~~~~~~~~~~~~~{split_name} shot number: {shot}~~~~~~~~~~~~~~~~")
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
			scores_proto = X @ prototypes.T
			yhat_proto = np.argmax(scores_proto, axis=1)
			proto_top1 = (yhat_proto == y).mean()
			proto_bal = balanced_acc(y, yhat_proto, K)
			proto_top5 = topk_acc(y, scores_proto, 5)
			proto_macro = f1_score(y, yhat_proto, average='macro')
			split_records.append((shot, "prototype", split_name, backbone, proto_top1, proto_top5, proto_bal, proto_macro))

			# linear probe
			print("~~~~~~~~~~~~~linear probe~~~~~~~~~~~~~~~~")
			model = train_linear_probe(X_sup, y_sup, X_val=None, y_val=None, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device, seed=args.seed + int(shot))
			model.eval()
			with torch.no_grad():
				xt = torch.from_numpy(X).float().to(args.device)
				logits = model(xt).cpu().numpy()
				yhat_lin = np.argmax(logits, axis=1)
			lin_top1 = (yhat_lin == y).mean()
			lin_bal = balanced_acc(y, yhat_lin, K)
			lin_top5 = topk_acc(y, logits, 5)
			lin_macro = f1_score(y, yhat_lin, average='macro')
			split_records.append((shot, "linear", split_name, backbone, lin_top1, lin_top5, lin_bal, lin_macro))

			print(f"{split_name} shot={shot}: proto top1={proto_top1:.4f}, lin top1={lin_top1:.4f}")
		
		return split_records

	# Run evaluation for validation and test sets
	print("~~~~~~~~~~~~~evaluate splits~~~~~~~~~~~~~~~~")
	for split_name, (X, y) in [("val", (X_vl, y_vl)), ("test", (X_te, y_te))]:
		split_records = evaluate_split(split_name, X, y)
		backbone_records.extend(split_records)
	
	return backbone_records

def main():
	# ap = argparse.ArgumentParser()
	# ap.add_argument("--data-root", required=True)
	# ap.add_argument("--train-csv", required=True)
	# ap.add_argument("--test-csv", required=True)
	# ap.add_argument("--labels", required=True)
	# ap.add_argument("--backbones", nargs="+", default=["ViT-B-32"])
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

	# Evaluate all backbones
	all_records = []
	for backbone in args.backbones:
		backbone_records = evaluate_backbone(backbone, args, label_names, K)
		all_records.extend(backbone_records)

	records = []

	def evaluate_split(split_name, X, y):
		split_records = []
		print(f"~~~~~~~~~~~~~Evaluating {split_name}~~~~~~~~~~~~~~~~")
		
		# if 0 in shots then compute zero-shot baseline
		if 0 in args.shots:
			scores_zs = X @ text_emb
			yhat_zs = np.argmax(scores_zs, axis=1)
			zs_top1 = (yhat_zs == y).mean()
			zs_bal = balanced_acc(y, yhat_zs, K)
			zs_top5 = topk_acc(y, scores_zs, 5)
			zs_macro = f1_score(y, yhat_zs, average='macro')
			split_records.append((0, "zero-shot", split_name, zs_top1, zs_top5, zs_bal, zs_macro))
			print(f"{split_name} shot=0: zs top1={zs_top1:.4f}")
		
		print(f"~~~~~~~~~~~~~{split_name} few shot~~~~~~~~~~~~~~~~")
		
		for shot in sorted([s for s in args.shots if s>0]):
			print(f"~~~~~~~~~~~~~{split_name} shot number: {shot}~~~~~~~~~~~~~~~~")
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
			scores_proto = X @ prototypes.T
			yhat_proto = np.argmax(scores_proto, axis=1)
			proto_top1 = (yhat_proto == y).mean()
			proto_bal = balanced_acc(y, yhat_proto, K)
			proto_top5 = topk_acc(y, scores_proto, 5)
			proto_macro = f1_score(y, yhat_proto, average='macro')
			split_records.append((shot, "prototype", split_name, proto_top1, proto_top5, proto_bal, proto_macro))

			# linear probe
			print("~~~~~~~~~~~~~linear probe~~~~~~~~~~~~~~~~")
			model = train_linear_probe(X_sup, y_sup, X_val=None, y_val=None, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device, seed=args.seed + int(shot))
			model.eval()
			with torch.no_grad():
				xt = torch.from_numpy(X).float().to(args.device)
				logits = model(xt).cpu().numpy()
				yhat_lin = np.argmax(logits, axis=1)
			lin_top1 = (yhat_lin == y).mean()
			lin_bal = balanced_acc(y, yhat_lin, K)
			lin_top5 = topk_acc(y, logits, 5)
			lin_macro = f1_score(y, yhat_lin, average='macro')
			split_records.append((shot, "linear", split_name, lin_top1, lin_top5, lin_bal, lin_macro))

			print(f"{split_name} shot={shot}: proto top1={proto_top1:.4f}, lin top1={lin_top1:.4f}")
		
		return split_records
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
			scores_proto = X @ prototypes.T
			yhat_proto = np.argmax(scores_proto, axis=1)
			proto_top1 = (yhat_proto == y).mean()
			proto_bal = balanced_acc(y, yhat_proto, K)
			proto_top5 = topk_acc(y, scores_proto, 5)
			proto_macro = f1_score(y, yhat_proto, average='macro')
			split_records.append((shot, "prototype", split_name, proto_top1, proto_top5, proto_bal, proto_macro))

			# linear probe
			print("~~~~~~~~~~~~~linear probe~~~~~~~~~~~~~~~~")
			model = train_linear_probe(X_sup, y_sup, X_val=None, y_val=None, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device, seed=args.seed + int(shot))
			model.eval()
			with torch.no_grad():
				xt = torch.from_numpy(X).float().to(args.device)
				logits = model(xt).cpu().numpy()
				yhat_lin = np.argmax(logits, axis=1)
			lin_top1 = (yhat_lin == y).mean()
			lin_bal = balanced_acc(y, yhat_lin, K)
			lin_top5 = topk_acc(y, logits, 5)
			lin_macro = f1_score(y, yhat_lin, average='macro')
			split_records.append((shot, "linear", split_name, lin_top1, lin_top5, lin_bal, lin_macro))

			print(f"{split_name} shot={shot}: proto top1={proto_top1:.4f}, lin top1={lin_top1:.4f}")
		return split_records

	# Run evaluation for validation and test sets
	print("~~~~~~~~~~~~~evaluate splits~~~~~~~~~~~~~~~~")
	all_records = []
	for split_name, (X, y) in [("val", (X_vl, y_vl)), ("test", (X_te, y_te))]:
		split_records = evaluate_split(split_name, X, y)
		all_records.extend(split_records)

	# Save CSV table with all results
	print("~~~~~~~~~~~~~save csv table~~~~~~~~~~~~~~~~")
	out_csv = os.path.join(args.results_dir, "few_shot_table_all_backbones.csv")
	with open(out_csv, "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		# columns: shot, model, split, backbone, top1, top5, balanced_acc, macro_f1
		w.writerow(["shot","model","split","backbone","top1","top5","balanced_acc","macro_f1"])
		for r in all_records:
			w.writerow(r)
	print(f"Wrote table -> {out_csv}")

	# Plot accuracy vs shots (top1) for all backbones and splits
	print("~~~~~~~~~~~~~plot~~~~~~~~~~~~~~~~")
	df = pd.DataFrame(all_records, columns=["shot","model","split","backbone","top1","top5","balanced_acc","macro_f1"])
	
	# Create a 2x2 grid of plots (backbones vs splits)
	n_backbones = len(args.backbones)
	fig, axes = plt.subplots(2, n_backbones, figsize=(5*n_backbones, 8))
	
	for i, backbone in enumerate(args.backbones):
		backbone_data = df[df['backbone'] == backbone]
		
		# Validation plot (top row)
		val_data = backbone_data[backbone_data['split'] == 'val']
		pivot_val = val_data.pivot(index="shot", columns="model", values="top1")
		pivot_val = pivot_val.reindex(sorted(pivot_val.index))
		
		for col in pivot_val.columns:
			axes[0,i].plot(pivot_val.index, pivot_val[col], marker='o', label=col)
		axes[0,i].set_xlabel("shots")
		axes[0,i].set_ylabel("Top-1 accuracy")
		axes[0,i].set_title(f"{backbone} - Validation")
		axes[0,i].set_xscale('log' if max(pivot_val.index) / min([s for s in pivot_val.index if s>0]) > 8 else 'linear')
		axes[0,i].set_xticks(pivot_val.index)
		axes[0,i].grid(alpha=0.3)
		axes[0,i].legend()

		# Test plot (bottom row)
		test_data = backbone_data[backbone_data['split'] == 'test']
		pivot_test = test_data.pivot(index="shot", columns="model", values="top1")
		pivot_test = pivot_test.reindex(sorted(pivot_test.index))
		
		for col in pivot_test.columns:
			axes[1,i].plot(pivot_test.index, pivot_test[col], marker='o', label=col)
		axes[1,i].set_xlabel("shots")
		axes[1,i].set_ylabel("Top-1 accuracy")
		axes[1,i].set_title(f"{backbone} - Test")
		axes[1,i].set_xscale('log' if max(pivot_test.index) / min([s for s in pivot_test.index if s>0]) > 8 else 'linear')
		axes[1,i].set_xticks(pivot_test.index)
		axes[1,i].grid(alpha=0.3)
		axes[1,i].legend()

	plt.tight_layout()
	out_png = os.path.join(args.results_dir, "few_shot_plot_all_backbones.png")
	plt.savefig(out_png, dpi=180)
	print(f"Wrote plot -> {out_png}")

	# Also create a backbone comparison plot
	plt.figure(figsize=(10,8))
	plt.subplot(2,1,1)
	# Compare validation performance across backbones
	for backbone in args.backbones:
		data = df[(df['backbone'] == backbone) & (df['split'] == 'val') & (df['model'] == 'linear')]
		plt.plot(data['shot'], data['top1'], marker='o', label=backbone)
	plt.xlabel("shots")
	plt.ylabel("Linear probe top-1 accuracy")
	plt.title("Validation Set - Backbone Comparison")
	plt.xscale('log' if max(df['shot']) / min([s for s in df['shot'] if s>0]) > 8 else 'linear')
	plt.grid(alpha=0.3)
	plt.legend()

	plt.subplot(2,1,2)
	# Compare test performance across backbones
	for backbone in args.backbones:
		data = df[(df['backbone'] == backbone) & (df['split'] == 'test') & (df['model'] == 'linear')]
		plt.plot(data['shot'], data['top1'], marker='o', label=backbone)
	plt.xlabel("shots")
	plt.ylabel("Linear probe top-1 accuracy")
	plt.title("Test Set - Backbone Comparison")
	plt.xscale('log' if max(df['shot']) / min([s for s in df['shot'] if s>0]) > 8 else 'linear')
	plt.grid(alpha=0.3)
	plt.legend()

	plt.tight_layout()
	out_png = os.path.join(args.results_dir, "backbone_comparison_plot.png")
	plt.savefig(out_png, dpi=180)
	print(f"Wrote backbone comparison plot -> {out_png}")


if __name__ == "__main__":
	main()

