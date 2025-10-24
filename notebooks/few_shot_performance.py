#!/usr/bin/env python3
import os
# Use student directory for Hugging Face cache to avoid disk quota issues
os.environ["HF_HOME"] = "/home/c/dkorot/AI4GOOD/provided_dir/student/dkorot/.cache/huggingface"

"""Run few-shot performance experiments and plot accuracy vs shots.

This script expects cached CLIP features in the same format used by
`scripts/dump_features.py`: features/<backbone>/<split>.npz containing X, y, paths.

It runs:
 - Zero-shot using text prompts (requires open_clip)
 - Prototype classifier using k-shot supports
 - Linear probe trained on k-shot supports

Outputs:
 - CSV table with metrics
 - PNG plot accuracy vs shots comparing zero-shot, prototype, and linear"""
import json
import argparse
"""
Outputs:
 - CSV table with metrics
 - PNG plot accuracy vs shots comparing zero-shot, prototype, and linear
"""
from pathlib import Path
import sys
import numpy as np
import torch
import open_clip
import matplotlib.pyplot as plt
import csv
import json
import pandas as pd
from sklearn.metrics import f1_score
from scripts.few_shot import ensure_features, load_labels, prototype_classifier, train_linear_probe, topk_acc, balanced_acc
from scripts.prompt_utils import DELTA_PROMPTS, load_prompts
# Import prompt-based few-shot evaluator
from scripts.new_few_shots import evaluate_prompts_and_few_shot


def zero_shot_scores(label_names, backbone, pretrained, device, prompts_path=None):
	print("creating model")
	model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, device=device)
	print("evaluatemodel")
	model.eval()
	print("get token")
	tokenizer = open_clip.get_tokenizer(backbone)

	# Load prompts from DELTA_PROMPTS (JSON) for each label
	if prompts_path is None:
		prompts_path = DELTA_PROMPTS
	label_prompts = load_prompts(prompts_path, label_names)

	# Initialize storage for embeddings
	all_embeddings = []
	label_indices = []  # Track which label each embedding belongs to

	print("start encoding text")
	with torch.no_grad():
		for idx, label in enumerate(label_names):
			prompts = label_prompts[label]
			T = tokenizer(prompts).to(device)
			# print(f"encode text for {label} ({len(prompts)} prompts)")
			text = model.encode_text(T)
			text = text / text.norm(dim=-1, keepdim=True)
			all_embeddings.append(text.cpu())
			label_indices.extend([idx] * len(prompts))

		# Stack all embeddings and compute mean per label
		all_embeddings = torch.cat(all_embeddings, dim=0)
		label_indices = torch.tensor(label_indices)

		# Compute mean embedding per label
		mean_embeddings = torch.zeros(len(label_names), all_embeddings.shape[1])
		for i in range(len(label_names)):
			mask = (label_indices == i)
			if mask.any():
				mean_embeddings[i] = all_embeddings[mask].mean(0)

		# Normalize means and convert to numpy
		mean_embeddings = mean_embeddings / mean_embeddings.norm(dim=-1, keepdim=True)
		mean_embeddings = mean_embeddings.numpy().T  # [D, K]

	return mean_embeddings

pretrained_dict = {"ViT-B-32-quickgelu": "openai",
					"ViT-L-14": "openai",
					"ViT-B-16": "openai",
					"ViT-H-14-378-quickgelu": "dfn5b",
					"ViT-gopt-16-SigLIP2-384": "webli",
					"PE-Core-bigG-14-448": "meta"
					}
class Args:
    data_root = "/home/c/dkorot/AI4GOOD/provided_dir/datasets/mushroom/merged_dataset"
    train_csv = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/train.csv"
    val_csv = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/val.csv"
    test_csv = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/splits/test.csv"
    labels = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/data_prompts_label/labels.tsv"
    backbones = list(pretrained_dict.keys())
    pretrained = pretrained_dict
    shots = [1]
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
	X_tr, y_tr, _ = ensure_features("train", backbone, args.data_root, args.train_csv, args.labels, save_dir=args.save_dir, pretrained=args.pretrained[backbone])
	X_vl, y_vl, _ = ensure_features("val",   backbone, args.data_root, args.val_csv,   args.labels, save_dir=args.save_dir, pretrained=args.pretrained[backbone])
	X_te, y_te, _ = ensure_features("test",  backbone, args.data_root, args.test_csv,  args.labels, save_dir=args.save_dir, pretrained=args.pretrained[backbone])

	# zero-shot text embeddings for this backbone
	print("~~~~~~~~~~~~~0 shot~~~~~~~~~~~~~~~~")
	text_embeddings = zero_shot_scores(label_names, backbone, args.pretrained[backbone], args.device, args.prompts_path)

	backbone_records = []
	
	def evaluate_split(split_name, X, y):
		split_records = []
		print(f"~~~~~~~~~~~~~Evaluating {split_name} for {backbone}~~~~~~~~~~~~~~~~")
		
		# if 0 in shots then compute zero-shot baseline
		if 0 in args.shots:
			scores_zs = X @ text_embeddings
			yhat_zs = np.argmax(scores_zs, axis=1)
			zs_top1 = (yhat_zs == y).mean()
			zs_bal = balanced_acc(y, yhat_zs, K)
			zs_top5 = topk_acc(y, scores_zs, 5)
			zs_macro = f1_score(y, yhat_zs, average='macro')
			split_records.append((0, "zero-shot", split_name, backbone, zs_top1, zs_top5, zs_bal, zs_macro))
			print(f">>>>>>>>>>> {split_name} shot=0: zs top1={zs_top1:.4f}")
		
		# print(f"~~~~~~~~~~~~~{split_name} few shot~~~~~~~~~~~~~~~~")
		
		for shot in sorted([s for s in args.shots if s>0]):
			print(f"~~~~~~~~~~~~~{split_name} shot number: {shot}~~~~~~~~~~~~~~~~")
			# sample supports
			print("sample supports")
			rng = np.random.default_rng(args.seed + int(shot))
			from scripts.few_shot import sample_few_shot_indices
			sup_idx, sup_labels = sample_few_shot_indices(y_tr, K, shot, rng)
			X_sup = X_tr[sup_idx]
			y_sup = sup_labels

			# prototype
			print("prototype")
			prototypes = prototype_classifier(X_sup, y_sup, K)
			scores_proto = X @ prototypes.T
			yhat_proto = np.argmax(scores_proto, axis=1)
			proto_top1 = (yhat_proto == y).mean()
			proto_bal = balanced_acc(y, yhat_proto, K)
			proto_top5 = topk_acc(y, scores_proto, 5)
			proto_macro = f1_score(y, yhat_proto, average='macro')
			split_records.append((shot, "prototype", split_name, backbone, proto_top1, proto_top5, proto_bal, proto_macro))

			# linear probe
			print("linear prob")
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

			print(f">>>>>>>>>>> {split_name} shot={shot}: proto top1={proto_top1:.4f}, lin top1={lin_top1:.4f}")
		
		return split_records

	# Run evaluation for validation and test sets
	print("evaluate splits")
	for split_name, (X, y) in [("val", (X_vl, y_vl)), ("test", (X_te, y_te))]:
		data_split = args.val_csv if split_name == "val" else args.test_csv
		split_records = evaluate_split(split_name, X, y)
		backbone_records.extend(split_records)
		# For the test and val split, also run the prompt-based few-shot evaluator (if applicable)
		if backbone is not None:
			print(f"Running prompt-based few-shot eval for backbone {backbone}...")
			prompt_results = evaluate_prompts_and_few_shot(label_names, args.train_csv, data_split, shots=args.shots, model_name=backbone, device=args.device, num_samples=200, delta_prompts_path=args.prompts_path, random_state=args.seed, pretrained=args.pretrained[backbone])
			# Append prompt-based prototype rows to backbone_records. Use best-effort placeholders for top5/balanced/macro.
			for k, acc in sorted(prompt_results.get('few_shot_results', {}).items()):
				backbone_records.append((int(k), 'prompt-prototype', split_name, backbone, float(acc), float(acc), float(acc), float(acc)))
			# Also add delta prompt zero-shot entries
			backbone_records.append((0, 'delta-net-new-mean', split_name, backbone, float(prompt_results.get('delta_net_new_mean', 0.0)), float(prompt_results.get('delta_net_new_mean', 0.0)), float(prompt_results.get('delta_net_new_mean', 0.0)), float(prompt_results.get('delta_net_new_mean', 0.0))))
			backbone_records.append((0, 'delta-net-new-pool', split_name, backbone, float(prompt_results.get('delta_net_new_pool', 0.0)), float(prompt_results.get('delta_net_new_pool', 0.0)), float(prompt_results.get('delta_net_new_pool', 0.0)), float(prompt_results.get('delta_net_new_pool', 0.0))))
		else:
			print(f"No matching CLIP model mapping for backbone {backbone}; skipping prompt-based evaluation.")
	
	return backbone_records

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data-root", required=True)
	ap.add_argument("--train-csv", required=True)
	ap.add_argument("--test-csv", required=True)
	ap.add_argument("--labels", required=True)
	ap.add_argument("--backbones", nargs="+", default=["ViT-B-32"])
	ap.add_argument("--pretrained", default={"ViT-B-32": "openai"})
	ap.add_argument("--shots", nargs="+", type=int, default=[0,1,5,10,20])
	ap.add_argument("--save-dir", default="features")
	ap.add_argument("--results-dir", default="results")
	ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--epochs", type=int, default=200)
	ap.add_argument("--lr", type=float, default=1e-2)
	ap.add_argument("--batch-size", type=int, default=32)
	ap.add_argument("--prompts-path", help="Path to JSON file containing delta prompts for each class")
	# args = ap.parse_args()
	print("~~~~~~~~~~~~~set up~~~~~~~~~~~~~~~~")
	args = Args()

	torch.manual_seed(args.seed); np.random.seed(args.seed)

	Path(args.results_dir).mkdir(parents=True, exist_ok=True)

	label_names, name2id = load_labels(args.labels)
	K = len(label_names)

	# Use DELTA_PROMPTS for zero-shot prompt loading
	args.prompts_path = DELTA_PROMPTS

	# Evaluate all backbones
	all_records = []
	for backbone in args.backbones:
		backbone_records = evaluate_backbone(backbone, args, label_names, K)
		all_records.extend(backbone_records)

	# `all_records` already contains per-backbone results collected above

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


if __name__ == "__main__":
	main()

