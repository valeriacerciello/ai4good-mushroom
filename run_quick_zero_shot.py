"""
Quick zero-shot comparison:
- Loads CLIP model
- Loads a small subset of test images and true labels
- Compares two text embedding strategies:
    A) Original averaged prompts from `mushroom_prompts.json` (already present)
    B) New embeddings from `clip_utils.get_text_embeddings(labels, prompt_set)`
- Reports accuracy and top-1 changes
"""

import json
import random
import torch
import torch.nn.functional as F
from clip_utils import get_text_embeddings
from clip_mushroom_classifier import CLIPMushroomClassifier
import clip
import pandas as pd
import os

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"
TEST_CSV = "/zfs/ai4good/datasets/mushroom/test.csv"
PROMPTS_FILE = "mushroom_prompts.json"
CLEANED_PROMPTS_FILE = "enhanced_with_common_names_prompts_clean.json"
DELTA_CSV = "zero_shot_per_class_delta.csv"
CHANGES_JSON = "zero_shot_prediction_changes.json"
NUM_SAMPLES = 200  # small subset for quick run


def load_test_subset(csv_file, n=200):
    df = pd.read_csv(csv_file)
    # Shuffle and sample n rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df.head(n)


def encode_images(model, preprocess, image_paths):
    images = []
    for p in image_paths:
        local = p.replace('/kaggle/working/', '/zfs/ai4good/datasets/mushroom/')
        if os.path.exists(local):
            from PIL import Image
            im = Image.open(local).convert('RGB')
            img_t = preprocess(im).unsqueeze(0).to(DEVICE)
            images.append(img_t)
        else:
            images.append(None)
    # Batch encode where possible
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


def compute_accuracy(preds, trues):
    corr = sum(1 for p,t in zip(preds,trues) if p==t)
    return corr / len(trues)


def main():
    print("Loading CLIP model...")
    model, preprocess = clip.load(MODEL_NAME, DEVICE)

    # Load test subset
    df = load_test_subset(TEST_CSV, NUM_SAMPLES)
    image_paths = df['image_path'].tolist()
    true_labels = df['label'].tolist()

    # Encode images
    print("Encoding images (quick)...")
    image_embeddings = encode_images(model, preprocess, image_paths)

    # Load original prompts (averaged) - using classifier util
    print("Loading original averaged prompts...")
    with open(PROMPTS_FILE, 'r') as f:
        orig_prompts = json.load(f)

    # Build original label embeddings by averaging all prompts (same as earlier code)
    orig_label_embeddings = {}
    for label, prompts in orig_prompts.items():
        # tokenize and encode in small batches
        toks = clip.tokenize(prompts).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_text(toks)
            emb = F.normalize(emb, dim=-1)
            avg = F.normalize(emb.mean(dim=0), dim=-1)
            orig_label_embeddings[label] = avg.cpu()

    # Build new embeddings: short prompts (v1)
    labels = sorted(list(set(true_labels)))
    print(f"Computing new embeddings for {len(labels)} labels (v1)...")
    new_embs_v1 = get_text_embeddings(labels, prompt_set='v1', model_name=MODEL_NAME, device=DEVICE)

    # And ensemble (including common names file)
    print(f"Computing new embeddings for {len(labels)} labels (ensemble)...")
    new_embs_ens = get_text_embeddings(labels, prompt_set='ensemble', model_name=MODEL_NAME, device=DEVICE)
    print(f"Computing new embeddings for {len(labels)} labels (ensemble+common)...")
    # Prefer cleaned prompts if available
    common_path = CLEANED_PROMPTS_FILE if os.path.exists(CLEANED_PROMPTS_FILE) else None
    new_embs_ens_common = get_text_embeddings(labels, prompt_set='ensemble', model_name=MODEL_NAME, device=DEVICE, include_common_names=True, common_prompts_path=common_path)

    # Helper to predict from image embedding using label embeddings dict (cpu tensors)
    def predict_from_emb(image_emb, label_emb_dict):
        sims = {}
        for label, emb in label_emb_dict.items():
            # emb may be CPU tensor
            sims[label] = float(torch.cosine_similarity(image_emb.to(torch.float32), emb.to(torch.float32), dim=0))
        # Return top label
        return max(sims.items(), key=lambda x: x[1])[0]

    # Per-prompt softmax-pooling predictor: for each class gather its prompt embeddings and pool
    def predict_with_prompt_pooling(image_emb, embeddings_bundle, temp=0.1):
        # embeddings_bundle: dict[label] -> {'prompts': [...], 'prompt_embeddings': tensor (P x D)}
        # We'll compute per-prompt similarities and do a softmax over prompts per class
        scores = {}
        for label, info in embeddings_bundle.items():
            pe = info['prompt_embeddings']  # CPU tensor P x D
            if pe.numel() == 0:
                scores[label] = -1e9
                continue
            # compute cosine similarities between image_emb and each prompt emb
            sims = torch.cosine_similarity(image_emb.unsqueeze(0), pe.to(image_emb.dtype), dim=1)
            # apply softmax pooling over prompts for this class
            w = torch.softmax(sims / temp, dim=0)
            pooled = float((w * sims).sum())
            scores[label] = pooled
        return max(scores.items(), key=lambda x: x[1])[0]

    # Prepare label embeddings dicts for prediction (CPU tensors)
    # For original, ensure all labels present
    orig_labels_used = {label: orig_label_embeddings[label] for label in labels if label in orig_label_embeddings}
    new_v1_used = {label: new_embs_v1[label]['label_embedding'] for label in labels}
    new_ens_used = {label: new_embs_ens[label]['label_embedding'] for label in labels}
    new_ens_common_used = {label: new_embs_ens_common[label]['label_embedding'] for label in labels}

    # Make predictions
    preds_orig = []
    preds_v1 = []
    preds_ens = []
    preds_ens_common = []

    for img_emb in image_embeddings:
        if img_emb is None:
            preds_orig.append(None)
            preds_v1.append(None)
            preds_ens.append(None)
            preds_ens_common.append(None)
            continue
        img_cpu = img_emb
        preds_orig.append(predict_from_emb(img_cpu, orig_labels_used))
        preds_v1.append(predict_from_emb(img_cpu, new_v1_used))
        preds_ens.append(predict_from_emb(img_cpu, new_ens_used))
        # per-prompt pooling predictions (ensemble + common names)
        try:
            ppool = predict_with_prompt_pooling(img_cpu, new_embs_ens_common, temp=0.12)
        except Exception:
            ppool = predict_from_emb(img_cpu, new_ens_common_used)
        preds_ens_common.append(ppool)

    # Compute accuracies only on non-None predictions
    mask = [p is not None for p in preds_orig]
    trues_masked = [t for t,m in zip(true_labels, mask) if m]
    orig_masked = [p for p,m in zip(preds_orig, mask) if m]
    v1_masked = [p for p,m in zip(preds_v1, mask) if m]
    ens_masked = [p for p,m in zip(preds_ens, mask) if m]
    ens_common_masked = [p for p,m in zip(preds_ens_common, mask) if m]

    acc_orig = sum(1 for p,t in zip(orig_masked, trues_masked) if p==t)/len(trues_masked)
    acc_v1 = sum(1 for p,t in zip(v1_masked, trues_masked) if p==t)/len(trues_masked)
    acc_ens = sum(1 for p,t in zip(ens_masked, trues_masked) if p==t)/len(trues_masked)
    acc_ens_common = sum(1 for p,t in zip(ens_common_masked, trues_masked) if p==t)/len(trues_masked)

    print("\n=== Quick Zero-Shot Report ===")
    print(f"Samples evaluated: {len(trues_masked)}")
    print(f"Original averaged prompts accuracy: {acc_orig:.4f}")
    print(f"Short prompts (v1) accuracy: {acc_v1:.4f}")
    print(f"Ensemble prompts accuracy: {acc_ens:.4f}")
    print(f"Ensemble + common-name prompts (pooled) accuracy: {acc_ens_common:.4f}")

    # Save report
    report = {
        'n_samples': len(trues_masked),
        'acc_original': acc_orig,
        'acc_v1': acc_v1,
        'acc_ensemble': acc_ens,
        'acc_ensemble_common_pooled': acc_ens_common
    }
    with open('quick_zero_shot_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("Report saved to quick_zero_shot_report.json")

    # Produce per-class delta CSV and example changes
    per_class = {}
    for label in labels:
        per_class[label] = {'orig_correct':0, 'ens_common_correct':0, 'total':0}

    changes = []
    for img_path, true, orig_p, ensc_p in zip(image_paths, trues_masked, orig_masked, ens_common_masked):
        per_class[true]['total'] += 1
        if orig_p == true:
            per_class[true]['orig_correct'] += 1
        if ensc_p == true:
            per_class[true]['ens_common_correct'] += 1
        if orig_p != ensc_p:
            changes.append({'image_path': img_path, 'true': true, 'orig_pred': orig_p, 'ens_common_pred': ensc_p})

    # write CSV
    import csv
    with open(DELTA_CSV, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['label','total','orig_correct','ens_common_correct','orig_acc','ens_common_acc','delta'])
        for k,v in per_class.items():
            total = v['total']
            if total == 0:
                continue
            orig_acc = v['orig_correct']/total
            ens_acc = v['ens_common_correct']/total
            w.writerow([k,total,v['orig_correct'],v['ens_common_correct'],f"{orig_acc:.4f}",f"{ens_acc:.4f}",f"{ens_acc-orig_acc:.4f}"])

    with open(CHANGES_JSON, 'w') as cf:
        json.dump(changes, cf, indent=2)

    print(f"Wrote per-class delta CSV: {DELTA_CSV}")
    print(f"Wrote prediction changes JSON: {CHANGES_JSON}")

if __name__ == '__main__':
    main()
