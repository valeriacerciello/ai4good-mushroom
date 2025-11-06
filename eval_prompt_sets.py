"""Evaluate several prompt sets (mean-prototype and per-prompt pooled) on the quick 200-sample test subset.

Produces a small JSON report and prints results.
"""
import json
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import clip
import pandas as pd
from clip_utils import get_text_embeddings, SHORT_PROMPTS


TEST_CSV = "/zfs/ai4good/datasets/mushroom/test.csv"
ENHANCED_IMG_PROMPTS = "enhanced_mushroom_prompts.json"
ENHANCED_COMMON_PROMPTS = "enhanced_with_common_names_prompts.json"
DELTA_PROMPTS = "delta_prompts.json"
NUM_SAMPLES = 200
TEMP = 0.12


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


def predict_from_emb(image_emb, label_emb_dict):
    sims = {label: float(torch.cosine_similarity(image_emb.to(torch.float32), emb.to(torch.float32), dim=0))
            for label, emb in label_emb_dict.items()}
    return max(sims.items(), key=lambda x: x[1])[0]


def predict_with_prompt_pooling(image_emb, embeddings_bundle, temp=0.12):
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
    return max(scores.items(), key=lambda x: x[1])[0]


def build_union_prompts(labels, files):
    # files: list of json file paths to load per-label prompts
    union = {lbl: [] for lbl in labels}
    # add v1 templates first
    for lbl in labels:
        for t in SHORT_PROMPTS:
            union[lbl].append(t.format(label=lbl))
    # load each file and append prompts if present
    for f in files:
        if not os.path.exists(f):
            continue
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        for lbl in labels:
            if lbl in data:
                for p in data[lbl]:
                    if p not in union[lbl]:
                        union[lbl].append(p)
    return union


def encode_prompt_bundle_manual(bundle, model, device, batch_size=128):
    # bundle: dict[label] -> list[str]
    results = {}
    for label, prompts in bundle.items():
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                toks = clip.tokenize(prompts[i:i+batch_size]).to(device)
                emb = model.encode_text(toks)
                emb = F.normalize(emb, dim=-1)
                all_embs.append(emb.cpu())
        if all_embs:
            pe = torch.cat(all_embs, dim=0)
            le = F.normalize(pe.mean(dim=0), dim=-1)
        else:
            pe = torch.empty((0, model.text_projection.shape[1]))
            le = torch.zeros((model.text_projection.shape[1],))
        results[label] = {'prompts': prompts, 'prompt_embeddings': pe, 'label_embedding': le}
    return results


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    df = load_test_subset(TEST_CSV, NUM_SAMPLES)
    image_paths = df['image_path'].tolist()
    true_labels = df['label'].tolist()

    print("Encoding images...")
    image_embeddings = encode_images(model, preprocess, image_paths, device)

    labels = sorted(list(set(true_labels)))

    report = {}

    # 1) v1
    print('Evaluating v1 templates...')
    v1 = get_text_embeddings(labels, prompt_set='v1', model_name='ViT-B/32', device=device)
    v1_mean = {lbl: v1[lbl]['label_embedding'] for lbl in labels}
    v1_pool = v1  # per-prompt available

    # 2) enhanced_mushroom_prompts.json (image-derived)
    print('Evaluating image-derived prompts (enhanced_mushroom_prompts.json)...')
    path_img = ENHANCED_IMG_PROMPTS
    new_img = get_text_embeddings(labels, prompt_set='ensemble', model_name='ViT-B/32', device=device, include_common_names=True, common_prompts_path=path_img)
    new_img_mean = {lbl: new_img[lbl]['label_embedding'] for lbl in labels}

    # 3) enhanced_with_common_names_prompts.json
    print('Evaluating enhanced_with_common_names_prompts.json...')
    path_common = ENHANCED_COMMON_PROMPTS
    new_common = get_text_embeddings(labels, prompt_set='ensemble', model_name='ViT-B/32', device=device, include_common_names=True, common_prompts_path=path_common)
    new_common_mean = {lbl: new_common[lbl]['label_embedding'] for lbl in labels}

    # 4) delta_prompts.json (net-new prompts only)
    print('Evaluating delta_prompts.json (net new)...')
    path_delta = DELTA_PROMPTS
    delta_embs = get_text_embeddings(labels, prompt_set='ensemble', model_name='ViT-B/32', device=device, include_common_names=True, common_prompts_path=path_delta)
    delta_mean = {lbl: delta_embs[lbl]['label_embedding'] for lbl in labels}

    # 5) ensemble union = v1 + enhanced_mushroom + enhanced_with_common
    print('Building explicit ensemble union and encoding...')
    union_prompts = build_union_prompts(labels, [ENHANCED_IMG_PROMPTS, ENHANCED_COMMON_PROMPTS])
    union_embs = encode_prompt_bundle_manual(union_prompts, model, device)
    union_mean = {lbl: union_embs[lbl]['label_embedding'] for lbl in labels}

    # scoring helpers
    def score_all(mean_dict, bundle_dict=None, use_pooling=False, temp=TEMP):
        preds = []
        for img_emb in image_embeddings:
            if img_emb is None:
                preds.append(None)
                continue
            if not use_pooling:
                pred = predict_from_emb(img_emb, mean_dict)
            else:
                pred = predict_with_prompt_pooling(img_emb, bundle_dict, temp=temp)
            preds.append(pred)
        return preds

    # Evaluate each set: report mean-prototype accuracy and pooled accuracy (if available)
    def acc_from_preds(preds, trues):
        mask = [p is not None for p in preds]
        trues_m = [t for t,m in zip(trues, mask) if m]
        preds_m = [p for p,m in zip(preds, mask) if m]
        return sum(1 for p,t in zip(preds_m, trues_m) if p==t)/len(trues_m)

    # v1
    preds_v1_mean = score_all(v1_mean, None, use_pooling=False)
    preds_v1_pool = score_all(None, v1_pool, use_pooling=True, temp=TEMP)
    acc_v1_mean = acc_from_preds(preds_v1_mean, true_labels)
    acc_v1_pool = acc_from_preds(preds_v1_pool, true_labels)

    # image-derived
    preds_img_mean = score_all(new_img_mean, None, use_pooling=False)
    preds_img_pool = score_all(None, new_img, use_pooling=True, temp=TEMP)
    acc_img_mean = acc_from_preds(preds_img_mean, true_labels)
    acc_img_pool = acc_from_preds(preds_img_pool, true_labels)

    # common-name enhanced
    preds_common_mean = score_all(new_common_mean, None, use_pooling=False)
    preds_common_pool = score_all(None, new_common, use_pooling=True, temp=TEMP)
    acc_common_mean = acc_from_preds(preds_common_mean, true_labels)
    acc_common_pool = acc_from_preds(preds_common_pool, true_labels)

    # delta (net-new only)
    preds_delta_mean = score_all(delta_mean, None, use_pooling=False)
    preds_delta_pool = score_all(None, delta_embs, use_pooling=True, temp=TEMP)
    acc_delta_mean = acc_from_preds(preds_delta_mean, true_labels)
    acc_delta_pool = acc_from_preds(preds_delta_pool, true_labels)

    # union ensemble
    preds_union_mean = score_all(union_mean, None, use_pooling=False)
    preds_union_pool = score_all(None, union_embs, use_pooling=True, temp=TEMP)
    acc_union_mean = acc_from_preds(preds_union_mean, true_labels)
    acc_union_pool = acc_from_preds(preds_union_pool, true_labels)

    report = {
        'n_samples': NUM_SAMPLES,
        'v1_mean': acc_v1_mean,
        'v1_pool': acc_v1_pool,
        'image_derived_mean': acc_img_mean,
        'image_derived_pool': acc_img_pool,
        'common_enhanced_mean': acc_common_mean,
        'common_enhanced_pool': acc_common_pool,
        'delta_net_new_mean': acc_delta_mean,
        'delta_net_new_pool': acc_delta_pool,
        'union_mean': acc_union_mean,
        'union_pool': acc_union_pool,
    }

    with open('prompt_sets_report.json','w') as f:
        json.dump(report, f, indent=2)

    print('\nPrompt sets evaluation results:')
    for k,v in report.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    run()
