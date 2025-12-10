"""Evaluate delta_prompts.json only (net-new prompts) on 200-sample test subset."""
import json
import os
import torch
import torch.nn.functional as F
import clip
import pandas as pd
from clip_utils import get_text_embeddings

TEST_CSV = "/zfs/ai4good/datasets/mushroom/test.csv"
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


def predict_from_emb(image_emb, label_emb_dict, top_k=1):
    sims = {label: float(torch.cosine_similarity(image_emb.to(torch.float32), emb.to(torch.float32), dim=0))
            for label, emb in label_emb_dict.items()}
    sorted_labels = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    if top_k == 1:
        return sorted_labels[0][0]
    else:
        return [lbl for lbl, _ in sorted_labels[:top_k]]


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

    print(f'Evaluating delta_prompts.json (net-new prompts only)...')
    delta_embs = get_text_embeddings(labels, prompt_set='ensemble', model_name='ViT-B/32', device=device,
                                      include_common_names=True, common_prompts_path=DELTA_PROMPTS)
    delta_mean = {lbl: delta_embs[lbl]['label_embedding'] for lbl in labels}

    # scoring
    def score_all(mean_dict=None, bundle_dict=None, use_pooling=False, temp=TEMP, top_k=1):
        preds = []
        for img_emb in image_embeddings:
            if img_emb is None:
                preds.append(None)
                continue
            if not use_pooling:
                pred = predict_from_emb(img_emb, mean_dict, top_k=top_k)
            else:
                pred = predict_with_prompt_pooling(img_emb, bundle_dict, temp=temp, top_k=top_k)
            preds.append(pred)
        return preds

    def acc_from_preds(preds, trues, top_k=1):
        mask = [p is not None for p in preds]
        trues_m = [t for t, m in zip(trues, mask) if m]
        preds_m = [p for p, m in zip(preds, mask) if m]
        if top_k == 1:
            return sum(1 for p, t in zip(preds_m, trues_m) if p == t) / len(trues_m)
        else:
            # top-k: check if true label is in top-k predictions
            return sum(1 for p, t in zip(preds_m, trues_m) if t in p) / len(trues_m)

    # top-1
    preds_delta_mean = score_all(delta_mean, None, use_pooling=False, top_k=1)
    preds_delta_pool = score_all(None, delta_embs, use_pooling=True, temp=TEMP, top_k=1)
    acc_delta_mean = acc_from_preds(preds_delta_mean, true_labels, top_k=1)
    acc_delta_pool = acc_from_preds(preds_delta_pool, true_labels, top_k=1)

    # top-5
    preds_delta_mean_top5 = score_all(delta_mean, None, use_pooling=False, top_k=5)
    preds_delta_pool_top5 = score_all(None, delta_embs, use_pooling=True, temp=TEMP, top_k=5)
    acc_delta_mean_top5 = acc_from_preds(preds_delta_mean_top5, true_labels, top_k=5)
    acc_delta_pool_top5 = acc_from_preds(preds_delta_pool_top5, true_labels, top_k=5)

    report = {
        'n_samples': NUM_SAMPLES,
        'delta_net_new_mean_top1': acc_delta_mean,
        'delta_net_new_pool_top1': acc_delta_pool,
        'delta_net_new_mean_top5': acc_delta_mean_top5,
        'delta_net_new_pool_top5': acc_delta_pool_top5,
    }

    with open('delta_only_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print('\nDelta prompts evaluation results:')
    print(f'n_samples: {NUM_SAMPLES}')
    print(f'Top-1 mean: {acc_delta_mean:.4f}')
    print(f'Top-1 pooled: {acc_delta_pool:.4f}')
    print(f'Top-5 mean: {acc_delta_mean_top5:.4f}')
    print(f'Top-5 pooled: {acc_delta_pool_top5:.4f}')


if __name__ == '__main__':
    run()
