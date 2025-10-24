"""Evaluate several prompt sets (mean-prototype and per-prompt pooled) on the quick 200-sample test subset.

Produces a small JSON report and prints results.
"""
import json
from pathlib import Path
import os
import torch
import torch.nn.functional as F
# import clip
import open_clip
import pandas as pd
from scripts.clip_utils import get_text_embeddings, SHORT_PROMPTS

import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score


# TEST_CSV = "/zfs/ai4good/datasets/mushroom/test.csv"
DELTA_PROMPTS = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/data_prompts_label/delta_prompts.json"
NUM_SAMPLES = 200
TEMP = 0.12
TRAIN_CSV = "/zfs/ai4good/datasets/mushroom/train.csv"

def encode_images(model, preprocess, image_paths, device):
    images = []
    for p in image_paths:
        # local = p.replace('/kaggle/working/', '/zfs/ai4good/datasets/mushroom/')
        local = '/zfs/ai4good/datasets/mushroom/merged_dataset/' + p
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

def encode_prompt_bundle_manual(bundle, model, device, batch_size=128):
    # bundle: dict[label] -> list[str]
    results = {}
    for label, prompts in bundle.items():
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                # toks = clip.tokenize(prompts[i:i+batch_size]).to(device)
                tokenizer = open_clip.get_tokenizer(model)
                toks = tokenizer(prompts[i:i+batch_size]).to(device)
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


def build_few_shot_prototypes(labels, train_csv, shots, model, preprocess, device, random_state=42):
    """Return dict[label] -> prototype embedding (cpu tensor) built from up to `shots` train images per label.

    Exposed as a top-level function for import.
    """
    # load train csv
    df_train = pd.read_csv(train_csv)
    prototypes = {}
    for lbl in labels:
        df_lbl = df_train[df_train['label'] == lbl]
        if df_lbl.shape[0] == 0:
            # empty prototype
            prototypes[lbl] = torch.zeros(model.text_projection.shape[1])
            continue
        # sample up to shots rows
        n = min(shots, len(df_lbl))
        df_samp = df_lbl.sample(n=n, random_state=random_state)
        img_paths = df_samp['filepath'].tolist()
        embs = encode_images(model, preprocess, img_paths, device)
        # filter None
        embs = [e for e in embs if e is not None]
        if not embs:
            prototypes[lbl] = torch.zeros(model.text_projection.shape[1])
            continue
        stacked = torch.stack(embs, dim=0)
        proto = F.normalize(stacked.mean(dim=0), dim=-1)
        prototypes[lbl] = proto.cpu()
    return prototypes


def evaluate_prompts_and_few_shot(labels, train_csv, split_csv, shots=(1,5,10,20,50), model_name='ViT-B/32', device='cpu', num_samples=200, temp=0.12, delta_prompts_path=None, pretrained="openai", random_state=42):
    """Evaluate prompt sets (delta prompts) and few-shot image prototypes.

    Returns a dict with keys: n_samples, delta_net_new_mean, delta_net_new_pool, few_shot_results (dict shot->acc).
    This is importable and used by other scripts.
    """
    device = device
    # model, preprocess = clip.load(model_name, device)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)


    # load small split_csv subset
    df = pd.read_csv(split_csv)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df = df.head(num_samples) # TODO: maybe remove?
    image_paths = df['filepath'].tolist()
    true_labels = df['label'].tolist()

    image_embeddings = encode_images(model, preprocess, image_paths, device)

    labels = list(labels)

    # delta prompts
    path_delta = delta_prompts_path or DELTA_PROMPTS
    delta_embs = get_text_embeddings(labels, prompt_set='ensemble', model_name=model_name, device=device, include_common_names=True, common_prompts_path=path_delta, pretrained=pretrained)
    delta_mean = {lbl: delta_embs[lbl]['label_embedding'] for lbl in labels}

    def score_all(mean_dict, bundle_dict=None, use_pooling=False, temp=temp):
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

    def acc_from_preds(preds, trues):
        mask = [p is not None for p in preds]
        trues_m = [t for t,m in zip(trues, mask) if m]
        preds_m = [p for p,m in zip(preds, mask) if m]
        return sum(1 for p,t in zip(preds_m, trues_m) if p==t)/len(trues_m)

    preds_delta_mean = score_all(delta_mean, None, use_pooling=False)
    preds_delta_pool = score_all(None, delta_embs, use_pooling=True, temp=temp)
    acc_delta_mean = acc_from_preds(preds_delta_mean, true_labels)
    acc_delta_pool = acc_from_preds(preds_delta_pool, true_labels)

    few_shot_results = {}
    for k in shots:
        proto_dict = build_few_shot_prototypes(labels, train_csv, k, model, preprocess, device, random_state=random_state)
        preds_proto = score_all(proto_dict, None, use_pooling=False)
        acc_proto = acc_from_preds(preds_proto, true_labels)
        few_shot_results[int(k)] = acc_proto

    return {
        'n_samples': num_samples,
        'delta_net_new_mean': acc_delta_mean,
        'delta_net_new_pool': acc_delta_pool,
        'few_shot_results': few_shot_results
    }