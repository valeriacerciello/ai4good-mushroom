"""
Minimal CoOp-style prompt tuning implementation (soft prompt vectors) for CLIP ViT-B/32.
This is a simplified implementation intended as a starting point. Running it on CPU will be slow.

Usage example (GPU recommended):
python3 coop_prompt_tune.py --epochs 5 --lr 1e-3 --num_ctx 8 --out tuned_prompts.json

Notes:
- This script learns `num_ctx` continuous prompt vectors per class (or shared) and optimizes them with cross-entropy
  using frozen CLIP image/text encoders.
- For production use prefer a tested CoOp implementation and GPU.
"""
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import clip
from PIL import Image
from fusion_clip_resnet import load_split_df, load_labels
import numpy as np


def build_data(df, preprocess, max_samples=None):
    paths = df['image_path'].tolist()
    labels = df['label'].tolist()
    if max_samples:
        paths = paths[:max_samples]
        labels = labels[:max_samples]
    images = []
    for p in paths:
        try:
            img = Image.open(p).convert('RGB')
            images.append(preprocess(img))
        except Exception:
            images.append(torch.zeros((3, 224, 224)))
    images = torch.stack(images, dim=0)
    return images, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_ctx', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out', type=str, default='tuned_prompts.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_train', type=int, default=2000)
    args = parser.parse_args()

    device = args.device
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    df_train = load_split_df('train', limit=args.max_train)
    df_val = load_split_df('val', limit=200)
    labels = load_labels([df_train, df_val])

    # Build train images and label indices
    images, lbls = build_data(df_train, preprocess, max_samples=args.max_train)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y = torch.tensor([label_to_idx[l] for l in lbls], dtype=torch.long)

    # Initialize continuous prompt vectors (shared across classes for simplicity)
    ctx = nn.Parameter(torch.randn((args.num_ctx, model.transformer.width), device=device) * 0.01)
    optimizer = optim.Adam([ctx], lr=args.lr)

    # Build text token prefix/template: we will insert ctx before class name tokens
    # For simplicity, we will use raw class names tokenized and obtain their text embeddings once.
    text_tokens = [clip.tokenize(c) for c in labels]
    text_tokens = torch.cat(text_tokens).to(device)
    with torch.no_grad():
        text_embed = model.encode_text(text_tokens).detach()  # (C, D)

    # Training loop (very simplified): encode images, combine ctx with text_embed via dot product
    dataset = torch.utils.data.TensorDataset(images, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(xb).float()
                img_feat = F.normalize(img_feat, dim=1)
            # Combine ctx and text_embed by a simple additive scheme (NOT exact CoOp)
            # Project ctx to text embedding dim by average
            ctx_proj = ctx.mean(dim=0, keepdim=True)  # (1, D)
            # Create new text prototypes by adding ctx_proj to text_embed
            text_mod = text_embed + ctx_proj
            text_mod = F.normalize(text_mod, dim=1)
            logits = img_feat @ text_mod.t()
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(dataset)
        print(f'Epoch {epoch+1}/{args.epochs} loss={avg:.4f}')

    # Save ctx vectors (cpu)
    out = {'num_ctx': args.num_ctx, 'ctx': ctx.detach().cpu().numpy().tolist()}
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print('Saved tuned prompts to', args.out)

if __name__ == '__main__':
    main()
