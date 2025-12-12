#====================================================pkgs=============================================================
#os: Operating system interface for file path manipulation, directory creation, and environment variable configuration
#time: Timing functions for tracking training duration and performance metrics
#json: JSON file parsing and serialization for loading prompt templates
#PIL.Image: Image processing library for loading, converting, and handling RGB image data
#torch.nn.functional: Neural network operations (loss functions, activations)
#torch.utils.data: Data loading utilities (Dataset abstraction, DataLoader for batching)
#torchvision.transforms: Image preprocessing, augmentation, and normalization pipelines
#torch.backends.cudnn: CUDA Deep Neural Network library backend for GPU acceleration optimizations
#open_clip: Open-source CLIP implementation providing vision-language models and tokenizers
#peft: Parameter-Efficient Fine-Tuning library
#--LoraConfig, get_peft_model: LoRA (Low-Rank Adaptation) configuration and model adaptation
#sklearn.metrics: 
#--f1_score, accuracy_score: Classification performance metrics
#--confusion_matrix: Error analysis and class confusion visualization

import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import open_clip
from peft import LoraConfig, get_peft_model

# ======================user settings=============================
ROOT_MUSHROOM = "/zfs/ai4good/datasets/mushroom"
TEST_CSV = os.path.join(ROOT_MUSHROOM, "test.csv")
CHECKPOINT_DIR = "lora_stable_fixed_b32"
CKPT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
MERGED_DATA = "/zfs/ai4good/datasets/mushroom/merged_dataset"

BATCH_SIZE = 128           
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED_CHECKPOINT = "laion400m_e32"

# LoRA config
LORA_TARGET_MODULES = [
    "visual.transformer.resblocks.10.attn.in_proj",
    "visual.transformer.resblocks.10.attn.out_proj",
    "visual.transformer.resblocks.11.attn.in_proj",
    "visual.transformer.resblocks.11.attn.out_proj",
    "visual.transformer.resblocks.10.mlp.c_fc",
    "visual.transformer.resblocks.10.mlp.c_proj",
    "visual.transformer.resblocks.11.mlp.c_fc",
    "visual.transformer.resblocks.11.mlp.c_proj",
]
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.1

NORMALIZE_MEAN = [0.4815, 0.4578, 0.4082]
NORMALIZE_STD = [0.2686, 0.2613, 0.2758]

# ============= transforms & dataset ===========
def test_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])

class FixedMushroomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, class_mapping=None):
        df = pd.read_csv(csv_file, header=0).iloc[:, :2]
        df.columns = ["path", "label"]
        df['label'] = df['label'].astype(str)

        if class_mapping is None:
            classes = sorted(df['label'].unique())
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
            self.classes = classes
        else:
            self.class_to_idx = class_mapping
            self.classes = list(class_mapping.keys())

        df = df[df['label'].isin(self.class_to_idx.keys())]

        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        for _, row in df.iterrows():
            img_path = str(row['path'])
            label_name = str(row['label'])
            filename = os.path.basename(img_path)
            full_path = os.path.join(MERGED_DATA, label_name, filename)
            if full_path and os.path.exists(full_path):
                self.data.append((full_path, self.class_to_idx[label_name]))
            else:
                # skip missing file 
                pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ====================model=======================
def build_clip_with_lora(device):
    clip_model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_CHECKPOINT)
    clip_model.to(device)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    clip_model = get_peft_model(clip_model, lora_cfg)
    clip_model.to(device)
    return clip_model

# ===================evaluation =================
@torch.no_grad()
def evaluate_on_test(model, test_loader, text_features, device):
    model.eval()
    text_features = text_features.to(device).detach()
    all_labels, all_preds, all_top5 = [], [], []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T

        preds = logits.argmax(dim=1)
        _, top5 = logits.topk(5, dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_top5.append(top5.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_top5 = np.concatenate(all_top5, axis=0)

    top1_acc = accuracy_score(all_labels, all_preds)
    top5_acc = ((all_top5 == all_labels.reshape(-1, 1)).any(axis=1)).mean()
    cm = confusion_matrix(all_labels, all_preds)
    balanced_acc = np.mean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1))
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Test results: Top1={top1_acc:.4f}, Top5={top5_acc:.4f}, Balanced={balanced_acc:.4f}, MacroF1={macro_f1:.4f}")
    return {'top1_acc': top1_acc, 'top5_acc': top5_acc, 'balanced_acc': balanced_acc, 'macro_f1': macro_f1}

# ========================main ===============================
def main():
    print("Device:", DEVICE)
    # dataset & loader
    test_ds = FixedMushroomDataset(TEST_CSV, ROOT_MUSHROOM, transform=test_transform(), class_mapping=None)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # model
    clip_model = build_clip_with_lora(DEVICE)

    # load checkpoint (simple)
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

    # get state dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        state = ckpt

    clip_model.load_state_dict(state, strict=False)

    # load text_features (assume saved)
    if isinstance(ckpt, dict) and 'text_features' in ckpt:
        text_features = ckpt['text_features'].to(DEVICE)
    else:
        raise RuntimeError("text_features not found in checkpoint. Use original script to rebuild or save them.")

    # evaluate
    metrics = evaluate_on_test(clip_model, test_loader, text_features, DEVICE)

    # save metrics
    out_path = os.path.join(CHECKPOINT_DIR, "final_test_metrics_minimal.json")
    save_dict = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
    with open(out_path, 'w') as f:
        json.dump(save_dict, f, indent=2)
    print("Saved metrics to", out_path)

if __name__ == "__main__":
    main()
