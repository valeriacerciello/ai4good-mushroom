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
import time
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import open_clip
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from collections import defaultdict, Counter

from torchvision import set_image_backend # conda install accimage
set_image_backend('accimage')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#reduce tokenizer parallelism noise and enable cudnn benchmark when available
##### ======================================================setting==========================================
#paths
ROOT_MUSHROOM = "/zfs/ai4good/datasets/mushroom"
TRAIN_CSV = os.path.join(ROOT_MUSHROOM, "train.csv")
VAL_CSV = os.path.join(ROOT_MUSHROOM, "val.csv")
TEST_CSV = os.path.join(ROOT_MUSHROOM, "test.csv")
PROMPT_PATH = "/zfs/ai4good/student/dkorot/ai4good-mushroom/data_prompts_label/delta_prompts.json"
CHECKPOINT_DIR = "lora_b32"
MERGED_DATA = "/zfs/ai4good/datasets/mushroom/merged_dataset"

#limits and performance
MAX_TRAIN_SAMPLES = 200000
MAX_VAL_SAMPLES = 20000
BATCH_SIZE = 520
NUM_WORKERS = 64

#hyperparameters
MAX_EPOCHS = 200
PATIENCE = 50
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.1

Lora_target_modules = [
    "visual.transformer.resblocks.10.attn.in_proj",
    "visual.transformer.resblocks.10.attn.out_proj",
    "visual.transformer.resblocks.11.attn.in_proj",
    "visual.transformer.resblocks.11.attn.out_proj",
    "visual.transformer.resblocks.10.mlp.c_fc",
    "visual.transformer.resblocks.10.mlp.c_proj",
    "visual.transformer.resblocks.11.mlp.c_fc",
    "visual.transformer.resblocks.11.mlp.c_proj",
]
#only use layer 10-11
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"TRAINING - using: {device}")

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED_CHECKPOINT = "openai"

LR = 1e-3
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.005
LABEL_SMOOTHING = 0.03
GRAD_ACCUM_STEPS = 2
# seed
SEED = 42

#==========================================================Error analysis===============================================
class ErrorAnalyzer:
    """
    Collects prediction errors for detailed analysis.
    -error records
    -top-5 error
    -confusion pairs
    """
    def __init__(self, class_names):
        self.class_names = class_names
        self.error_records = defaultdict(list)  #records
        self.top5_errors = defaultdict(Counter)  
        self.confusion_pairs = Counter() 
        
    def update(self, true_labels, predictions, top5_predictions, image_paths=None):
        true_labels = true_labels.cpu().numpy() if torch.is_tensor(true_labels) else true_labels
        predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        top5_predictions = top5_predictions.cpu().numpy() if torch.is_tensor(top5_predictions) else top5_predictions
        
        for i, (true_label, pred_label, top5_preds) in enumerate(zip(true_labels, predictions, top5_predictions)):
            true_class = self.class_names[true_label]
            pred_class = self.class_names[pred_label]
            
            # top-1 error
            if true_label != pred_label:
                self.error_records[true_class].append({
                    'predicted': pred_class,
                    'top5_preds': [self.class_names[p] for p in top5_preds],
                    'image_path': image_paths[i] if image_paths else None
                })
                
                # record confusion pair
                confusion_pair = (true_class, pred_class)
                self.confusion_pairs[confusion_pair] += 1
            #top-5 error
            if true_label not in top5_preds:
                self.top5_errors[true_class].update([self.class_names[p] for p in top5_preds])
    
    def get_top5_error_analysis(self, top_k=10):
        """
        print and return a summary of top-5 miss statistics and common confusions.
        """
        print("\n" + "="*80)
        print("TOP-5 error report")
        print("="*80)
        
        top5_missed_classes = {}
        for class_name, error_counter in self.top5_errors.items():
            if error_counter:  
                total_samples = sum(error_counter.values())
                top5_missed_classes[class_name] = total_samples
        
        if top5_missed_classes:
            print(f"\nTop-5 totally wrong class has amount（{len(top5_missed_classes)}）:")
            sorted_missed = sorted(top5_missed_classes.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for i, (class_name, count) in enumerate(sorted_missed, 1):
                print(f"   {i:2d}. {class_name:30s} - {count}tiems not in Top-5")
        
        if self.confusion_pairs:
            print(f"\n most common confusion pair:")
            sorted_pairs = self.confusion_pairs.most_common(top_k)
            for i, ((true_class, pred_class), count) in enumerate(sorted_pairs, 1):
                print(f"   {i:2d}. {true_class:25s} → {pred_class:25s} - {count}times")
        
        # Top-5 error patterns for each class
        print(f"\n Top-5 Error Patterns per Class:")
        analysis_classes = list(self.top5_errors.keys())[:top_k]
        for class_name in analysis_classes:
            if self.top5_errors[class_name]:
                print(f"\n {class_name}:")
                total_errors = sum(self.top5_errors[class_name].values())
                top_wrong_preds = self.top5_errors[class_name].most_common(5)
                for pred_class, count in top_wrong_preds:
                    percentage = (count / total_errors) * 100
                    print(f"       misclassified as: {pred_class:25s} - {count:3d}times ({percentage:5.1f}%)")
        
        # stat info
        total_errors = sum(len(records) for records in self.error_records.values())
        total_top5_errors = sum(sum(counter.values()) for counter in self.top5_errors.values())
        
        print(f"\n error stat:")
        print(f"   • total error: {total_errors}")
        print(f"   • total top5 error: {total_top5_errors}")
        print(f"   • self error records: {len(self.error_records)}")
        if self.confusion_pairs:
            most_confused = max(self.confusion_pairs, key=self.confusion_pairs.get)
            print(f"   -most common confusion pair: {most_confused[0]} → {most_confused[1]}")
        
        return {
            'top5_missed_classes': dict(sorted_missed) if top5_missed_classes else {},
            'confusion_pairs': dict(sorted_pairs) if self.confusion_pairs else {},
            'error_stats': {
                'total_errors': total_errors,
                'total_top5_errors': total_top5_errors,
                'affected_classes': len(self.error_records)
            }
        }
    
    def save_detailed_report(self, filepath="error_analysis_report.txt"):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("error report\n")
            f.write("="*50 + "\n\n")
            
            f.write("1. Top-5 totally wrong class:\n")
            top5_missed = {}
            for class_name, counter in self.top5_errors.items():
                if counter:
                    total = sum(counter.values())
                    top5_missed[class_name] = total
            
            for class_name, count in sorted(top5_missed.items(), key=lambda x: x[1], reverse=True):
                f.write(f"   {class_name}: {count}times\n")
            
            f.write("\n2. most common confusion pair:\n")
            for (true_class, pred_class), count in self.confusion_pairs.most_common(20):
                f.write(f"   {true_class} -> {pred_class}: {count}times\n")
            
            f.write("\n3. Detail\n")
            for class_name in sorted(self.top5_errors.keys()):
                if self.top5_errors[class_name]:
                    f.write(f"\n   {class_name}:\n")
                    for pred_class, count in self.top5_errors[class_name].most_common():
                        f.write(f"      misclassified as: {pred_class}: {count}times\n")

#========================================transformer======================================
def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            224, 
            scale=(0.6, 1.0),
            ratio=(0.8, 1.2),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),#Randomly crops and resizes image to 224×224 pixels；Crops 60-100% of the original image area;
        #Varies aspect ratio between 0.8 and 1.2;High-quality resampling that preserves fine mushroom details.
        transforms.RandomHorizontalFlip(p=0.5),#Horizontally flips the image with 50% probability
        transforms.RandomVerticalFlip(p=0.3),#Vertically flips the image with 30% prob.
        transforms.RandomRotation(degrees=15),#Randomly rotates the image within ±15 degrees
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1),# +- horizontal/vertical shift 
            scale=(0.9, 1.1),#90-110% scaling
            shear=5#+-5 degrees shear transformation
        ),#applies geometric transformations
        transforms.ColorJitter(
            brightness=0.2,#brightness variation
            contrast=0.2,
            saturation=0.1,
            hue=0.02
        ),# randomly adjusts color properties of the image
        transforms.ToTensor(),#converts to  pytorch tensor
        transforms.Normalize(
            mean=[0.4815, 0.4578, 0.4082], 
            std=[0.2686, 0.2613, 0.2758]
        ),
    ])

def test_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])
    ])

def val_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])
    ])

class edMushroomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_samples=None,
                class_mapping=None, balanced_sampling=True):
        df = pd.read_csv(csv_file, header=0).iloc[:, :2]
        df.columns = ["path", "label"]
        df['label'] = df['label'].astype(str)
        #create map between class name and numeric indices
        if class_mapping is None:
            classes = sorted(df['label'].unique())
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
            self.classes = classes
        else:
            self.class_to_idx = class_mapping
            self.classes = list(class_mapping.keys())

        df = df[df['label'].isin(self.class_to_idx.keys())]

        if balanced_sampling and max_samples and len(df) > max_samples:
            df = self._balanced_sampling(df, max_samples)
        elif max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=SEED)

        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        #create empty list to store tuples. Iterates through each row
        for _, row in df.iterrows():
            img_path = str(row['path'])
            label_name = str(row['label'])
            filename = os.path.basename(img_path)
            full_path = self.find_image_path(label_name, filename, img_path)
            if full_path and os.path.exists(full_path):
                self.data.append((full_path, self.class_to_idx[label_name]))
            else:
                print(f"Warning: Image not found: {img_path}")

        print(f"Loaded {len(self.data)} samples, {len(self.classes)} classes")
        #Calculate target sample per class. Then for each class, if class has more samples than target
        #randomly selects target number, if class has fewer: use all available samples.
    def _balanced_sampling(self, df, max_samples):
        samples_per_class = max(1, max_samples // len(df['label'].unique()))
        sampled_dfs = []
        for class_name in df['label'].unique():
            class_df = df[df['label'] == class_name]
            if len(class_df) > samples_per_class:
                sampled_class = class_df.sample(samples_per_class, random_state=SEED)
            else:
                sampled_class = class_df
            sampled_dfs.append(sampled_class)
        return pd.concat(sampled_dfs, ignore_index=True)
    #this just define the image path
    def find_image_path(self, label_name, filename, original_path):
        path = os.path.join(MERGED_DATA, label_name, filename)
        return path

    def __len__(self):
        return len(self.data)
    #open image file and convert to RGB format then  apply transformation pipeline.
    #error if loading fails! Then return random tensor as placeholder image, with original label
    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return torch.randn(3, 224, 224), label

#This adaptive learning rate is scheduling with warmup and cosine decay,and loss-based adjustments.
#first we linearly increases learning rate from 0 to 5 epochs.
#then gradually decreases following cosine curve.

#if current loss is greater than 1, lr=lr*0.3, if less than 0.3 then lr=lr*1.2
def adaptive_lr(epoch, batch_idx, total_batches, current_loss):
    warmup_epochs = 5
    warmup_batches = warmup_epochs * total_batches
    total_step = (epoch - 1) * total_batches + batch_idx
    
    base_lr = LR
    
    if total_step < warmup_batches:
        lr = base_lr * (total_step / warmup_batches)
    else:
        progress = (total_step - warmup_batches) / (MAX_EPOCHS * total_batches - warmup_batches)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    if current_loss > 1.0:
        lr = lr * 0.3
    elif current_loss < 0.3:
        lr = lr * 1.2
    
    return max(lr, base_lr * 0.0001)

#Dynamically adjusts the temperature.
#when epoch<5: higher temperature(1.5)
#when loss>1: higher temperature(1.2)=>smoother gradients=>stabilizes training.31
def adaptive_temperature(current_epoch, current_loss):
    base_temp = TEMPERATURE
    
    if current_epoch < 5:
        return base_temp * 1.5
    elif current_loss > 1.0:
        return base_temp * 1.2
    else:
        return base_temp
#stable cross entropy loss
def stable_cross_entropy(logits, labels, smoothing=0.03):
    logits = torch.clamp(logits, min=-50, max=50)#logits range
    
    confidence = 1.0 - smoothing# weight for the true class vs smoothing distribution, here smoothing =0.03 so confidence=0.97
    log_probs = F.log_softmax(logits, dim=-1)# log of softmax probabilities
    
    nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)#remove extra dimension
    
    smooth_loss = -log_probs.mean(dim=-1)
    
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()

@torch.no_grad()
def evaluate_with_error_analysis(model, loader, text_features, class_names, error_analyzer=None):
    """
    run forward pass on loader and compute metrics:
    -top1,top5,balanced acc, macro F1
    """
    model.eval()
    all_preds, all_labels = [], []
    all_logits = []
    all_top5_preds = []
    total_val_loss = 0
    #forward pass
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast('cuda'):
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            #loss calculation
            val_loss = stable_cross_entropy(logits / TEMPERATURE, labels, LABEL_SMOOTHING)

        total_val_loss += val_loss.item()
        _, top5_preds = logits.topk(5, dim=1)
        
        all_logits.append(logits.cpu())
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_top5_preds.extend(top5_preds.cpu().numpy())
    
    avg_val_loss = total_val_loss / len(loader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_top5_preds = np.array(all_top5_preds)
    
    if error_analyzer:
        error_analyzer.update(all_labels, all_preds, all_top5_preds)
    
    #accuracy metrics
    top1_acc = accuracy_score(all_labels, all_preds)
    
    top5_correct = all_top5_preds == all_labels.reshape(-1, 1)
    top5_acc = top5_correct.any(axis=1).sum() / len(all_labels)
    
    cm = confusion_matrix(all_labels, all_preds)
    recall_per_class = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    balanced_acc = np.mean(recall_per_class)
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"Val - Top-1: {top1_acc:.4f} | Top-5: {top5_acc:.4f} | "
        f"Balanced: {balanced_acc:.4f} | Macro F1: {macro_f1:.4f}")
    
    return {
        'top1_acc': top1_acc, 'top5_acc': top5_acc,
        'balanced_acc': balanced_acc, 'macro_f1': macro_f1,
        'val_loss': avg_val_loss
    }

def build_text_features(model, processor, class_names, class_prompts, device):
    """
    Build a normalized mean text embedding per class using prompts
    - processor: tokenizer from open clip
    """
    text_features_list = []
    
    model.eval() 
    with torch.no_grad():
        for class_name in class_names:
            #if not in our pormpt list then we just use this general prompt
            if class_name not in class_prompts:
                prompts = [f"a photo of {class_name} mushroom", 
                        f"{class_name} mushroom"]
            else:
                prompts = class_prompts[class_name]
            #only use 3 prompts
            prompts = prompts[:3]
            #text encoding
            text_inputs = processor(prompts).to(device)
            
            class_text_features = model.encode_text(text_inputs)
            #feature aggregation
            class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
            class_mean_feature = class_text_features.mean(dim=0)
            class_mean_feature = class_mean_feature / class_mean_feature.norm()
            
            text_features_list.append(class_mean_feature)
    
    text_features = torch.stack(text_features_list)
    text_features = text_features.to(device).requires_grad_(True)
    
    return text_features

def validate_initial_forward_pass(model, loader, text_features, device):
    model.eval()
    
    with torch.no_grad():
        test_batch = next(iter(loader))
        images, test_labels = test_batch
        images = images[:4].to(device)
        test_labels = test_labels[:4].to(device)
        
        with torch.amp.autocast('cuda'):
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T

            initial_loss = stable_cross_entropy(logits / TEMPERATURE, test_labels, LABEL_SMOOTHING)
        
        print(f"Image features shape: {image_features.shape}")
        print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                
        if initial_loss.item() < 10.0:
            print("Initial forward pass is ok")
            return True
        else:
            print("not ok")
            return False


#===================================================main====================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed(SEED)

    print("Loading datasets...")
    train_dataset = edMushroomDataset(
        TRAIN_CSV, ROOT_MUSHROOM, train_transform(), 
        max_samples=MAX_TRAIN_SAMPLES, balanced_sampling=True
    )
    
    val_dataset = edMushroomDataset(
        VAL_CSV, ROOT_MUSHROOM, val_transform(), 
        max_samples=None, balanced_sampling=True
    )

    test_dataset = edMushroomDataset(
        TEST_CSV,
        ROOT_MUSHROOM,
        transform=test_transform(),
        class_mapping=train_dataset.class_to_idx,
        max_samples=None 
    )
#data loading pipelines
    test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=5, 
    persistent_workers=True
    )
    
    val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=5, 
    persistent_workers=True
    )
    

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        drop_last=True, prefetch_factor=5, 
    )


    print(f"\nDataset Info:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_dataset.classes)} classes")
    print(f"   Val: {len(test_dataset)} samples")
#for track errors during evaluation, using erroranalyzer we defined early
    error_analyzer = ErrorAnalyzer(train_dataset.classes)
#load pre trained clip model and applies lora for fine tuning
    print("\nLoading CLIP model...")
    clip_model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_CHECKPOINT)
    clip_model.to(device)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=Lora_target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    clip_model = get_peft_model(clip_model, lora_config)
    clip_model.to(device)
#create text embeddings for each mushroom class using prompts
    processor = open_clip.get_tokenizer(MODEL_NAME)
    with open(PROMPT_PATH, "r") as f:
        class_prompts = json.load(f)
    print(f"Loaded prompts for {len(class_prompts)} classes")

    text_features = build_text_features(clip_model, processor, train_dataset.classes, class_prompts, device)

#optimizer configuration
    trainable_params = [
        {'params': clip_model.parameters()},
        {'params': [text_features], 'lr': LR * 0.1}  #give a small lr
    ]
    
    optimizer = torch.optim.AdamW(
        trainable_params,  # use adamw
        lr=LR, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-6
    )
#performs check before training begins, if fails with temp=0.005, try temp=1
    if not validate_initial_forward_pass(clip_model, train_loader, text_features, device):
        print("\nWarning: Initial forward pass failed!")
        global TEMPERATURE
        TEMPERATURE = 1.0
        if not validate_initial_forward_pass(clip_model, train_loader, text_features, device):
            print("Critical: Cannot proceed")
            return
#initializes training tracking variables and creates logging infrastructure
    print(f"\nTraining...")
    best_acc = 0
    best_metrics = {}
    patience_counter = 0
    accumulation_steps = GRAD_ACCUM_STEPS

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    acc_log_path = os.path.join(CHECKPOINT_DIR, "training_log.csv")

    with open(acc_log_path, "w") as f:
        f.write("epoch,batch,train_loss,val_top1,val_top5,val_balanced,val_f1,val_loss,learning_rate,temperature,grad_norm\n")
#main loop
    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{MAX_EPOCHS} ===")
        clip_model.train()
        
        if not text_features.requires_grad:#ensure text features are trainable
            text_features.requires_grad_(True)
        
        total_loss = 0
        start_time = time.time()
        grad_norms = []
        processed_batches = 0
        
        optimizer.zero_grad()
        #batch loop
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # forward pass with AMP
            with torch.amp.autocast('cuda'):
                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T

                if batch_idx == 1:
                    print(f"First batch - Logits requires_grad: {logits.requires_grad}")
                    print(f"First batch - Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

                current_temperature = adaptive_temperature(
                    epoch, total_loss / max(1, processed_batches)
                )

                loss = stable_cross_entropy(
                    logits / current_temperature, labels, LABEL_SMOOTHING
                )

            # skip problematic batches
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 50.0:
                print(f"Skipping batch {batch_idx}, loss: {loss.item():.4f}")
                optimizer.zero_grad()
                continue

            # gradient accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            total_loss += loss.item() * accumulation_steps
            processed_batches += 1

            # perform optimizer step when needed
            if (batch_idx % accumulation_steps == 0) or (batch_idx == len(train_loader)):
                current_lr = adaptive_lr(epoch, batch_idx, len(train_loader),
                                        loss.item() * accumulation_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                # Gradient clipping (must unscale first)
                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)

                if text_features.grad is not None:
                    text_grad_norm = torch.nn.utils.clip_grad_norm_(
                        [text_features], max_norm=0.1
                    )

                # AMP-safe optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # logging every 20 batches
            if batch_idx % 20 == 0:
                avg_loss = total_loss / processed_batches if processed_batches > 0 else 0

                print(
                    f"Batch {batch_idx:3d}/{len(train_loader)} | "
                    f"Loss: {loss.item() * accumulation_steps:.4f} | Avg: {avg_loss:.4f}"
                )

                with open(acc_log_path, "a") as f:
                    f.write(
                        f"{epoch},{batch_idx},{loss.item()*accumulation_steps:.6f},,,,,,"
                        f"{current_lr:.2e},{current_temperature:.4f},0.0\n"
                    )

        #summary of epoch
        epoch_time = time.time() - start_time
        if processed_batches > 0:
            avg_epoch_loss = total_loss / processed_batches
        else:
            avg_epoch_loss = 0
            print("Warning: No batches processed this epoch")
        
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
        print(f"Train Loss: {avg_epoch_loss:.4f}")
        
        # evaluation
        metrics = evaluate_with_error_analysis(
        clip_model, val_loader, text_features, 
        train_dataset.classes, error_analyzer)
        
        with open(acc_log_path, "a") as f:
            f.write(f"{epoch},-1,{avg_epoch_loss:.6f},{metrics['top1_acc']:.6f},{metrics['top5_acc']:.6f},"
                    f"{metrics['balanced_acc']:.6f},{metrics['macro_f1']:.6f},{metrics['val_loss']:.6f},,,,\n")
        
        if epoch % 5 == 0 or epoch == 1:
            error_analyzer.get_top5_error_analysis(top_k=10)
        
        current_acc = metrics['top1_acc']
        if current_acc > best_acc:
            best_acc = current_acc
            patience_counter = 0
            best_metrics = metrics.copy()
            
            torch.save({
                'model_state_dict': clip_model.state_dict(),
                'text_features': text_features.detach().cpu(),
                'class_to_idx': train_dataset.class_to_idx,
                'classes': train_dataset.classes,
                'metrics': metrics
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            
            print(f"Best model saved(Acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{PATIENCE} epochs")
            
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

#======================================summary& save results===================================
    print(f"Best Model Performance:")
    print(f"  Top-1 Accuracy: {best_metrics['top1_acc']:.4f}")
    print(f"  Top-5 Accuracy: {best_metrics['top5_acc']:.4f}")
    print(f"  Balanced Accuracy: {best_metrics['balanced_acc']:.4f}")
    print(f"  Macro F1 Score: {best_metrics['macro_f1']:.4f}")
    
    print(f"\nFinal Error Analysis")
    final_analysis = error_analyzer.get_top5_error_analysis(top_k=15)
    
    error_analyzer.save_detailed_report(os.path.join(CHECKPOINT_DIR, "error_analysis.txt"))
    
    summary_path = os.path.join(CHECKPOINT_DIR, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Training Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Best Top-1 Accuracy: {best_metrics['top1_acc']:.4f}\n")
        f.write(f"Best Top-5 Accuracy: {best_metrics['top5_acc']:.4f}\n")
        f.write(f"Best Balanced Accuracy: {best_metrics['balanced_acc']:.4f}\n")
        f.write(f"Best Macro F1: {best_metrics['macro_f1']:.4f}\n")
        f.write(f"Final Epoch: {min(epoch, MAX_EPOCHS)}\n")
    
    print(f"\nResults saved to {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()
