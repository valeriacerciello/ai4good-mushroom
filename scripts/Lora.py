import os
import time
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from collections import defaultdict, Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = True

#####setting
ROOT_MUSHROOM = "/zfs/ai4good/datasets/mushroom"
TRAIN_CSV = os.path.join(ROOT_MUSHROOM, "train.csv")
VAL_CSV = os.path.join(ROOT_MUSHROOM, "val.csv")
MAX_TRAIN_SAMPLES = 200000
MAX_VAL_SAMPLES = 2000
BATCH_SIZE = 384
NUM_WORKERS = 32
MAX_EPOCHS = 30
PATIENCE = 10
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.1

Lora_target_modules = [
    "encoder.layers.10.self_attn.q_proj",
    "encoder.layers.10.self_attn.k_proj",
    "encoder.layers.10.self_attn.v_proj",
    "encoder.layers.10.self_attn.out_proj",
    "encoder.layers.11.self_attn.q_proj",
    "encoder.layers.11.self_attn.k_proj",
    "encoder.layers.11.self_attn.v_proj",
    "encoder.layers.11.self_attn.out_proj",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" FIXED VERSION - using: {device}")

LR = 1e-3
WEIGHT_DECAY = 0.02
TEMPERATURE = 0.005
LABEL_SMOOTHING = 0.03
GRAD_ACCUM_STEPS = 2

CHECKPOINT_DIR = "lora_stable_enhanced"
SEED = 42

# Error analysis
class ErrorAnalyzer:
    def __init__(self, class_names):
        self.class_names = class_names
        self.error_records = defaultdict(list)  #records
        self.top5_errors = defaultdict(Counter)  
        self.confusion_pairs = Counter() 
        
    def update(self, true_labels, predictions, top5_predictions, image_paths=None):
        """update"""
        true_labels = true_labels.cpu().numpy() if torch.is_tensor(true_labels) else true_labels
        predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        top5_predictions = top5_predictions.cpu().numpy() if torch.is_tensor(top5_predictions) else top5_predictions
        
        for i, (true_label, pred_label, top5_preds) in enumerate(zip(true_labels, predictions, top5_predictions)):
            true_class = self.class_names[true_label]
            pred_class = self.class_names[pred_label]
            
            # if pred wrong
            if true_label != pred_label:
                self.error_records[true_class].append({
                    'predicted': pred_class,
                    'top5_preds': [self.class_names[p] for p in top5_preds],
                    'image_path': image_paths[i] if image_paths else None
                })
                
                # record confusion pair
                confusion_pair = (true_class, pred_class)
                self.confusion_pairs[confusion_pair] += 1
            
            if true_label not in top5_preds:
                self.top5_errors[true_class].update([self.class_names[p] for p in top5_preds])
    
    def get_top5_error_analysis(self, top_k=10):
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
        
        # 3. Top-5 error patterns for each class
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
        
        # 4. stat info
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

def get_enhanced_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            224, 
            scale=(0.6, 1.0),
            ratio=(0.8, 1.2),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
        
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4, 
            saturation=0.3,
            hue=0.05
        ),
        
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),
        
        transforms.RandomApply([
            transforms.Grayscale(num_output_channels=3)
        ], p=0.1),
        
        transforms.ToTensor(),
        
        transforms.RandomErasing(
            p=0.2, 
            scale=(0.02, 0.15), 
            ratio=(0.3, 3.3),
            value='random'
        ),
        
        transforms.Normalize(
            mean=[0.4815, 0.4578, 0.4082], 
            std=[0.2686, 0.2613, 0.2758]
        ),
    ])

def get_enhanced_val_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])
    ])

class FixedMushroomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_samples=None,
                 class_mapping=None, balanced_sampling=True):
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

        if balanced_sampling and max_samples and len(df) > max_samples:
            df = self._balanced_sampling(df, max_samples)
        elif max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=SEED)

        self.data = []
        self.root_dir = root_dir
        self.transform = transform

        for _, row in df.iterrows():
            img_path = str(row['path'])
            label_name = str(row['label'])
            filename = os.path.basename(img_path)
            full_path = self._find_image_path(label_name, filename, img_path)
            if full_path and os.path.exists(full_path):
                self.data.append((full_path, self.class_to_idx[label_name]))

        print(f"Loaded {len(self.data)} samples, {len(self.classes)} classes")
        
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

    def _find_image_path(self, label_name, filename, original_path):
        possible_paths = [
            os.path.join(self.root_dir, "merged_dataset", label_name, filename),
            os.path.join(self.root_dir, label_name, filename),
            original_path,
        ]
        if "merged_dataset/" in original_path:
            alt_rel = original_path.split("merged_dataset/")[-1]
            possible_paths.append(os.path.join(self.root_dir, "merged_dataset", alt_rel))

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            return torch.randn(3, 224, 224), label

def get_adaptive_lr(epoch, batch_idx, total_batches, current_loss):
    warmup_epochs = 3
    warmup_batches = warmup_epochs * total_batches
    total_step = (epoch - 1) * total_batches + batch_idx
    
    base_lr = LR
    
    # Warmup
    if total_step < warmup_batches:
        lr = base_lr * (total_step / warmup_batches)
    else:
        progress = (total_step - warmup_batches) / (MAX_EPOCHS * total_batches - warmup_batches)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    # Dynamically adjust based on loss
    if current_loss > 0.8:
        lr = lr * 0.5  # low lr when high loss 
    elif current_loss < 0.4:
        lr = lr * 1.2  # high lr when low loss
    
    return max(lr, base_lr * 0.001)

def get_adaptive_temperature(current_epoch, current_loss):
    base_temp = 0.005
    
    if current_epoch < 3:
        return base_temp * 2.0  # more temp in the begining
    elif current_loss > 0.8:
        return base_temp * 1.5  
    else:
        return base_temp

def stable_cross_entropy(logits, labels, smoothing=0.03):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    
    nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    
    smooth_loss = -log_probs.mean(dim=-1)
    
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()

def difficulty_aware_loss(logits, labels, temperature=0.005, smoothing=0.03):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    
    with torch.no_grad():
        probs = F.softmax(logits / temperature, dim=-1)
        correct_probs = probs[torch.arange(probs.size(0)), labels]
        difficulty_weights = 1.0 - correct_probs 
    
    nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    
    smooth_loss = -log_probs.mean(dim=-1)
    
    loss = confidence * nll_loss + smoothing * smooth_loss
    weighted_loss = (loss * difficulty_weights).mean()
    
    return weighted_loss

def regularized_loss(model, logits, labels, alpha=0.01):
    ce_loss = stable_cross_entropy(logits / TEMPERATURE, labels, LABEL_SMOOTHING)
    
    l2_reg = torch.tensor(0.0).to(device)
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            l2_reg += torch.norm(param)
    
    return ce_loss + alpha * l2_reg

@torch.no_grad()
def evaluate_with_error_analysis(model, loader, text_features, class_names, error_analyzer=None):
    model.eval()
    all_preds, all_labels = [], []
    all_logits = []
    all_top5_preds = []
    total_val_loss = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        image_features = model.get_image_features(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T
        
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

def build_stable_text_features(model, processor, class_names, class_prompts, device):
    text_features_list = []
    
    with torch.no_grad():
        for class_name in class_names:
            if class_name not in class_prompts:
                prompts = [f"a photo of {class_name} mushroom"]
            else:
                prompts = class_prompts[class_name]
            
            prompts = prompts[:6] if len(prompts) > 6 else prompts
            
            text_inputs = processor(
                text=prompts,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            class_text_features = model.get_text_features(**text_inputs)
            class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
            class_mean_feature = class_text_features.mean(dim=0)
            class_mean_feature = class_mean_feature / class_mean_feature.norm()
            
            text_features_list.append(class_mean_feature)
    
    text_features = torch.stack(text_features_list)
    print(f"Stable text features built: {text_features.shape}")
    return text_features

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"STABLE TRAINING & ERROR ANALYSIS - using: {device}")

    print("Loading datasets ...")
    train_dataset = FixedMushroomDataset(
        TRAIN_CSV, ROOT_MUSHROOM, get_enhanced_train_transform(), MAX_TRAIN_SAMPLES, balanced_sampling=True
    )

    val_dataset = FixedMushroomDataset(
        VAL_CSV, ROOT_MUSHROOM, get_enhanced_val_transform(), MAX_VAL_SAMPLES,
        class_mapping=train_dataset.class_to_idx, balanced_sampling=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=8, pin_memory=True
    )

    print(f"Dataset Info:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_dataset.classes)} classes")
    print(f"   Val: {len(val_dataset)} samples")

    error_analyzer = ErrorAnalyzer(train_dataset.classes)

    print(" Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.requires_grad_(False)

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

    trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    PROMPT_PATH = "/zfs/ai4good/student/hgupta/enhanced_with_common_names_prompts_clean.json"
    with open(PROMPT_PATH, "r") as f:
        class_prompts = json.load(f)

    print(f"Loaded prompts for {len(class_prompts)} classes")

    text_features = build_stable_text_features(clip_model, processor, train_dataset.classes, class_prompts, device)

    optimizer = torch.optim.AdamW(
        clip_model.parameters(), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    print(f"\n Starting STABLE training with enhanced augmentation & error analysis...")
    best_acc = 0
    best_metrics = {}
    patience_counter = 0
    accumulation_steps = GRAD_ACCUM_STEPS

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    acc_log_path = os.path.join(CHECKPOINT_DIR, "stable_enhanced_log.csv")

    with open(acc_log_path, "w") as f:
        f.write("epoch,batch,train_loss,val_top1,val_top5,val_balanced,val_f1,val_loss,learning_rate,temperature,grad_norm\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{MAX_EPOCHS} ===")
        clip_model.train()
        total_loss = 0
        start_time = time.time()
        grad_norms = []
        
        optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            image_features = clip_model.get_image_features(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = image_features @ text_features.T
            
            current_temperature = get_adaptive_temperature(epoch, total_loss / max(1, batch_idx))
            
            loss = difficulty_aware_loss(logits, labels, current_temperature, LABEL_SMOOTHING)
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 5.0:
                print(f" jump batch {batch_idx}, loss: {loss.item():.4f}")
                optimizer.zero_grad()
                continue
            
            loss = loss / accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            
            if (batch_idx % accumulation_steps == 0) or (batch_idx == len(train_loader)):
                current_lr = get_adaptive_lr(epoch, batch_idx, len(train_loader), loss.item() * accumulation_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                current_loss = loss.item() * accumulation_steps
                max_norm = 0.3 if current_loss > 0.8 else 1.0
                
                total_norm = torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=max_norm)
                grad_norms.append(total_norm.item())
                
                optimizer.step()
                optimizer.zero_grad()
            
            if batch_idx % 5 == 0 and (batch_idx % accumulation_steps == 0):
                avg_loss = total_loss / batch_idx
                avg_grad_norm = np.mean(grad_norms[-5:]) if grad_norms else 0
                
                print(f" Batch {batch_idx:3d}/{len(train_loader)} | Loss: {current_loss:.4f} | Avg: {avg_loss:.4f} | "
                      f"Grad: {avg_grad_norm:.2f} | LR: {current_lr:.2e} | Temp: {current_temperature:.4f}")
                
                with open(acc_log_path, "a") as f:
                    f.write(f"{epoch},{batch_idx},{current_loss:.6f},,,,,,{current_lr:.2e},{current_temperature:.4f},{avg_grad_norm:.4f}\n")
        
        epoch_time = time.time() - start_time
        metrics = evaluate_with_error_analysis(clip_model, val_loader, text_features, 
                                             train_dataset.classes, error_analyzer)
        avg_epoch_loss = total_loss / len(train_loader)
        
        print(f"  Epoch {epoch} completed in {epoch_time:.1f}s")
        print(f" Train Loss: {avg_epoch_loss:.4f}")
        
        with open(acc_log_path, "a") as f:
            f.write(f"{epoch},-1,{avg_epoch_loss:.6f},{metrics['top1_acc']:.6f},{metrics['top5_acc']:.6f},"
                    f"{metrics['balanced_acc']:.6f},{metrics['macro_f1']:.6f},{metrics['val_loss']:.6f},,,,\n")
        
        if epoch % 3 == 0 or epoch >= MAX_EPOCHS - 2 or epoch == 1:
            error_analyzer.get_top5_error_analysis(top_k=15)
        
        current_acc = metrics['top1_acc']
        if current_acc > best_acc:
            best_acc = current_acc
            patience_counter = 0
            best_metrics = metrics.copy()
            
            clip_model.save_pretrained(CHECKPOINT_DIR)
            print(f"best model (Acc: {best_acc:.4f}) saved!")
            
            if current_acc >= 0.85:
                target_dir = f"{CHECKPOINT_DIR}_85_achieved"
                clip_model.save_pretrained(target_dir)
                print(f" Model saved to {target_dir}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{PATIENCE} epochs")
            
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining completed!")
    print(f"Best Model - Top-1: {best_metrics['top1_acc']:.4f} | Top-5: {best_metrics['top5_acc']:.4f} | "
          f"Balanced: {best_metrics['balanced_acc']:.4f} | Macro F1: {best_metrics['macro_f1']:.4f}")

    print(f"\n{'='*80}")
    print("error analysis")
    print(f"{'='*80}")
    final_analysis = error_analyzer.get_top5_error_analysis(top_k=20)
    
    error_analyzer.save_detailed_report(os.path.join(CHECKPOINT_DIR, "error_analysis.txt"))
    
    summary_path = os.path.join(CHECKPOINT_DIR, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Training Summary - Enhanced Version with Error Analysis\n")
        f.write("======================================================\n")
        f.write(f"Best Top-1 Accuracy: {best_metrics['top1_acc']:.4f}\n")
        f.write(f"Best Top-5 Accuracy: {best_metrics['top5_acc']:.4f}\n")
        f.write(f"Best Balanced Accuracy: {best_metrics['balanced_acc']:.4f}\n")
        f.write(f"Best Macro F1: {best_metrics['macro_f1']:.4f}\n")
        f.write(f"Final Val Loss: {best_metrics['val_loss']:.4f}\n")
        f.write(f"Training completed at epoch: {min(epoch, MAX_EPOCHS)}\n")
        f.write(f"Enhanced Augmentation: Enabled\n")
        f.write(f"Stable Training: Enabled\n")
        f.write(f"Error Analysis: Enabled\n\n")
        
        f.write("Top-5 keywords:\n")
        f.write(f"- Top-5 missed classed: {len(final_analysis['top5_missed_classes'])}\n")
        f.write(f"- total error: {final_analysis['error_stats']['total_errors']}\n")
        f.write(f"- affected classes: {final_analysis['error_stats']['affected_classes']}\n")

    print(f"Detailed logs saved to: {acc_log_path}")
    print(f"Training summary saved to: {summary_path}")
    print(f" Error analysis saved to: {os.path.join(CHECKPOINT_DIR, 'error_analysis.txt')}")

if __name__ == "__main__":
    main()
