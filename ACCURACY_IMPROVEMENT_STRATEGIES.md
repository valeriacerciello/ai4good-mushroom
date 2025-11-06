# STRATEGIES TO INCREASE MUSHROOM CLASSIFICATION ACCURACY

## Current Performance Baseline
- **CLIP (ViT-B/32)**: 25.0% top-1, 48.0% top-5 (zero-shot)
- **ResNet18 (1 epoch)**: 1.3% top-1 (severely undertrained)
- **Best Fusion**: 25.0% top-1 (defaults to CLIP-only)

## Target Goals
- **Short-term**: 35-40% top-1 accuracy
- **Medium-term**: 50-60% top-1 accuracy  
- **Long-term**: 70-80% top-1 accuracy

---

## IMMEDIATE ACTIONS (Quick Wins - Hours to Days)

### 1. Train Proper ResNet Baseline ⭐⭐⭐ (HIGHEST PRIORITY)
**Current Problem**: ResNet18 with only 1 epoch is useless (~1% accuracy)

**Actions**:
```bash
# Option A: Train ResNet18 for 20 epochs on full dataset
cd ai4good-mushroom/external/ecovision_mushroom
mushroom_env/bin/python3 src/train.py \
  experiment=base model=resnet18 logger=csv \
  trainer.max_epochs=20 \
  trainer.accelerator=cpu trainer.precision=32

# Option B: Train ResNet50 for 10-15 epochs (better but slower)
mushroom_env/bin/python3 src/train.py \
  experiment=base model=resnet50 logger=csv \
  trainer.max_epochs=15 \
  trainer.accelerator=cpu trainer.precision=32
```

**Expected Improvement**:
- ResNet18 (20 epochs): 45-55% top-1 accuracy
- ResNet50 (15 epochs): 60-70% top-1 accuracy
- Fusion with strong ResNet: +5-10% over CLIP alone = **30-35% top-1**

**Time Investment**: 8-24 hours training time
**Impact**: HIGH - Essential for meaningful fusion

---

### 2. Use ALL Available Prompts ⭐⭐
**Current**: Using only delta prompts (10,389 prompts)
**Available**: enhanced_with_common_names_prompts_clean.json (19,141 prompts)

**Action**:
```bash
# Re-run CLIP evaluation with full prompt set
mushroom_env/bin/python3 eval_delta_only.py \
  --prompts enhanced_with_common_names_prompts_clean.json \
  --temperature 0.05 \
  --n_samples 200

# Update fusion script to use full prompts
mushroom_env/bin/python3 fusion_clip_resnet.py \
  --prompts enhanced_with_common_names_prompts_clean.json \
  --clip_temp 0.05 \
  ...
```

**Expected Improvement**: +2-5% top-1 accuracy
**Time Investment**: 1-2 hours
**Impact**: MEDIUM - More prompts = better coverage

---

### 3. Use Larger CLIP Model ⭐⭐⭐
**Current**: ViT-B/32 (smallest CLIP variant)
**Available**: ViT-B/16, ViT-L/14, ViT-L/14@336px

**Action**:
```python
# In clip_utils.py or evaluation scripts
import clip

# Option 1: ViT-B/16 (2× more patches, better detail)
model, preprocess = clip.load("ViT-B/16", device=device)

# Option 2: ViT-L/14 (3× larger model, best performance)
model, preprocess = clip.load("ViT-L/14", device=device)

# Option 3: ViT-L/14@336px (highest resolution)
model, preprocess = clip.load("ViT-L/14@336px", device=device)
```

**Expected Improvement**:
- ViT-B/16: +3-5% over ViT-B/32 = **28-30% top-1**
- ViT-L/14: +8-12% over ViT-B/32 = **33-37% top-1**
- ViT-L/14@336px: +10-15% over ViT-B/32 = **35-40% top-1**

**Time Investment**: 2-4 hours (slower inference)
**Impact**: HIGH - Proven accuracy boost

---

### 4. Optimize Temperature Per-Class ⭐
**Current**: Fixed T=0.05 for all classes
**Better**: Learn optimal temperature per species

**Action**:
```python
# Add to temperature_sweep.py
def optimize_per_class_temperature(val_loader, clip_model, prompts, device):
    """Find best temperature for each class on validation set"""
    temps = [0.03, 0.05, 0.07, 0.10]
    best_temps = {}
    
    for class_idx, label in enumerate(labels):
        # Get samples for this class
        class_samples = [x for x in val_loader if x['label'] == class_idx]
        
        # Try each temperature
        best_acc = 0
        best_t = 0.05
        for t in temps:
            acc = evaluate_class_with_temp(class_samples, t)
            if acc > best_acc:
                best_acc = acc
                best_t = t
        
        best_temps[label] = best_t
    
    return best_temps
```

**Expected Improvement**: +1-3% top-1 accuracy
**Time Investment**: 2-3 hours
**Impact**: LOW-MEDIUM - Helps with hard classes

---

### 5. Ensemble Multiple CLIP Models ⭐⭐
**Concept**: Average predictions from different CLIP architectures

**Action**:
```python
import clip

# Load multiple CLIP variants
model_b32, preprocess_b32 = clip.load("ViT-B/32", device=device)
model_b16, preprocess_b16 = clip.load("ViT-B/16", device=device)
model_l14, preprocess_l14 = clip.load("ViT-L/14", device=device)

# For each test image
probs_b32 = compute_clip_probs(image, model_b32, preprocess_b32)
probs_b16 = compute_clip_probs(image, model_b16, preprocess_b16)
probs_l14 = compute_clip_probs(image, model_l14, preprocess_l14)

# Ensemble (equal weights or tuned on validation)
ensemble_probs = (probs_b32 + probs_b16 + probs_l14) / 3
prediction = torch.argmax(ensemble_probs)
```

**Expected Improvement**: +3-6% over best single model
**Time Investment**: 4-6 hours
**Impact**: MEDIUM-HIGH - Diverse models help

---

## SHORT-TERM ACTIONS (Days to Weeks)

### 6. Fine-tune CLIP on Mushroom Dataset ⭐⭐⭐ (VERY HIGH IMPACT)
**Current**: Zero-shot CLIP (never seen mushrooms during training)
**Better**: Fine-tune on mushroom training data

**Action**:
```python
# Create fine-tuning script
import clip
import torch
from torch import optim

# Load pretrained CLIP
model, preprocess = clip.load("ViT-B/32", device=device)

# Freeze vision encoder initially (optional)
for param in model.visual.parameters():
    param.requires_grad = False

# Only train text encoder + classifier head
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Training loop
for epoch in range(5):
    for images, labels in train_loader:
        # Get image features
        image_features = model.encode_image(images)
        
        # Get text features for all class prompts
        text_features = get_all_class_text_features(model, prompts)
        
        # Compute similarity and loss
        logits = image_features @ text_features.T
        loss = F.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Expected Improvement**: +15-25% over zero-shot = **40-50% top-1**
**Time Investment**: 1-3 days training
**Impact**: VERY HIGH - Adapts to mushroom domain

---

### 7. Add Data Augmentation ⭐⭐
**Current**: Basic transforms in ResNet training
**Better**: Aggressive augmentation for fine-grained recognition

**Action**:
```python
# In ResNet training config or dataloader
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # Mushrooms have no up/down
    transforms.RandomRotation(180),    # Rotation invariance
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Expected Improvement**: +3-7% for ResNet training
**Time Investment**: 1-2 hours to implement
**Impact**: MEDIUM - Better generalization

---

### 8. Implement Feature-Level Fusion ⭐⭐
**Current**: Score-level fusion (late fusion)
**Better**: Feature-level fusion (early fusion)

**Action**:
```python
# New fusion architecture
class CLIPResNetFusion(nn.Module):
    def __init__(self, clip_model, resnet_model, num_classes=169):
        super().__init__()
        self.clip_encoder = clip_model.visual
        self.resnet_encoder = resnet_model
        
        # Remove final classification layers
        self.resnet_encoder.fc = nn.Identity()
        
        # Fusion layer
        clip_dim = 512  # ViT-B/32 output dim
        resnet_dim = 512  # ResNet18 output dim
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim + resnet_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        clip_features = self.clip_encoder(x)
        resnet_features = self.resnet_encoder(x)
        
        # Concatenate features
        combined = torch.cat([clip_features, resnet_features], dim=1)
        
        # Final classification
        logits = self.fusion(combined)
        return logits
```

**Expected Improvement**: +3-8% over score-level fusion
**Time Investment**: 1-2 days (implementation + training)
**Impact**: MEDIUM-HIGH - Better feature interaction

---

### 9. Use Class Weights / Focal Loss ⭐
**Current**: Standard cross-entropy (ignores class imbalance)
**Problem**: Balanced accuracy (15.2%) << Top-1 (25.0%)

**Action**:
```python
# Compute class weights from training set
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.FloatTensor(class_weights).to(device)

# Use weighted loss
criterion = nn.CrossEntropyLoss(weight=class_weights)

# OR use Focal Loss (better for hard examples)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=1, gamma=2)
```

**Expected Improvement**: +2-5% balanced accuracy and macro-F1
**Time Investment**: 2-3 hours
**Impact**: MEDIUM - Helps rare classes

---

### 10. Add Self-Training / Pseudo-Labeling ⭐⭐
**Concept**: Use model predictions on unlabeled data to expand training set

**Action**:
```python
# Step 1: Train initial model on labeled data
model = train_model(labeled_train_data)

# Step 2: Predict on unlabeled data
unlabeled_loader = get_unlabeled_data()
pseudo_labels = []
high_confidence_samples = []

for images in unlabeled_loader:
    with torch.no_grad():
        probs = F.softmax(model(images), dim=1)
        max_probs, predictions = probs.max(dim=1)
        
        # Keep only high-confidence predictions (e.g., > 0.9)
        confident = max_probs > 0.9
        high_confidence_samples.extend(images[confident])
        pseudo_labels.extend(predictions[confident])

# Step 3: Retrain on labeled + pseudo-labeled data
combined_train_data = labeled_train_data + high_confidence_samples
retrained_model = train_model(combined_train_data)
```

**Expected Improvement**: +3-7% if unlabeled data available
**Time Investment**: 2-3 days
**Impact**: MEDIUM - Depends on data availability

---

## MEDIUM-TERM ACTIONS (Weeks to Months)

### 11. Use Vision Transformer (ViT) Instead of ResNet ⭐⭐⭐
**Concept**: ViT architectures often outperform CNNs on fine-grained tasks

**Action**:
```python
# Use timm library for pretrained ViT models
import timm

# Load pretrained ViT (ImageNet-21k → ImageNet-1k)
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=169)

# OR use larger model
model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=169)

# Train as usual
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
```

**Expected Improvement**: +5-15% over ResNet = **50-70% top-1**
**Time Investment**: 1-2 weeks training
**Impact**: HIGH - ViT excels at fine-grained details

---

### 12. Implement Attention-Based Fusion ⭐⭐
**Concept**: Learn to attend to CLIP vs ResNet dynamically per sample

**Action**:
```python
class AttentionFusion(nn.Module):
    def __init__(self, clip_model, resnet_model, num_classes=169):
        super().__init__()
        self.clip_model = clip_model
        self.resnet_model = resnet_model
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(2 * num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Get logits from both models
        clip_logits = self.clip_model(x)
        resnet_logits = self.resnet_model(x)
        
        # Compute attention weights based on both predictions
        combined_logits = torch.cat([clip_logits, resnet_logits], dim=1)
        attention_weights = self.attention(combined_logits)  # [batch, 2]
        
        # Weighted combination
        fused_logits = (attention_weights[:, 0:1] * clip_logits + 
                       attention_weights[:, 1:2] * resnet_logits)
        return fused_logits
```

**Expected Improvement**: +2-5% over fixed-weight fusion
**Time Investment**: 1 week
**Impact**: MEDIUM - Adaptive fusion per sample

---

### 13. Add Hierarchical Classification ⭐⭐
**Concept**: Use taxonomic hierarchy (genus → species)

**Action**:
```python
# Define taxonomy
taxonomy = {
    'Amanita': ['Amanita muscaria', 'Amanita phalloides', ...],
    'Boletus': ['Boletus edulis', 'Boletus rex-veris', ...],
    # ... for all 169 species
}

# Two-stage classifier
class HierarchicalClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.genus_classifier = nn.Linear(512, num_genera)
        self.species_classifiers = nn.ModuleDict({
            genus: nn.Linear(512, len(species))
            for genus, species in taxonomy.items()
        })
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Stage 1: Predict genus
        genus_logits = self.genus_classifier(features)
        predicted_genus = torch.argmax(genus_logits, dim=1)
        
        # Stage 2: Predict species within genus
        species_logits = self.species_classifiers[predicted_genus](features)
        return species_logits

# Loss: Combine genus loss + species loss
loss = genus_loss + species_loss
```

**Expected Improvement**: +5-10% (especially for similar species)
**Time Investment**: 2-3 weeks (requires taxonomy data)
**Impact**: MEDIUM-HIGH - Leverages domain knowledge

---

### 14. Use Contrastive Learning ⭐⭐⭐
**Concept**: Learn embeddings that group similar mushrooms together

**Action**:
```python
# SimCLR-style contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity = features @ features.T / self.temperature
        
        # Create positive pairs mask (same class)
        labels = labels.unsqueeze(0)
        mask = (labels == labels.T).float()
        
        # Contrastive loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -(log_prob * mask).sum(dim=1) / mask.sum(dim=1)
        return loss.mean()

# Training
for images, labels in train_loader:
    features = backbone(images)
    loss = contrastive_loss(features, labels)
    ...
```

**Expected Improvement**: +8-15% (better embeddings)
**Time Investment**: 2-3 weeks
**Impact**: HIGH - Excellent for fine-grained tasks

---

### 15. Implement Test-Time Augmentation (TTA) ⭐
**Concept**: Average predictions over multiple augmented versions

**Action**:
```python
def test_time_augmentation(model, image, n_augments=10):
    """Predict with multiple augmented versions"""
    predictions = []
    
    # Original image
    pred = model(image)
    predictions.append(pred)
    
    # Augmented versions
    for _ in range(n_augments - 1):
        aug_image = augment(image)  # Random crop, flip, rotate, etc.
        pred = model(aug_image)
        predictions.append(pred)
    
    # Average predictions
    avg_prediction = torch.stack(predictions).mean(dim=0)
    return avg_prediction

# Use during evaluation
for images, labels in test_loader:
    predictions = test_time_augmentation(model, images, n_augments=10)
    accuracy = compute_accuracy(predictions, labels)
```

**Expected Improvement**: +2-4% (free accuracy boost)
**Time Investment**: 1-2 hours
**Impact**: LOW-MEDIUM - No retraining needed

---

## ADVANCED / LONG-TERM STRATEGIES

### 16. Use Foundation Models (DINOv2, SAM) ⭐⭐⭐
```python
# DINOv2 has excellent fine-grained recognition
import torch
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
```

### 17. Meta-Learning / Few-Shot Learning ⭐⭐
For rare species with few training examples

### 18. Multi-Task Learning ⭐⭐
Jointly predict edibility, genus, habitat, etc.

### 19. External Data Augmentation ⭐⭐⭐
Scrape iNaturalist, Mushroom Observer for more images

### 20. Ensemble of Everything ⭐⭐⭐
Combine CLIP (multiple scales) + ViT + ResNet + DINOv2

---

## RECOMMENDED PRIORITY ORDER

### Phase 1: Quick Wins (This Week)
1. ✅ Train ResNet18 for 20 epochs (8-12 hours)
2. ✅ Use larger CLIP model (ViT-L/14) (4-6 hours)
3. ✅ Use all available prompts (2 hours)
4. ✅ Re-run fusion with strong models (1 hour)

**Expected Result**: 35-45% top-1 accuracy

### Phase 2: Medium Effort (Next 2 Weeks)
5. ✅ Fine-tune CLIP on mushroom data (2-3 days)
6. ✅ Implement feature-level fusion (2-3 days)
7. ✅ Add class weights / focal loss (1 day)
8. ✅ Train ViT instead of ResNet (1 week)

**Expected Result**: 50-65% top-1 accuracy

### Phase 3: Advanced (Next Month)
9. ✅ Contrastive learning pretraining (2 weeks)
10. ✅ Hierarchical classification with taxonomy (2 weeks)
11. ✅ CLIP ensemble + ViT ensemble (1 week)

**Expected Result**: 70-80% top-1 accuracy

---

## IMMEDIATE NEXT STEPS

Run this now for quick improvement:

```bash
# 1. Start training strong ResNet18 (run overnight)
cd /zfs/ai4good/student/hgupta/ai4good-mushroom/external/ecovision_mushroom
nohup mushroom_env/bin/python3 src/train.py \
  experiment=base model=resnet18 logger=csv \
  trainer.max_epochs=20 trainer.accelerator=cpu \
  > training.log 2>&1 &

# 2. Evaluate with larger CLIP model (tomorrow)
cd /zfs/ai4good/student/hgupta
# Create eval_large_clip.py (copy eval_delta_only.py, change model to ViT-L/14)

# 3. Re-run fusion with strong models (after training completes)
mushroom_env/bin/python3 fusion_clip_resnet.py \
  --model resnet18 \
  --ckpt logs/mushroom/train/runs/LATEST/checkpoints/best.ckpt \
  --prompts enhanced_with_common_names_prompts_clean.json \
  --clip_temp 0.05 \
  --val_n -1 --test_n -1 \
  --out fusion_results_strong.json
```

---

## EXPECTED TIMELINE TO REACH GOALS

- **35-40% top-1**: 1-2 days (strong ResNet + larger CLIP)
- **50-60% top-1**: 2-3 weeks (fine-tuned CLIP + ViT)
- **70-80% top-1**: 1-2 months (ensembles + advanced techniques)

The biggest single improvement will come from **fine-tuning CLIP** on your mushroom dataset - this alone could give you +15-20% absolute improvement!
