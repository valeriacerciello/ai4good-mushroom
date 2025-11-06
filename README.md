# CLIP-based Mushroom Classification

This project implements a **zero-shot mushroom classification system** using CLIP (Contrastive Language-Image Pretraining) models. Instead of traditional supervised learning, this approach uses text descriptions to classify mushroom species, allowing for classification of new species without retraining.

## 🍄 Overview

The system works by:
1. **Generating comprehensive text prompts** for each mushroom species
2. **Encoding text prompts** using CLIP's text encoder
3. **Encoding mushroom images** using CLIP's vision encoder
4. **Comparing similarities** between image and text embeddings
5. **Classifying** based on highest similarity scores

## 📊 Your Dataset

- **169 mushroom/fungi species**
- **~720,000 total images**
- **Train set**: 689,520 images
- **Validation set**: 15,616 images  
- **Test set**: 15,614 images

Location: `/zfs/ai4good/datasets/mushroom/`

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install PyTorch (if not already installed)
pip install torch torchvision

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Install other requirements
pip install -r requirements.txt
```

### 2. Generate Text Prompts for All Species

```bash
python generate_clip_prompts.py
```

This will:
- Read your training data from `/zfs/ai4good/datasets/mushroom/train.csv`
- Generate multiple text prompts for each of the 169 species
- Save results to `mushroom_prompts.json`

### 3. Run CLIP Classification Demo

```bash
python clip_mushroom_classifier.py --demo
```

### 4. Full Evaluation (once prompts are ready)

```bash
python clip_mushroom_classifier.py \
    --prompts mushroom_prompts.json \
    --test_csv /zfs/ai4good/datasets/mushroom/test.csv \
    --output results.json
```

## 📁 Project Structure

```
~/mushroom_clip/
├── generate_clip_prompts.py      # Prompt generation system
├── clip_mushroom_classifier.py   # CLIP classification model
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── (generated files)
    ├── mushroom_prompts.json     # Text prompts for all species
    └── results.json              # Classification results
```

## 🛠 How It Works

### 1. Text Prompt Generation

For each species (e.g., "Amanita muscaria"), the system generates:

**Basic prompts:**
```
- "a photo of Amanita muscaria"
- "Amanita muscaria mushroom" 
- "the fungus Amanita muscaria"
```

**Descriptive prompts (for known species):**
```
- "a photo of fly agaric"
- "a mushroom with red cap and white spots"
- "a bright red mushroom"
- "Amanita muscaria growing in birch and pine forests"
```

**Contextual prompts:**
```
- "Amanita muscaria in natural habitat"
- "wild Amanita muscaria"
- "close-up of Amanita muscaria"
- "mature Amanita muscaria"
```

### 2. CLIP Classification Process

```python
# 1. Load CLIP model
model, preprocess = clip.load("ViT-B/32")

# 2. Encode all text prompts for each species
for species in all_species:
    text_features[species] = model.encode_text(species_prompts[species])

# 3. For each image:
image_features = model.encode_image(image)

# 4. Compare with all species
similarities = cosine_similarity(image_features, text_features)

# 5. Return most similar species
predicted_species = max(similarities)
```

## 🎯 Advantages of This Approach

1. **Zero-shot learning**: Can classify new species by just adding text descriptions
2. **No retraining needed**: Works immediately with any new species
3. **Interpretable**: Can see which text descriptions match best
4. **Flexible**: Easy to modify descriptions to improve accuracy
5. **Multilingual potential**: Can work with descriptions in different languages

## 📊 Expected Performance

Based on similar CLIP applications:
- **Top-1 accuracy**: 50-75% (varies by species distinctiveness)
- **Top-5 accuracy**: 75-90%
- **Best performance**: Distinctive species (Amanita muscaria, Boletus edulis)
- **Challenging**: Similar-looking species within same genus

## 🔧 Customization Options

### CLIP Models Available:
- `ViT-B/32` (default) - Fast, good performance
- `ViT-B/16` - Better accuracy, slower
- `ViT-L/14` - Best accuracy, requires more GPU memory

### Adding New Species Knowledge:

Edit `generate_clip_prompts.py` and add to `species_knowledge`:

```python
'Your Species Name': {
    'common_names': ['common name 1', 'common name 2'],
    'key_features': ['distinctive feature 1', 'feature 2'],
    'colors': ['color1', 'color2'],
    'habitat': ['habitat type'],
    'toxicity': 'edible/poisonous/medicinal',
    'size': 'small/medium/large'
}
```

## 🧪 Usage Examples

### Single Image Prediction
```python
from clip_mushroom_classifier import CLIPMushroomClassifier

classifier = CLIPMushroomClassifier()
classifier.load_prompts("mushroom_prompts.json")
classifier.encode_text_prompts()

# Predict
predictions = classifier.predict_single("path/to/mushroom.jpg", top_k=5)
for species, confidence in predictions:
    print(f"{species}: {confidence:.3f}")
```

### Evaluate on Test Set
```python
results = classifier.evaluate_dataset("/zfs/ai4good/datasets/mushroom/test.csv")
print(f"Accuracy: {results['accuracy']:.3f}")
```

## ⚠️ Important Notes

1. **GPU Recommended**: CLIP models work much faster on GPU
2. **Memory Usage**: Encoding 169 species prompts requires ~2GB GPU memory
3. **Image Processing**: Handles various image formats automatically
4. **Path Conversion**: Automatically converts Kaggle paths to your local paths

## 🚀 Next Steps & Improvements

1. **Optimize prompts**: Experiment with different prompt formulations
2. **Ensemble models**: Combine multiple CLIP model predictions
3. **Fine-tuning**: Fine-tune CLIP on your specific mushroom dataset
4. **Hierarchical classification**: Use taxonomic hierarchy (genus → species)
5. **Active learning**: Iteratively improve with user feedback

## 📚 Key Species in Your Dataset

Your dataset includes major groups:
- **Amanita** (20+ species including A. muscaria, A. phalloides)
- **Boletus** (edible boletes)
- **Psilocybe** (psychoactive species) 
- **Ganoderma** (reishi/lingzhi)
- **Cantharellus** (chanterelles)
- **Trametes** (bracket fungi)
- **Lichens** (Cladonia, Parmelia, etc.)

## 🤝 Contributing

Areas for improvement:
- Add more species-specific knowledge
- Optimize prompt templates
- Improve evaluation metrics
- Add uncertainty quantification

## ⚠️ Safety Warning

**This tool is for educational/research purposes only. Never consume wild mushrooms based solely on automated identification!**

---

## Getting Started Right Now:

1. **Install CLIP**: `pip install git+https://github.com/openai/CLIP.git`
2. **Generate prompts**: `python generate_clip_prompts.py`  
3. **Run demo**: `python clip_mushroom_classifier.py --demo`

The system will automatically use your mushroom dataset at `/zfs/ai4good/datasets/mushroom/`!