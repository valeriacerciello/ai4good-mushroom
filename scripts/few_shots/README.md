# Few-Shot Learning Script

This directory contains the complete Script for few-shot learning with OpenCLIP on the mushroom classification dataset. 

## Directory Structure

### Main Scripts

- **`few_shot_hyper_test.py`** - Hyperparameter sweep script
  - Searches over few-shot counts, prompts, and linear probe hyperparameters
  - Tests multiple OpenCLIP backbones
  - Generates performance reports and best hyperparameters
  - Outputs: Sweep results JSONs, best alpha weights

- **`train_best_model.py`** - Final model training script
  - Loads best hyperparameters from sweep
  - Trains a linear+prompts model on full training set
  - Mixes image embeddings with text embeddings using per-class alpha weights
  - Outputs: `final_model.pt`, model evaluation metrics

- **`eval_final_model.py`** - Evaluation script
  - Evaluates trained model on validation and test sets
  - Computes top-1 and top-5 accuracy
  - Generates confusion matrices
  - Calculates balanced accuracy and macro F1 scores
  - Provides inference pipeline for single-image prediction
  - Outputs: Performance JSON with detailed metrics

### Configuration & Imports

- **`import_n_config/`** - Centralized configuration module
  - `constants.py` - All hyperparameters, paths, and model settings
  - `shared_setup.py` - Common imports and dependencies
  - `train_setup.py` - Training-specific imports
  - `eval_setup.py` - Evaluation-specific imports
  - `hyper_setup.py` - Hyperparameter sweep-specific imports
  - See `import_n_config/README.md` for detailed documentation

### Additional Files

- **`hyperparameter_analysis.ipynb`** - Jupyter notebook for analyzing sweep results
  - Visualizes performance across different hyperparameter combinations
  - Plots prompt effectiveness and few-shot scaling

## Workflow

### 1. Hyperparameter Sweep

```bash
python -m scripts.few_shots.few_shot_hyper_test
```

This runs a comprehensive grid search:
- Few-shot counts: 0, 1, 5, 10, 20, 50, 100
- Prompt sets: "ensemble", "v1", "names", "delta"
- Learning rates: [1e-3, 3e-3, 3e-2, 3e-1]
- Weight decays: [0, 1e-4, 5e-4]
- Multiple OpenCLIP backbones

**Outputs:**
- Detailed sweep results to `results/`

### 2. Train Final Model

```bash
python -m scripts.few_shots.train_best_model
```

Uses best hyperparameters to train final model:
- Mixes image and text embeddings per class
- Trains linear probe on few shot of training set
- Saves model to `results/final_model.pt`
  - If the analyzed backbone was "ViT-B-32-quickgelu", that a `_b32` suffix is added to the filename.


### 3. Evaluate Model

```bash
python -m scripts.few_shots.eval_final_model
```

Evaluates the trained model:
- Top-1 and top-5 accuracy
- Confusion matrices
- Balanced accuracy (handles class imbalance)
- Macro F1 score

**Outputs:**
- Performance metrics to `results/final_model_eval.json`
  - If the analyzed backbone was "ViT-B-32-quickgelu", that a `_b32` suffix is added to the filename.
- Console reports with detailed breakdown of the model

**Additional usage (inference):**
```bash
# Predict a single image using the evaluation/inference script
python -m scripts.few_shots.eval_final_model --predict /path/to/image.jpg
```

Note: inference and evaluation can be run together.

## Key Concepts

### Few-Shot Learning
The pipeline supports K-shot learning (K ∈ {0, 1, 5, 10, 20, 50, 100}) where models learn from limited labeled examples per class. Zero-shot (0-shot) uses only text embeddings.

### Prompt Engineering
Multiple prompt templates are evaluated:
- **ensemble**: Combined prompts for robustness
- **v1**: Generic mushroom photo prompts
- **names**: Scientific/common names
- **delta**: Optimized prompts via hyperparameter search

### Feature Mixing
Text embeddings are mixed with image embeddings using per-class alpha weights:
```
mixed_feature = a * text_embedding + (1-a) * image_embedding
```

The best alpha for each class is found during hyperparameter sweep.


## Configuration

All configuration is centralized in `import_n_config/constants.py`. Key adjustable parameters:

```python
# Model selection
FINAL_BACKBONE = "ViT-B-32-quickgelu"  # or "PE-Core-bigG-14-448"
BACKBONE_TOGGLE = 1  # 0/1 to switch backbones

# Paths (adjust for your setup)
WORK_ENV = "/home/c/dkorot/AI4GOOD/ai4good-mushroom"
DATA_ROOT = "/zfs/ai4good/datasets/mushroom/merged_dataset"

# adjust for parallelism and memory if nescessary
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WORKERS = 64
BATCH_SIZE = 512
```

See `import_n_config/README.md` for complete configuration documentation.

## Dependencies

All dependencies are imported through the `import_n_config/` module:
- PyTorch & CUDA
- OpenCLIP (`open_clip_torch`)
- scikit-learn
- NumPy, Pandas
- PIL/Pillow
- tqdm

Install with:
```bash
pip install torch torchvision open_clip_torch scikit-learn numpy pandas pillow tqdm
```

## Outputs

### Results Directory Structure
```
results/
├── best_alpha.json              # Per-class alpha weights
├── final_model.pt               # Trained model checkpoint
├── final_model_eval.json        # Final evaluation metrics
├── sweep_results_*.json         # Per-configuration results
└── logs/                        # Training and sweep logs
```

### Metrics
- **Top-1 Accuracy**: Fraction of correct top-1 predictions
- **Top-5 Accuracy**: Fraction where true label in top-5
- **Confusion Matrix**: Per-class predictions breakdown
- **Balanced Accuracy**: Accuracy averaged per class (handles imbalance)
- **Macro F1**: F1 score averaged across all classes

## Troubleshooting

**CUDA Out of Memory:**
- run code in `/zfs/ai4good/student/<your_name>` diretctory
  - make a symbolic link from a dir in you home, for easier directory management if needed
- Reduce `FINAL_BATCH_SIZE` in `constants.py`
- Reduce `MAX_WORKERS` for feature caching
- Clear feature cache in `features/` directory

**Slow Hyperparameter Sweep:**
- Reduce number of SHOTS or ALPHAS in `constants.py`
- Reduce `SHOTS` or `ALPHAS` ranges temporarily

**Import Errors:**
- Verify `import_n_config/` is discoverable (check `__init__.py`)
- Ensure all dependencies installed
- Check `WORK_ENV` and `DATA_ROOT` paths in `constants.py`

## Notes

- Feature caching significantly speeds up sweeps (stored in `features/` and `.cache_text_embeddings/`)
- PyTorch optimizations are enabled for faster training (matmul precision, cudnn benchmarking)
- Multi-threading is tuned for efficient hyperparameter sweeps
