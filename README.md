# AI4Good 2025 – Fine-Grained Mushroom Classification
## Zero-shot, Few-shot & Prompt-aware CLIP/BLIP System
This repository contains our full codebase for the AI4Good fine-grained mushroom identification challenge.

It includes:

- BLIP-based caption + attribute extraction
- Automatic prompt construction
- CLIP zero-shot and few-shot baselines
- Prompt-aware few-shot methods (prototype/linear)
- Hyperparameter sweeps and model selection
- Final model training and evaluation
- BLIP and LoRA baselines
- Qualitative analysis tools
The entire pipeline is documented and reproducible.

For a script-by-script description, see:
- `SCRIPTS_OVERVIEW.md`
- `PIPELINE_OVERVIEW.md` (diagram of the whole system)

### Installation

```console
pip install -r requirements_blipclip.txt
```
Requires Python ≥ 3.10, PyTorch ≥ 2.0, OpenCLIP, and HuggingFace Transformers.

We recommend running the heavy pipelines on a GPU cluster.

### Dataset Structure

Place the mushroom dataset under:

```console
/zfs/ai4good/datasets/mushroom/
    |-- <class_name>/
    |       |-- *.jpg
    |       |-- *.jpeg
```

Splits are provided in `splits/` (`train.csv`, `val.csv`, `test.csv`).

A consistent label mapping is provided by `data_prompts_label/labels.tsv`.

### 1. Dump CLIP visual features (once per backbone)

```console
python scripts/dump_features.py \
    --data-root /zfs/ai4good/datasets/mushroom \
    --csv splits/train.csv \
    --labels data_prompts_label/labels.tsv \
    --backbone ViT-B-32 --pretrained openai
```
Repeat for `val.csv` and `test.csv`.

### 2. Build BLIP captions → attributes → prompts

Run once:

```console
python scripts/blip_caption_bank.py --root /zfs/ai4good/datasets/mushroom
python scripts/extract_attributes.py
python scripts/build_text_prompts.py
```
Outputs are stored in `data/prompts/`.

### 3. Evaluate zero-shot CLIP with all prompt sets

```console
python scripts/eval_zero_shot.py \
    --data-root /zfs/ai4good/datasets/mushroom \
    --labels data_prompts_label/labels.tsv \
    --train-csv splits/train.csv \
    --val-csv splits/val.csv \
    --test-csv splits/test.csv
```
Metrics appear in `results/metrics/`.

### 4. Few-Shot Classification with Prompt-Aware Training

The few-shot script combines CLIP image features with text embeddings and prompt engineering to enable robust mushroom classification from minimal labeled examples. The pipeline automatically optimizes the blending of visual and textual information using per-class alpha weights, supporting K-shot learning for K ∈ {0, 1, 5, 10, 20, 50, 100} (where K=0 is zero-shot using only text).

#### Step 4a: Hyperparameter Sweep & Prompt Selection

Run a comprehensive grid search over few-shot counts, prompts, and linear probe hyperparameters:

```console
python scripts/few_shots/few_shot_hyper_test.py
```

This sweep evaluates:
- **Few-shot counts**: 0, 1, 5, 10, 20, 50, 100 examples per class
- **Prompt templates**: "ensemble", "v1", "names", "delta"
- **Learning rates**: [1e-3, 3e-3, 3e-2, 3e-1]
- **Weight decays**: [0, 1e-4, 5e-4]
- **Multiple OpenCLIP backbones** for robustness

The sweep produces:
- Per-class alpha accuracies
- Performance metrics across all configurations
- Detailed results in `results/few_shot_overall_results.json`

Use `hyperparameter_analysis.ipynb` in the `few_shots/` directory to visualize sweep results and select the best configuration and alpha parameters per class.

#### Step 4b: Train Final Model

Train the final classifier using the best hyperparameters from the sweep:

```console
python scripts/few_shots/train_best_model.py
```

This step:
1. Loads optimal hyperparameters and per-class alpha values
2. Trains a linear classifier on the mixed feature space
3. Saves the trained model as `final_model.pt` (with backbone suffix if applicable, e.g., `final_model_b32.pt`)

#### Step 4c: Evaluate Final Model

Assess model performance on validation and test splits:

```console
python scripts/few_shots/eval_final_model.py
```

Reports comprehensive metrics:
- **Confusion Matrices**: Per-class prediction breakdown
- **Top-1 Accuracy**: Fraction of correct predictions
- **Top-5 Accuracy**: Fraction where true label appears in top-5 candidates
- **Balanced Accuracy**: Per-class recall averaged across all classes (handles class imbalance)
- **Macro F1**: F1 score averaged per class

Results are saved to `results/final_model_eval.json` (with backbone suffix if applicable).

#### Step 5. Few Shots Inference on New Images

Once the final model is trained, you can use it to classify new mushroom images:

```console
python scripts/few_shots/eval_final_model.py --predict /path/to/mushroom_image.jpg
```

This command:
- Loads the trained `final_model.pt` model
- Processes the input image using the same CLIP backbone
- Returns top-1 and top-5 predictions with confidence scores
- Outputs classification results to the console

For batch inference on multiple images, modify the evaluation script to iterate over image directories.

## External Repositories

`ecovision_mushroom/` is the original mushroom classification baseline repository, included for reference only.

We do not run or modify it for our main experiments.

## Repository Structure

```console
.
├── SCRIPTS_OVERVIEW.md        # Full documentation for each script
├── PIPELINE_OVERVIEW.md       # Mermaid diagram of the entire pipeline
├── scripts/                   # All runnable scripts (zero-shot, few-shot, BLIP, LoRA)
├── data/                      # BLIP attributes, prompts
├── data_prompts_label/        # Label mapping + delta prompts
├── splits/                    # Train/val/test CSV splits
├── features/                  # Cached CLIP image features + text caches
├── results/                   # Metrics, confusion matrices, qualitative figures
├── src/                       # Utilities (caption cleaning)
└── external/ecovision_mushroom/  # Original baseline repo (reference only)
```

## License

MIT License. 

## Contact

This work was created for the AI4Good 2025 course, University of Zurich.
For questions, contact the team members or open an issue.






