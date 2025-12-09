# Scripts Overview – How to Use This Folder

This directory contains all executable pipelines for:
- BLIP caption generation
- Prompt construction
- CLIP feature extraction
- Zero-shot and few-shot evaluation
    - Hyperparameter sweeps
    - hyperparameter notebook analysis
    - Final model training
- Baselines (BLIP, LoRA)
- Qualitative figures
For detailed script descriptions, see `../SCRIPTS_OVERVIEW.md`.

# Typical Workflow

## 1. BLIP Caption → Attribute → Prompt Pipeline

```console
python blip_caption_bank.py --root <dataset>
python extract_attributes.py
python build_text_prompts.py
```
## 2. Dump CLIP Image Features

```console
python dump_features.py --csv ../splits/train.csv \
    --labels ../data_prompts_label/labels.tsv \
    --backbone ViT-B-32
```
Repeat for val/test.

## 3. Zero-Shot CLIP Evaluation

```console
python eval_zero_shot.py \
    --train-csv ../splits/train.csv \
    --val-csv ../splits/val.csv \
    --test-csv ../splits/test.csv
```

## 4. Few-Shot Evaluation (simple)
Adjust the input and out directories inside the files.

### Hyperparameter Sweeps

```console
python few_shots/hypertuning/few_shot_hyper_test.py
```
Produces `best_alpha.json`.

### Train + Evaluate Final Model

```console
python few_shots/hypertuning/train_best_model.py --train
python few_shots/hypertuning/eval_final_model.py
```







