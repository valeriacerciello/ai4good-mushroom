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

### 4. Run few-shot + prompt-aware sweeps

```console
python scripts/few_shots/hypertuning/few_shot_hyper_test.py --fast
```
Full sweep outputs:
- Global metrics
- Per-class metrics
- `best_alpha.json`

### 5. Train final model

```console
python scripts/few_shots/hypertuning/train_best_model.py --train
```

The final model is stored as `final_model.pt`.

Evaluate:

```console
python scripts/few_shots/hypertuning/eval_final_model.py
```

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






