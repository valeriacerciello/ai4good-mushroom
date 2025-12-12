# AI4Good 2025 – Fine-Grained Mushroom Classification  
## Zero-shot, Few-shot & Prompt-aware CLIP/BLIP System

This repository contains our full codebase for the **AI4Good fine-grained mushroom identification** challenge.

We study CLIP and BLIP on a balanced dataset of **169 mushroom species** and build a full pipeline for:

- BLIP-based caption + attribute extraction  
- Automatic prompt construction  
- CLIP zero-shot and few-shot baselines  
- Prompt-aware few-shot methods (prototype / linear probe)  
- Hyperparameter sweeps and model selection  
- Final model training, evaluation **and inference**  
- BLIP and LoRA baselines  
- Qualitative analysis and error diagnostics  

The entire pipeline is documented and reproducible.

For script-level details see:

- `SCRIPTS_OVERVIEW.md`  
- `PIPELINE_OVERVIEW.md` (diagram of the whole system)

---

## Quickstart: reproduce the final model

From a clean clone, the **minimal path** to the final model is:

```bash
# 0. Create environment and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_blipclip.txt

# 1. Make sure the dataset is available (see Section 1 below)

# 2. (Optional but recommended) precompute CLIP features with scripts/dump_features.py

# 3. Train final few-shot model with per-class α
python -m scripts.few_shots.train_best_model

# 4. Evaluate on val/test and write metrics to final_model_eval.json
python -m scripts.few_shots.eval_final_model

# 5. Run inference on a new image
python -m scripts.few_shots.eval_final_model --predict path/to/mushroom_image.jpg
```

All commands must be run from the repository root (the ai4good-mushroom folder).

---

## 0. Installation

Requires Python ≥ 3.10, PyTorch ≥ 2.0, OpenCLIP, and HuggingFace Transformers.

We recommend using a virtual environment:

```bash
git clone https://github.com/valeriacerciello/ai4good-mushroom.git
cd ai4good-mushroom

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Most experiments are designed to run on a GPU cluster.

---

## 1. Dataset

We assume the mushroom dataset is available under something like:

```text
/zfs/ai4good/datasets/mushroom/merged_dataset/
    ├── <class_name>/
    │     ├── *.jpg / *.jpeg
```

You can provide a custom root path via an environment variable:

```bash
export ROOT_MUSHROOM=/path/to/your/mushroom_dataset
```

Train/validation/test splits are stored in splits/:
- splits/train.csv
- splits/val.csv
- splits/test.csv

A consistent label mapping is provided by:
- data_prompts_label/labels.tsv

Note: BLIP captions, attributes and prompts are already precomputed
and stored in `data/.` You do not need to re-run BLIP to reproduce
the main CLIP results.

⸻

## 2. Dump CLIP visual features (once per backbone)

We precompute CLIP image embeddings for each backbone and split. Example
for the OpenAI ViT-B/32 backbone:

```bash
python scripts/dump_features.py \
    --data-root "${ROOT_MUSHROOM:-/zfs/ai4good/datasets/mushroom/merged_dataset}" \
    --csv splits/train.csv \
    --labels data_prompts_label/labels.tsv \
    --backbone ViT-B-32 \
    --pretrained openai
```

Repeat the call with `splits/val.csv` and `splits/test.csv` as needed.

For the final model, you should also dump features for the
`PE-Core-bigG-14-448` backbone (see `scripts/dump_features.py --help`
for the exact backbone name and pretrained tag). Features are cached
under `features/<backbone>/` and reused by all few-shot scripts.

⸻

## 3. (Optional) Rebuild BLIP captions → attributes → prompts

These steps were run by us already; the outputs live in data/prompts/.
Run them only if you want to regenerate everything from raw images:

```bash
python scripts/blip_caption_bank.py \
    --root "${ROOT_MUSHROOM:-/zfs/ai4good/datasets/mushroom/merged_dataset}"
python scripts/extract_attributes.py
python scripts/build_text_prompts.py
```

This produces BLIP captions, cleaned attributes and prompt JSONs used in
the CLIP experiments.

⸻

## 4. Zero-shot CLIP with all prompt sets

To evaluate zero-shot CLIP across prompt families and backbones:

```bash
python scripts/eval_zero_shot.py \
    --data-root "${ROOT_MUSHROOM:-/zfs/ai4good/datasets/mushroom/merged_dataset}" \
    --labels data_prompts_label/labels.tsv \
    --train-csv splits/train.csv \
    --val-csv splits/val.csv \
    --test-csv splits/test.csv
```

Metrics (top-1/top-5, balanced accuracy, macro-F1) are written to
`results/metrics/` as JSON files.

⸻

## 5. Few-shot classification with prompt-aware training

The few-shot pipeline combines CLIP image features with text embeddings
built from prompts. Per-class mixing coefficients (\alpha_c)
control how much each class relies on image vs text, and are optimised
using a sweep.

We support K-shot learning for (K \in {0, 1, 5, 10, 20, 50, 100})
((K = 0) corresponds to pure zero-shot).

### 5a. (Optional) Hyperparameter sweep & prompt selection

We ran a comprehensive grid search over:
- Few-shot counts: 0, 1, 5, 10, 20, 50, 100
- Prompt templates: "ensemble", "v1", "names", "delta"
- Learning rates: {1e-3, 3e-3, 3e-2, 3e-1}
- Weight decays: {0, 1e-4, 5e-4}
- Multiple OpenCLIP backbones

To reproduce the sweep:

```bash
python scripts/few_shots/few_shot_hyper_test.py
```

The sweep stores detailed results and per-class (\alpha) values in:
- results/few_shot_overall_results.json

The notebook:
- scripts/few_shots/hyperparameter_analysis.ipynb

can be used to inspect and select the best configuration.

### 5b. Train final model (few-shot + per-class α)

The final classifier is trained by:

```bash 
python -m scripts.few_shots.train_best_model
```

This script:
- Loads the best per-class alpha dictionary (best_alpha)
- Loads label names and cached CLIP features for the chosen backbone
- Samples the final support set (e.g., 100 shots per class)
- Mixes image and text embeddings using (\alpha_c)
- Trains a linear probe on the mixed features
- Saves the checkpoint to SAVE_PATH (default: results/final_model.pt)

The checkpoint contains:
- linear_head parameters
- Backbone name and pretrained tag
- Prompt set identifier
- best_alpha dict
- label_names

The final configuration used in the paper is:
- Backbone: PE-Core-bigG-14-448
- Model type: linear probe + prompts
- Shots per class: 100
- Prompt set: "delta"
- Per-class (\alpha_c) fusion

### 5c. Evaluate final model

Evaluate the trained model on validation and test sets:

```bash
python -m scripts.few_shots.eval_final_model
# equivalent:
# python -m scripts.few_shots.eval_final_model --eval
```

This will:
- Load final_model.pt from MODEL_PATH
- Load or compute CLIP features for val/test
- Apply the same per-class (\alpha_c) mixing
- Compute and print:
    - Top-1 accuracy
    - Top-5 accuracy
    - Balanced accuracy
    - Macro-F1

The metrics are also written to SAVE_JSON (default:
`results/final_model_eval.json`).

⸻

## 6. Inference on new images

Once `final_model.pt` exists, you can classify individual mushroom
images via:

```bash
python -m scripts.few_shots.eval_final_model \
    --predict /path/to/mushroom_image.jpg
```

This command:
- Loads the final model and configuration
- Encodes the input image with the same CLIP backbone
- Mixes the image embedding with each class text embedding using (\alpha_c)
- Applies the linear head and prints the predicted class label

You can also evaluate and predict in a single run:

```bash
python -m scripts.few_shots.eval_final_model \
    --eval \
    --predict /path/to/mushroom_image.jpg
```

For batch inference on many images, you can adapt `predict()` in
`scripts/few_shots/eval_final_model.py` to iterate over a directory.

⸻

### External repository

`external/ecovision_mushroom/` is the original mushroom classification
baseline repository provided to us. It is included for reference
only and is not required to reproduce our main experiments.

⸻

### Repository structure

```text
.
├── _legacy/                      # Early work iterations
├── scripts/                      # All runnable scripts (zero-shot, few-shot, BLIP, LoRA)
├── code/                         # Helper code (evaluation, plotting, resnet training, prompts)
├── data_prompts_label/           # Label mapping + delta prompts (Generated or downloaded from one drive)
├── splits/                       # Train/val/test CSV splits (Generated or downloaded from one drive)
├── features/                     # Cached CLIP image / text features (generated)
├── results/                      # Metrics, confusion matrices, qualitative figures (Generated or downloaded)
└── external/ecovision_mushroom/  # Original baseline repo (reference only)
```


⸻

### License

MIT License.

⸻

### Contact

This work was created for the AI4Good 2025 course, University of
Zurich.

For questions or issues, please contact the authors or open a GitHub
issue.

