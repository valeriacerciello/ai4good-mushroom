# CLIP Baseline (Zero-shot + Few-shot)

This script runs a quick CLIP sanity check on our mushroom dataset:
- **Zero-shot** with prompt templates
- Optional **few-shot linear probe** (logistic regression on frozen CLIP image embeddings)

## Environment
Activate your env (already set up):

```bash
conda activate /zfs/ai4good/student/<your_user_name>/conda_envs/mushroom
```
Install deps (first time only):

```bash
pip install open_clip_torch pillow pandas scikit-learn tqdm
```

## Data layout

```bash
/zfs/ai4good/datasets/mushroom/
  ├── merged_dataset/  # subfolders per species, images inside
  ├── train.csv
  ├── val.csv
  └── test.csv

```
Our CSVs have image_path,label and paths starting with /kaggle/working/....
The script automatically remaps those to the ZFS root you pass via --root.

## Cache (avoid home quota issues)
We cache weights on ZFS:
Example flag: `--cache_dir /zfs/ai4good/student/vcerci/cache/open_clip`
(If you ever see a “Disk quota exceeded” from `~/.cache/huggingface`, remove that folder and use `--cache_dir`.)

## How to run
### zero-shot

```console
cd /zfs/ai4good/student/vcerci/ai4good-mushroom

python scripts/clip_baseline.py \
  --train_csv /zfs/ai4good/datasets/mushroom/train.csv \
  --val_csv   /zfs/ai4good/datasets/mushroom/val.csv \
  --root      /zfs/ai4good/datasets/mushroom \
  --model ViT-B-16 --pretrained openai \
  --limit_per_class 20 \
  --cache_dir /zfs/ai4good/student/vcerci/cache/open_clip
```

# Dump features

## Output

`features/<backbone>/<split>.npz`

This file contains three arrays:

- `X` — a 2-D array of numbers with shape `[N, D]`. One row per image; `D` is the model’s feature size.
- `y` — a 1-D array of length `N` with the numeric class id for each image.
- `paths` — a 1-D array of length `N` with the relative filepath of each image.

These arrays are later used by evaluation code so we don’t have to re-read and re-process images.

## Inputs

- `--data-root`: folder that contains all images (in our repo this is `data/raw`, a symlink to the shared dataset).
- `--csv`: a CSV file for one split (`splits/train.csv`, `splits/val.csv`, or `splits/test.csv`) with two columns:
  - `filepath` — path to the image relative to --data-root
  - `label` — the class name (Latin species name)
- `--labels`: `labels.tsv` with three tab-separated columns: `class_id`, `latin_name`, `common_names(optional)`. This fixes the mapping from name → numeric id.
- `--backbone`: which CLIP variant to use (e.g., `ViT-B-32-quickgelu`).

## How to run

```console
python scripts/dump_features.py \
  --data-root data/raw \
  --csv splits/train.csv \
  --labels labels.tsv \
  --backbone "ViT-B-32-quickgelu"
```

```console
python scripts/dump_features.py \
  --data-root data/raw \
  --csv splits/test.csv \
  --labels labels.tsv \
  --backbone "ViT-B-32-quickgelu"
```

```console
python scripts/dump_features.py \
  --data-root data/raw \
  --csv splits/val.csv \
  --labels labels.tsv \
  --backbone "ViT-B-32-quickgelu"
```

# Eval Zero Shot

## Outputs

- A results table `results/metrics_<backbone>.csv` 
  - columns: `split, model, top1, top5, balanced_acc, macro_f1`
  - rows: one for each model (random, majority, zero-shot) on each split (val/test).

- Confusion matrix images

## Inputs

- `--data-root` : folder that contains all images (e.g., `data/raw`).
- `--train-csv`, `--val-csv`, `--test-csv` : CSVs listing images and labels for each split. Columns must be `filepath,label`, where `filepath` is relative to `--data-root`.
- `--labels` : `labels.tsv` mapping `class_id` ↔ `latin_name`. This fixes the class order used everywhere.
- --backbone : CLIP variant name (must match the one used when you created features, e.g., ViT-B-32-quickgelu).

## Metrics

For each split:
- Top-1 accuracy: fraction of images whose predicted class is exactly correct.
- Top-5 accuracy: fraction where the correct class appears among the top 5 highest scores.
- Balanced accuracy (macro recall): average of per-class recalls; treats all classes equally, regardless of how many images they have.
- Macro-F1: F1 score computed per class and then averaged; measures a balance of precision and recall across classes.

## How to run

```console
python scripts/eval_zero_shot.py \
  --data-root data/raw \
  --train-csv splits/train.csv \
  --val-csv   splits/val.csv \
  --test-csv  splits/test.csv \
  --labels    labels.tsv \
  --backbone  "ViT-B-32-quickgelu" \
  --splits    val test
```
