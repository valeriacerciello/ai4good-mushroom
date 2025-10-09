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
  ├── merged_dataset/  # subfolders per    species, images inside
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

```bash
cd /zfs/ai4good/student/vcerci/ai4good-mushroom

python scripts/clip_baseline.py \
  --train_csv /zfs/ai4good/datasets/mushroom/train.csv \
  --val_csv   /zfs/ai4good/datasets/mushroom/val.csv \
  --root      /zfs/ai4good/datasets/mushroom \
  --model ViT-B-16 --pretrained openai \
  --limit_per_class 20 \
  --cache_dir /zfs/ai4good/student/vcerci/cache/open_clip
```
