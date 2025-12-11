"""
Constants for the mushroom few-shot classifier training and inference.
"""

import os
import torch
from pathlib import Path

###############################################################################
#                   Helper variables
###############################################################################

# Final model backbones:
#   index 0 -> ViT-B-32-quickgelu  (smaller, used for ablations)
#   index 1 -> PE-Core-bigG-14-448 (large backbone used for the final model)
FINAL_MODEL_BACKBONE_OPTIONS = ["ViT-B-32-quickgelu", "PE-Core-bigG-14-448"]

# Toggle which backbone configuration to treat as "final":
#   0 -> ViT-B-32 (results get suffix "_b32")
#   1 -> bigG     (results have no suffix)
BACKBONE_TOGGLE = 1
VARIATION_RESULTS_DIR = "" if BACKBONE_TOGGLE else "_b32"

###############################################################################
#                   Directories (reproducible paths)
###############################################################################
# We avoid hard-coded user paths and infer the repo root from this file.
# Layout assumed:
#   <repo_root>/
#       scripts/few_shots/import_n_config/constants.py  (this file)
###############################################################################

# Repo root (ai4good-mushroom/)
REPO_ROOT = Path(__file__).resolve().parents[3]

# Optional override for working env via env var (default: repo root)
WORK_ENV = Path(os.environ.get("AI4GOOD_WORK_ENV", str(REPO_ROOT)))

# Dataset root:
#   - if ROOT_MUSHROOM is set, we use that
#   - otherwise we fall back to the cluster path used in our experiments
DATA_ROOT = os.environ.get(
    "ROOT_MUSHROOM",
    "/zfs/ai4good/datasets/mushroom/merged_dataset",
)

# Where to cache CLIP image features
DEFAULT_CACHE_DIR = str(WORK_ENV / "features")

# Splits (CSV files tracked in the repo)
DATA_SPLITS_DIR = WORK_ENV / "splits"
TRAIN_CSV = str(DATA_SPLITS_DIR / "train.csv")
VAL_CSV = str(DATA_SPLITS_DIR / "val.csv")
TEST_CSV = str(DATA_SPLITS_DIR / "test.csv")

# Labels + prompt configs
LABELS_N_PROMPT_DIR = WORK_ENV / "data_prompts_label"
LABELS = str(LABELS_N_PROMPT_DIR / "labels.tsv")
DEFAULT_PROMPTS_JSON = str(LABELS_N_PROMPT_DIR / "delta_prompts.json")

# Results directory (metrics, final model, etc.)
RESULTS_DIR = WORK_ENV / "results"
SWEEP_OUTPUT = str(RESULTS_DIR / "few_shot_overall_results.json")
BEST_ALPHA_PATH = str(RESULTS_DIR / f"best_alpha{VARIATION_RESULTS_DIR}.json")
SAVE_JSON = str(RESULTS_DIR / f"final_model_eval{VARIATION_RESULTS_DIR}.json")
SAVE_PATH = str(RESULTS_DIR / f"final_model{VARIATION_RESULTS_DIR}.pt")

# Text embedding cache (local to the repo/work env)
DEFAULT_TEXT_CACHE_DIR = WORK_ENV / ".cache_text_embeddings"
DEFAULT_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
#                   Hyperparameters
###############################################################################

FINAL_EPOCHS = 200
FINAL_BATCH_SIZE = 512

# Sweep options
DEFAULT_PRETRAINED = {
    "PE-Core-bigG-14-448": "meta",
    "ViT-gopt-16-SigLIP2-384": "webli",
    "ViT-H-14-378-quickgelu": "dfn5b",
    "ViT-B-32-quickgelu": "openai",
    "ViT-L-14": "openai",
    "ViT-B-16": "openai",
}

SHOTS = [0, 1, 5, 10, 20, 50, 100]
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PROMPT_SET = ["ensemble", "v1", "names", "delta"]
LR_GRID = [1e-3, 3e-3, 3e-2, 3e-1]
WD_GRID = [0, 1e-4, 5e-4]

# Final model hyperparameters
# Use BACKBONE_TOGGLE to pick between B/32 and bigG
FINAL_BACKBONE = FINAL_MODEL_BACKBONE_OPTIONS[BACKBONE_TOGGLE]
FINAL_PRETRAINED = DEFAULT_PRETRAINED[FINAL_BACKBONE]
FINAL_MODEL_TYPE = "linear+prompts"
FINAL_SHOTS = 100
FINAL_PROMPT_SET = "delta"
FINAL_LR = 3e-2
FINAL_WD = 1e-4 if BACKBONE_TOGGLE else 0.0

###############################################################################
#                          Setup
###############################################################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_WORKERS = 64
BATCH_SIZE = 512
