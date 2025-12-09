"""
Constants for the mushroom  few shot classifier training and inference.
"""

import torch
from pathlib import Path

###############################################################################
#                   helper variables
###############################################################################
FINAL_MODEL_BACKBONE_OPTIONS = ["ViT-B-32-quickgelu", "PE-Core-bigG-14-448"]
BACKBONE_TOGGLE = 1 # toggle 0/1 on which backbone to use for hyperparameter search    
VARIATION_RESULTS_DIR = "" if BACKBONE_TOGGLE else "_b32"

###############################################################################
#                   directories (adjust as needed)
###############################################################################

WORK_ENV = "/home/c/dkorot/AI4GOOD/ai4good-mushroom"     # <--------------- your directory for code and results
DATA_ROOT = "/zfs/ai4good/datasets/mushroom/merged_dataset"

DEFAULT_CACHE_DIR = WORK_ENV + "/features"

DATA_SPLITS_DIR = WORK_ENV + "/splits"
TRAIN_CSV = DATA_SPLITS_DIR + "/train.csv"
VAL_CSV = DATA_SPLITS_DIR + "/val.csv"
TEST_CSV = DATA_SPLITS_DIR + "/test.csv"

LABELS_N_PROMPT_DIR = WORK_ENV + "/data_prompts_label"
LABELS = LABELS_N_PROMPT_DIR + "/labels.tsv"
DEFAULT_PROMPTS_JSON = LABELS_N_PROMPT_DIR + "/delta_prompts.json"

RESULTS_DIR = WORK_ENV + "/results"
BEST_ALPHA_PATH = RESULTS_DIR + "/best_alpha" + VARIATION_RESULTS_DIR + ".json"
SAVE_JSON = RESULTS_DIR + "/final_model_eval" + VARIATION_RESULTS_DIR + ".json"
SAVE_PATH = RESULTS_DIR + "/final_model" + VARIATION_RESULTS_DIR + ".pt"

DEFAULT_TEXT_CACHE_DIR = Path(".cache_text_embeddings")
DEFAULT_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
#                   Hyperparameters
###############################################################################

FINAL_EPOCHS = 200
FINAL_BATCH_SIZE = 512

# sweep options
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

# final model hyperparameters (toggle at the top to switch between bigG and B32)
FINAL_BACKBONE = FINAL_MODEL_BACKBONE_OPTIONS[0]  
FINAL_PRETRAINED = DEFAULT_PRETRAINED[FINAL_BACKBONE]
FINAL_MODEL_TYPE = "linear+prompts"
FINAL_SHOTS = 100
FINAL_PROMPT_SET = "delta"
FINAL_LR = 3e-2
FINAL_WD = 1e-4 if BACKBONE_TOGGLE else 0.0

###############################################################################
#                          setup
###############################################################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_WORKERS = 64
BATCH_SIZE = 512

