"""
Header for train_best_model.py
"""

from scripts.few_shots.import_n_config.shared_setup import *
from scripts.few_shots.import_n_config.constants import (
    FINAL_BACKBONE, FINAL_PRETRAINED, FINAL_MODEL_TYPE,
    FINAL_SHOTS, FINAL_PROMPT_SET,
    FINAL_LR, FINAL_WD, FINAL_EPOCHS, FINAL_BATCH_SIZE,
    BEST_ALPHA_PATH, SAVE_PATH, DEVICE,
)
from scripts.few_shots.few_shot_hyper_test import (
    ensure_features,
    load_labels,
    get_text_embeddings,
    train_linear_probe,
    encode_images,
    DEFAULT_CACHE_DIR,
    DATA_ROOT, TRAIN_CSV, VAL_CSV, TEST_CSV, LABELS,
)

__all__ = [name for name in globals().keys() if not name.startswith("_")]