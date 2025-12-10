"""
header for few_shot_hyper_test.py
"""

from scripts.few_shots.import_n_config.shared_setup import *
from scripts.few_shots.import_n_config.constants import (
    DEFAULT_PRETRAINED, SHOTS, ALPHAS, PROMPT_SET, LR_GRID, WD_GRID,
    DATA_ROOT, TRAIN_CSV, VAL_CSV, TEST_CSV, LABELS, RESULTS_DIR,
    DEFAULT_CACHE_DIR, DEFAULT_PROMPTS_JSON, DEFAULT_TEXT_CACHE_DIR,
    MAX_WORKERS, BATCH_SIZE,
)

# Torch & environment tuning
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 16)))
torch.set_float32_matmul_precision("high")

__all__ = [name for name in globals().keys() if not name.startswith("_")]
