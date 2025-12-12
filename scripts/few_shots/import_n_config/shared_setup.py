"""
Shared imports used across hyperparameter tuning, training, and evaluation scripts.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import f1_score, confusion_matrix

# Optional tqdm
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *args, **kwargs): return x

try:
    import open_clip
except Exception as e:
    raise ImportError("open_clip (open_clip_torch) is required. Install with: pip install open_clip_torch") from e

__all__ = [
    "os", "sys", "json", "argparse", "Path", "List", "Dict", "Any", "Optional", "Tuple",
    "np", "pd", "Image", "torch", "nn", "DataLoader", "TensorDataset", "Dataset",
    "f1_score", "confusion_matrix", "tqdm", "open_clip"
]
