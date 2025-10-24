"""Utility functions for loading prompt JSON files used across scripts/notebooks.

Exports:
 - DELTA_PROMPTS: absolute path to the delta prompts JSON file
 - load_prompts(prompt_path, labels): load and normalize prompts per label
"""
from pathlib import Path
import json
import os

# Absolute path used across the project
DELTA_PROMPTS = "/home/c/dkorot/AI4GOOD/ai4good-mushroom/data_prompts_label/delta_prompts.json"


def load_prompts(prompt_path, labels):
    """Load custom prompts from JSON file.

    Returns a dict mapping each label to a list of prompts. If the file does not
    exist or a label is missing, a default prompt of the form "a photo of {label}"
    is returned for that label.
    """
    if prompt_path is None:
        prompt_path = DELTA_PROMPTS
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        return {label: data.get(label, [f"a photo of {label}"]) for label in labels}
    return {label: [f"a photo of {label}"] for label in labels}
