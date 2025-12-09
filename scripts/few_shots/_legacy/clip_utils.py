#!/usr/bin/env python3
"""
CLIP utilities: get_text_embeddings and prompt templates
(optimized version — same interface and return types)

Changes:
- Batches + DataLoader-based tokenization
- torch.compile() for encode_text (PyTorch 2.x)
- Disk caching of text embeddings (.npz)
- Avoid duplicate prompt encoding
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from pathlib import Path
import hashlib
import json
import os
import numpy as np

try:
    import open_clip
except Exception:
    open_clip = None

from scripts.generate_clip_prompts import MushroomPromptGenerator

# ---------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------
SHORT_PROMPTS = [
    "a photo of {label}",
    "a close-up of {label} mushroom",
    "a photograph of {label}",
    "a picture of {label} in nature",
    "close-up of {label} cap",
]

NAME_PROMPT_TEMPLATES = [
    "a photo of {name}",
    "{name} mushroom",
    "{name} fungus",
]

# Cache directory for text embeddings
CACHE_DIR = Path("/zfs/ai4good/student/dkorot/ai4good-mushroom/cache_text")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Helper: build name list with MushroomPromptGenerator
# ---------------------------------------------------------------------
def _build_name_list(label: str, mg: MushroomPromptGenerator) -> List[str]:
    """Return list of names for a species: latin + common names if available."""
    names = [label]
    if label in mg.species_knowledge:
        info = mg.species_knowledge[label]
        for n in info.get("common_names", []):
            if n not in names:
                names.append(n)
    return names


# ---------------------------------------------------------------------
# Optimized text embedding function
# ---------------------------------------------------------------------
def get_text_embeddings(
    labels: List[str],
    prompt_set: str = "v1",
    model_name: str = "ViT-B/32",
    device: Optional[str] = None,
    batch_size: int = 128,
    include_common_names: bool = False,
    common_prompts_path: Optional[str] = None,
    pretrained: str = "openai",
) -> Dict[str, Any]:
    """
    Compute text embeddings for given labels using CLIP (optimized).

    ✅ Same output format as before:
        {
          label: {
            'prompts': [str],
            'prompt_embeddings': Tensor (N_prompts x D),
            'label_embedding': Tensor (D,)
          }
        }
    """
    if open_clip is None:
        raise ImportError("open_clip not installed. Use: pip install open_clip_torch")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load and optionally compile model
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    try:
        model.encode_text = torch.compile(model.encode_text)
    except Exception:
        pass

    mg = MushroomPromptGenerator()

    # Load prebuilt prompts (if provided)
    common_prompts = None
    if include_common_names:
        path = common_prompts_path or Path(__file__).parent / "enhanced_with_common_names_prompts.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                common_prompts = json.load(f)

    results: Dict[str, Any] = {}

    for label in labels:
        # print(f"Encoding prompts for label: {label}")

        # ------------------------------------------------------------------
        # Prompt generation
        # ------------------------------------------------------------------
        if common_prompts and label in common_prompts:
            prompts = list(common_prompts[label])
        else:
            if prompt_set == "v1":
                prompts = [t.format(label=label) for t in SHORT_PROMPTS]
            elif prompt_set == "names":
                names = _build_name_list(label, mg)
                prompts = [tmpl.format(name=n) for n in names for tmpl in NAME_PROMPT_TEMPLATES]
            elif prompt_set == "ensemble":
                prompts = [t.format(label=label) for t in SHORT_PROMPTS]
                names = _build_name_list(label, mg)
                for n in names:
                    for tmpl in NAME_PROMPT_TEMPLATES:
                        p = tmpl.format(name=n)
                        if p not in prompts:
                            prompts.append(p)
            else:
                raise ValueError(f"Unknown prompt_set: {prompt_set}")

        # ------------------------------------------------------------------
        # Caching: compute hash key for label + prompt list
        # ------------------------------------------------------------------
        prompt_hash = hashlib.sha1(("".join(prompts) + model_name + pretrained).encode()).hexdigest()
        cache_path = CACHE_DIR / f"{label}_{prompt_hash}.npz"

        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            results[label] = {
                "prompts": list(cached["prompts"]),
                "prompt_embeddings": torch.tensor(cached["prompt_embeddings"]),
                "label_embedding": torch.tensor(cached["label_embedding"]),
            }
            # print(f"Loaded text embeddings for {label}")
            continue

        # ------------------------------------------------------------------
        # Batched encoding
        # ------------------------------------------------------------------
        all_embeddings = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                text_tokens = tokenizer(batch_prompts).to(device)
                emb = model.encode_text(text_tokens)
                emb = F.normalize(emb, dim=-1)
                all_embeddings.append(emb.cpu())

        if all_embeddings:
            prompt_embeddings = torch.cat(all_embeddings, dim=0)
            label_embedding = F.normalize(prompt_embeddings.mean(dim=0), dim=-1)
        else:
            dim = model.text_projection.shape[1] if hasattr(model, "text_projection") else 512
            prompt_embeddings = torch.empty((0, dim))
            label_embedding = torch.zeros(dim)

        results[label] = {
            "prompts": prompts,
            "prompt_embeddings": prompt_embeddings,
            "label_embedding": label_embedding.cpu(),
        }

        # Save to cache
        np.savez_compressed(
            cache_path,
            prompts=np.array(prompts, dtype=object),
            prompt_embeddings=prompt_embeddings.numpy(),
            label_embedding=label_embedding.numpy(),
        )

    return results
