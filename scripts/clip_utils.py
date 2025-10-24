"""
CLIP utilities: get_text_embeddings and prompt templates

Provides:
- get_text_embeddings(labels, prompt_set='v1', model_name='ViT-B/32', device=None)
  returns per-label embeddings and prompt lists.

Prompt sets:
- 'v1' : short prompt templates (concise)
- 'names' : Latin + common names (if available)
- 'ensemble' : combine v1 and names

This module uses the existing generate_clip_prompts.MushroomPromptGenerator
for species common-name lookup when available.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

try:
    import open_clip
except Exception:
    open_clip = None

from scripts.generate_clip_prompts import MushroomPromptGenerator

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


def _build_name_list(label: str, mg: MushroomPromptGenerator) -> List[str]:
    """Return list of names for a species: latin + common names if available"""
    names = [label]
    if label in mg.species_knowledge:
        info = mg.species_knowledge[label]
        for n in info.get('common_names', []):
            if n not in names:
                names.append(n)
    return names


def get_text_embeddings(labels: List[str], prompt_set: str = "v1",
                        model_name: str = "ViT-B/32", device: Optional[str] = None,
                        batch_size: int = 128, include_common_names: bool = False,
                        common_prompts_path: Optional[str] = None, pretrained="openai") -> Dict[str, Any]:
    """
    Compute text embeddings for given labels using CLIP.

    Returns a dict mapping label -> {
        'prompts': [str],
        'prompt_embeddings': tensor (N_prompts x D),
        'label_embedding': tensor (D,)  # mean of prompt embeddings
    }
    """
    if open_clip is None:
        raise ImportError("open_clip library not available. install with: pip install open_clip_torch")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # model, _ = clip.load(model_name, device=device)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)

    mg = MushroomPromptGenerator()
    # If include_common_names is True and a prompts JSON path is provided (or default exists),
    # load prompts directly from that file instead of constructing templates.
    common_prompts = None
    if include_common_names:
        import os, json
        path = common_prompts_path or os.path.join(os.path.dirname(__file__), 'enhanced_with_common_names_prompts.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                common_prompts = json.load(f)

    results: Dict[str, Any] = {}

    for label in labels:
        prompts: List[str] = []

        # If we loaded a common_prompts bundle, prefer that (it contains image-based + common names)
        if common_prompts and label in common_prompts:
            prompts = list(common_prompts[label])
        else:
            if prompt_set == "v1":
                prompts = [t.format(label=label) for t in SHORT_PROMPTS]
            elif prompt_set == "names":
                names = _build_name_list(label, mg)
                for name in names:
                    for tmpl in NAME_PROMPT_TEMPLATES:
                        prompts.append(tmpl.format(name=name))
            elif prompt_set == "ensemble":
                # combine both
                prompts = [t.format(label=label) for t in SHORT_PROMPTS]
                names = _build_name_list(label, mg)
                for name in names:
                    for tmpl in NAME_PROMPT_TEMPLATES:
                        p = tmpl.format(name=name)
                        if p not in prompts:
                            prompts.append(p)
            else:
                raise ValueError(f"Unknown prompt_set: {prompt_set}")

        # Tokenize and encode in batches to avoid memory spikes
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                # text_tokens = clip.tokenize(batch_prompts).to(device)
                tokenizer = open_clip.get_tokenizer(model_name)
                text_tokens = tokenizer(batch_prompts).to(device)
                emb = model.encode_text(text_tokens)
                emb = F.normalize(emb, dim=-1)
                all_embeddings.append(emb.cpu())
        if all_embeddings:
            prompt_embeddings = torch.cat(all_embeddings, dim=0)
            # mean embedding per label
            label_embedding = F.normalize(prompt_embeddings.mean(dim=0), dim=-1)
        else:
            prompt_embeddings = torch.empty((0, model.text_projection.shape[1]))
            label_embedding = torch.zeros((model.text_projection.shape[1],))

        results[label] = {
            'prompts': prompts,
            'prompt_embeddings': prompt_embeddings,  # CPU tensor
            'label_embedding': label_embedding.cpu()
        }

    return results
