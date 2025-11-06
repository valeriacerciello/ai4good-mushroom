# Prompt Embeddings Utilities (quick guide)

This folder contains utilities to generate and compare CLIP text embeddings for mushroom species.

Files
- `clip_utils.py` : get_text_embeddings(labels, prompt_set='v1', ...) returns per-label prompts and embeddings.
- `run_quick_zero_shot.py` : quick comparison script that evaluates three strategies on a small subset and writes `quick_zero_shot_report.json`.
- `quick_zero_shot_report.json` : results from the latest quick run.

What `get_text_embeddings` does
- Builds prompt lists from templates:
  - `v1`: short photographic prompts (e.g., "a photo of {label}")
  - `names`: Latin + common names templates
  - `ensemble`: combined v1 + names
- Tokenizes and encodes prompts using CLIP, returns per-prompt embeddings and a mean label embedding.

Quick usage

1) Activate virtualenv (created in this project):

```bash
source mushroom_env/bin/activate
```

2) Compute embeddings for a few labels interactively:

```python
from clip_utils import get_text_embeddings
labels = ['Amanita muscaria', 'Boletus edulis']
res = get_text_embeddings(labels, prompt_set='v1')
# res[label]['prompts'] -> list of prompts
# res[label]['label_embedding'] -> mean embedding (CPU tensor)
```

3) Re-run the quick zero-shot comparison (this will use CLIP and may take a few minutes):

```bash
source mushroom_env/bin/activate
python run_quick_zero_shot.py
```

Latest quick report summary

```
Samples evaluated: 200
Original averaged prompts accuracy: 0.235
Short prompts (v1) accuracy: 0.22
Ensemble prompts accuracy: 0.195
```

Notes & next steps
- The short prompt set is intentionally minimal; expanding/curating templates is likely to improve results.
- `enhanced_mushroom_prompts.json` contains image-based prompts; try using it instead of `mushroom_prompts.json` for better image-grounded embeddings.
- Consider implementing per-prompt scoring (compute similarity to each prompt then aggregate) instead of averaging prompt embeddings.
- For full evaluation, run `clip_mushroom_classifier.py` with GPU and the desired prompts file.
