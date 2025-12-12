### `data/` – BLIP-derived Attributes + Prompts

This directory contains all intermediate text-side representations used for prompt engineering.

Files generated:
- `blip_captions.json` – raw BLIP captions + lighting estimation
- `blip_attributes.json` – per-image attribute dictionary
- `class_prompts_plain.json`
- `class_prompts_attr.json`
- `class_prompts_enriched.json`
- `class_prompts_attr_clean.json`
- `class_prompts_enriched_clean.json`

Generated via:

```console
python ../scripts/blip_caption_bank.py
python ../scripts/extract_attributes.py
python ../scripts/build_text_prompts.py
```