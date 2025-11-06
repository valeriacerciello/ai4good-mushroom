#!/usr/bin/env python3
"""
Create labels.tsv from enhanced_with_common_names_prompts_clean.json
Writes ai4good-mushroom/labels.tsv with format: id\tclassname
"""
import json
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "enhanced_with_common_names_prompts_clean.json"
OUT = Path(__file__).resolve().parents[1] / "ai4good-mushroom" / "labels.tsv"

if not PROMPTS.exists():
    raise SystemExit(f"Prompts file not found: {PROMPTS}")

with PROMPTS.open('r', encoding='utf-8') as f:
    data = json.load(f)

classes = list(data.keys())
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', encoding='utf-8') as f:
    for i, cls in enumerate(classes):
        f.write(f"{i}\t{cls}\n")

print(f"Wrote {len(classes)} labels to {OUT}")
