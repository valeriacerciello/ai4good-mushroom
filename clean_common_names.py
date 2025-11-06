#!/usr/bin/env python3
"""
Clean prompts in enhanced_with_common_names_prompts.json to remove noisy entries
and produce a cleaned prompts file plus a review CSV.

Outputs:
 - enhanced_with_common_names_prompts_clean.json
 - common_names_review.csv

Usage:
  python clean_common_names.py
"""
import json
import re
import os
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent
IN_PROMPTS = ROOT / 'enhanced_with_common_names_prompts.json'
OUT_PROMPTS = ROOT / 'enhanced_with_common_names_prompts_clean.json'
REVIEW_CSV = ROOT / 'common_names_review.csv'

# Patterns considered noisy
MEASUREMENT_RE = re.compile(r'\b\d+\s?(in|cm|mm|inch|inches|ft|m)\b', flags=re.IGNORECASE)
PAREN_MEASURE_RE = re.compile(r'\(.*?\d+.*?\)')
NON_LATIN_RE = re.compile(r'[\u4E00-\u9FFF\u0600-\u06FF\u0400-\u04FF]')
SHORT_TOKEN_RE = re.compile(r'^\W*$')

def is_noisy(s: str) -> bool:
    s_stripped = s.strip()
    if len(s_stripped) < 3:
        return True
    if len(s_stripped) > 120:
        return True
    if MEASUREMENT_RE.search(s_stripped):
        return True
    if PAREN_MEASURE_RE.search(s_stripped):
        return True
    if NON_LATIN_RE.search(s_stripped):
        return True
    if SHORT_TOKEN_RE.match(s_stripped):
        return True
    # numbers-only
    if re.fullmatch(r'[0-9\s\.,]+', s_stripped):
        return True
    return False

def clean_prompts(prompts):
    out = []
    seen = set()
    for p in prompts:
        p0 = p.strip()
        # normalize whitespace and remove weird non-breaking spaces
        p0 = p0.replace('\xa0', ' ')
        p0 = re.sub(r'\s+', ' ', p0)
        # strip leading/trailing punctuation
        p0 = p0.strip(' \t\n"\'')
        key = p0.lower()
        if key in seen:
            continue
        seen.add(key)
        if is_noisy(p0):
            continue
        out.append(p0)
    return out

def main():
    if not IN_PROMPTS.exists():
        print(f"Input prompts not found: {IN_PROMPTS}")
        return
    data = json.loads(IN_PROMPTS.read_text())
    clean_data = {}
    rows = []
    for species, prompts in data.items():
        cleaned = clean_prompts(prompts)
        clean_data[species] = cleaned
        removed = [p for p in prompts if p not in cleaned]
        rows.append([species, len(prompts), len(cleaned), ';'.join(cleaned[:5]), ';'.join(removed[:5])])

    json.dump(clean_data, OUT_PROMPTS.open('w'), indent=2, ensure_ascii=False)
    with REVIEW_CSV.open('w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['latin_name', 'n_original', 'n_cleaned', 'sample_cleaned', 'sample_removed'])
        for r in rows:
            writer.writerow(r)

    print(f"Wrote cleaned prompts: {OUT_PROMPTS}")
    print(f"Wrote review CSV: {REVIEW_CSV}")

if __name__ == '__main__':
    main()
