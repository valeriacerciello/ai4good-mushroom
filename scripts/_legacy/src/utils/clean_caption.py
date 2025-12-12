# src/utils/clean_caption.py
from __future__ import annotations
import json, re
from dataclasses import dataclass
from typing import List, Optional, Dict, Iterable

# vocab 
FUNGAL_PARTS = {
    "mushroom","mushrooms","fungus","fungi","cap","gills","stem","stalk","ring",
    "veil","volva","pores","pore","spore","spores","cup","cluster","shelf",
    "pile","bunch","group"
}


COLORS = {
    "white","yellow","brown","orange","red","pink","beige","gray","black","golden","cream"
}

TEXTURES = {
    "smooth","scaly","cracked","slimy","shiny","rough","dry","wet","fibrous"
}

LIGHTING = {
    "bright","daylight","light","sunlight","moderate","low","dim","shade","shadow","indoor","outdoor"
}

EXTRA_DESCRIPTORS = {"macro","close","detailed","group","single","small","large"}

NEGATIVE = {
    "forest","floor","ground","wood","tree","bark","leaves","grass","table","moss",
    "snail","moth","cloves","background","piece","nails","shell","dead"
}

STOPWORDS = {
    "a","an","the","this","that","these","those","and","or","of","with","on","in",
    "at","by","for","to","from","is","are","it","its","as","very","really",
    "close","up","photo","macro","image","picture","shot"
}


WHITELIST = FUNGAL_PARTS | COLORS | TEXTURES | LIGHTING | EXTRA_DESCRIPTORS


_TOKEN_RE = re.compile(r"[a-z]+")

@dataclass
class Cleaned:
    kept_tokens: List[str]           # tokens after filtering
    attrs: List[str]                 # tokens to use as attributes (no lighting)
    lighting: Optional[str] = None   # one lighting token if present
    dropped_tokens: List[str] = None # optional debugging

def _normalize(s: str) -> str:
    s = s.lower().replace("-", " ")
    s = s.replace("mushrooms", "mushroom")
    s = s.replace("fungi", "fungus")

    return s

def _tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall(_normalize(s))

def _unique_in_order(xs: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def clean(text: str) -> Cleaned:
    """
    Keep only fungi-related / color / shape / texture / lighting words.
    Return a small structured object for downstream prompt building.
    """
    toks = _tokenize(text)
    dropped, kept = [], []

    for t in toks:
        if t in STOPWORDS or t in NEGATIVE:
            dropped.append(t); continue
        if t in WHITELIST:
            kept.append(t)
        # else drop silently (generic scenery words etc.)

    kept = _unique_in_order(kept)

    light = None
    attrs = []
    for t in kept:
        if t in LIGHTING and light is None:
            light = t
        else:
            attrs.append(t)

    return Cleaned(kept_tokens=kept, attrs=attrs, lighting=light, dropped_tokens=dropped)

# CLI helper 
def preview_file(in_json: str, out_csv: str, limit: int = 200):
    """
    Read a JSON dict {image_id: {captions: [...], lighting: str}}, 
    write a CSV with raw/cleaned for quick inspection.
    """
    import csv
    data: Dict[str, dict] = json.load(open(in_json))
    rows = []
    for i, (k, v) in enumerate(data.items()):
        # Some entries might have dicts with "captions"
        if isinstance(v, dict) and "captions" in v:
            text = " ".join(v["captions"])
        elif isinstance(v, str):
            text = v
        else:
            text = str(v)

        c = clean(text)
        rows.append({
            "image_id": k,
            "raw_caption": text,
            "kept_tokens": " ".join(c.kept_tokens),
            "attributes": " ".join(c.attrs),
            "lighting": c.lighting or "",
            "dropped_tokens": " ".join(c.dropped_tokens or []),
        })
        if i + 1 >= limit:
            break

    import os
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows")


if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--captions", default="data/prompts/blip_captions.json")
    p.add_argument("--out", default="results/cleaning_comparison.csv")
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()
    os.makedirs("results", exist_ok=True)
    preview_file(args.captions, args.out, args.limit)
