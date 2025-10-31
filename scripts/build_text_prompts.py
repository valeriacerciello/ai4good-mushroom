#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse, csv, json, re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

from src.utils.clean_caption import clean


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def read_labels_tsv(tsv_path: Optional[str]) -> Dict[str, str]:
    """Optional: map numeric label -> class name using labels.tsv (tab- or comma-separated)."""
    if not tsv_path:
        return {}
    mapping = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        sep = "\t" if ("\t" in first) else ","
        parts = first.split(sep)
        # header?
        if parts and not parts[0].isdigit():
            for line in f:
                if not line.strip():
                    continue
                p = line.strip().split(sep)
                if len(p) >= 2:
                    mapping[p[0]] = p[1]
        else:
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
            for line in f:
                if not line.strip():
                    continue
                p = line.strip().split(sep)
                if len(p) >= 2:
                    mapping[p[0]] = p[1]
    return mapping


def titleize(name: str) -> str:
    # Keep scientific casing intact if it looks like "Genus species"
    if " " in name and name[0].isupper():
        return name
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip()


def dedup_keep_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for s in items:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


# -------------------------------------------------------------------------
# Attribute-based prompt generation
# -------------------------------------------------------------------------
def aggregate_class_attributes(class_to_images: Dict[str, List[str]],
                               attr_db: Dict[str, dict],
                               top_k_per_field: int = 2) -> Dict[str, Dict[str, List[str]]]:
    """
    For each class, tally attribute frequencies over its images and keep top-k per field.
    Returns: {class: {field: [values...]}}
    """
    per_class: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for cls, imgs in class_to_images.items():
        for rel in imgs:
            meta = attr_db.get(rel)
            if not meta:
                continue
            attrs = meta.get("attributes", {})
            for field, value in attrs.items():
                if value:
                    per_class[cls][field][value] += 1

    topk: Dict[str, Dict[str, List[str]]] = {}
    for cls, fields in per_class.items():
        topk[cls] = {}
        for field, ctr in fields.items():
            if not ctr:
                continue
            ranked = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k_per_field]
            topk[cls][field] = [v for v, _ in ranked]
    return topk


PLAIN_TEMPLATES = [
    "a close-up photo of {CLASS} mushroom",
    "a scientific field photo of {CLASS}",
    "a macro photograph of {CLASS} fungus",
    "a natural habitat photo of {CLASS} mushroom",
]

LIGHTING_VARIANTS = ["in bright daylight", "in moderate light", "in low light"]


def render_attr_phrase(attrs: Dict[str, List[str]]) -> str:
    parts = []
    if "cap_color" in attrs:
        parts.append(f"cap {attrs['cap_color'][0]}")
    if "surface" in attrs:
        parts.append(f"{attrs['surface'][0]} surface")
    if "hymenophore" in attrs:
        parts.append(f"{attrs['hymenophore'][0]}")
    if "hymen_adj" in attrs:
        parts.append(f"{attrs['hymen_adj'][0]}")
    if "cap_shape" in attrs:
        parts.append(f"{attrs['cap_shape'][0]}")
    if "habitat" in attrs:
        parts.append(f"{attrs['habitat'][0]}")
    return ", ".join(parts)


def build_prompts_for_class(cls: str,
                            top_attrs: Dict[str, List[str]],
                            min_plain: int = 4) -> Tuple[List[str], List[str]]:
    """Return (plain_prompts, enriched_prompts) lists for a class."""
    CLASS = titleize(cls)
    plain = [t.format(CLASS=CLASS) for t in PLAIN_TEMPLATES[:min_plain]]

    enriched = []
    attr_phrase = render_attr_phrase(top_attrs)
    if attr_phrase:
        base_templates = [
            "a close-up photo of {CLASS} mushroom, {ATTRS}, {LIGHT}",
            "a scientific field photo of {CLASS}, {ATTRS}, {LIGHT}",
            "a macro photograph of {CLASS} fungus, {ATTRS}, {LIGHT}",
        ]
        for L in LIGHTING_VARIANTS:
            for bt in base_templates:
                enriched.append(bt.format(CLASS=CLASS, ATTRS=attr_phrase, LIGHT=L))

    if not enriched:
        for L in LIGHTING_VARIANTS:
            enriched.append(f"a close-up photo of {CLASS} mushroom, {L}")
            enriched.append(f"a natural habitat photo of {CLASS} fungus, {L}")

    return dedup_keep_order(plain), dedup_keep_order(enriched)


def build_attr_only_prompts_for_class(cls: str,
                                      top_attrs: Dict[str, List[str]],
                                      min_prompts: int = 4) -> List[str]:
    """Prompts with attributes but WITHOUT lighting. If no attrs → fallback to plain."""
    CLASS = titleize(cls)
    attr_phrase = render_attr_phrase(top_attrs)
    if attr_phrase:
        base = [
            "a close-up photo of {CLASS} mushroom, {ATTRS}",
            "a scientific field photo of {CLASS}, {ATTRS}",
            "a macro photograph of {CLASS} fungus, {ATTRS}",
            "a natural habitat photo of {CLASS} mushroom, {ATTRS}",
        ]
        out = [t.format(CLASS=CLASS, ATTRS=attr_phrase) for t in base]
    else:
        out = [t.format(CLASS=CLASS) for t in PLAIN_TEMPLATES[:min_prompts]]
    return dedup_keep_order(out)[:12]


# -------------------------------------------------------------------------
# Cleaner-based BLIP prompt generation
# -------------------------------------------------------------------------
def build_clean_prompts_from_captions(blip_path: str, limit_per_image: int = 3):
    """
    Build prompts directly from BLIP captions using the cleaning function.
    Returns:
        attr_clean: {class: [prompts with attributes only]}
        enriched_clean: {class: [prompts with attributes + lighting]}
    """
    if not os.path.exists(blip_path):
        print(f"[warn] BLIP caption file not found: {blip_path}")
        return {}, {}

    # Canonicalize lighting tokens → phrases used in your templates
    LIGHT_MAP = {
        "low": "low light",
        "bright": "bright daylight",
        "daylight": "bright daylight",
        "moderate": "moderate light",
        "dim": "low light",
        "shadow": "shade",
        "shade": "shade",
        "indoor": "indoor light",
        "outdoor": "outdoor light",
        "light": None,  # too generic alone
    }
    GENERIC = {"mushroom", "fungus", "light", "detailed", "group", "close", "macro", "daylight"}
    LIGHT_KEYS = {"low", "bright", "daylight", "moderate", "dim", "shadow", "shade", "indoor", "outdoor", "light"}

    data = json.load(open(blip_path, "r", encoding="utf-8"))
    attr_clean, enriched_clean = defaultdict(list), defaultdict(list)

    for rel, meta in data.items():
        if not isinstance(meta, dict):
            continue
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        cls = parts[0].strip()
        captions = meta.get("captions", [])

        for c_text in captions[:limit_per_image]:
            c = clean(c_text)

            # Filter attributes once (remove generic + lighting tokens)
            c.attrs = [t for t in c.attrs if t not in GENERIC and t not in LIGHT_KEYS]

            # Normalize lighting
            c_lighting = None
            if c.lighting:
                c_lighting = LIGHT_MAP.get(c.lighting, c.lighting)

            # Skip if nothing useful
            if not c.attrs and not c_lighting:
                continue

            CLASS = cls.replace("_", " ").title()

            # Attribute-only
            if c.attrs:
                attr_phrase = ", ".join(c.attrs)
                attr_clean[cls].append(f"a close-up photo of {CLASS} mushroom, {attr_phrase}")

            # Attribute + lighting
            phrase = f"a close-up photo of {CLASS} mushroom"
            if c.attrs:
                phrase += ", " + ", ".join(c.attrs)
            if c_lighting:
                phrase += f", in {c_lighting}"
            enriched_clean[cls].append(phrase)

    # Dedup + cap
    for d in (attr_clean, enriched_clean):
        for cls, lst in d.items():
            d[cls] = dedup_keep_order(lst)[:12]
    return attr_clean, enriched_clean


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build per-class text prompts (plain, attr-only, enriched, cleaned).")
    ap.add_argument("--csv", default="splits/train.csv")
    ap.add_argument("--csv_image_col", default="filepath")
    ap.add_argument("--csv_label_col", default="label")
    ap.add_argument("--labels_tsv", default="")
    ap.add_argument("--attributes", default="data/prompts/blip_attributes.json")
    ap.add_argument("--out_plain", default="data/prompts/class_prompts_plain.json")
    ap.add_argument("--out_attr", default="data/prompts/class_prompts_attr.json")
    ap.add_argument("--out_enriched", default="data/prompts/class_prompts_enriched.json")
    ap.add_argument("--top_k_per_field", type=int, default=2)

    # Cleaned prompts
    ap.add_argument("--blip_captions", default="data/prompts/blip_captions.json")
    ap.add_argument("--out_attr_clean", default="data/prompts/class_prompts_attr_clean.json")
    ap.add_argument("--out_enriched_clean", default="data/prompts/class_prompts_enriched_clean.json")
    ap.add_argument("--captions_per_image", type=int, default=3)

    args = ap.parse_args()

    # --- attribute-based prompts ---
    attr_db = json.load(open(args.attributes, "r", encoding="utf-8"))
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    labmap = read_labels_tsv(args.labels_tsv) if args.labels_tsv else {}
    class_to_images: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        img = (r[args.csv_image_col] or "").strip()
        lbl = (r[args.csv_label_col] or "").strip()
        if not img or not lbl:
            continue
        lbl_name = labmap.get(lbl, lbl)
        class_to_images[lbl_name].append(img)

    top_attrs = aggregate_class_attributes(class_to_images, attr_db, top_k_per_field=args.top_k_per_field)

    out_plain, out_attr, out_enriched = {}, {}, {}
    for cls in sorted(class_to_images.keys()):
        p_plain, p_enriched = build_prompts_for_class(cls, top_attrs.get(cls, {}))
        if len(p_plain) < 4:
            p_plain = (p_plain + [t.format(CLASS=titleize(cls)) for t in PLAIN_TEMPLATES])[:4]
        out_plain[cls] = p_plain
        out_enriched[cls] = p_enriched[:12]
        out_attr[cls] = build_attr_only_prompts_for_class(cls, top_attrs.get(cls, {}))

    for path, obj in [
        (args.out_plain, out_plain),
        (args.out_attr, out_attr),
        (args.out_enriched, out_enriched),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)

    # --- cleaned prompts from BLIP captions ---
    print("[clean prompts] building from:", args.blip_captions)
    attr_clean, enriched_clean = build_clean_prompts_from_captions(
        args.blip_captions, limit_per_image=args.captions_per_image
    )

    for path, obj in [
        (args.out_attr_clean, attr_clean),
        (args.out_enriched_clean, enriched_clean),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    print("  wrote:", args.out_attr_clean, "and", args.out_enriched_clean)

    # --- summary ---
    print("[done]")
    print(" classes:", len(out_plain))
    if out_plain:
        some = next(iter(out_plain))
        print(" example class:", some)
        print("  plain:", out_plain[some][:2])
        print("  attr-only:", out_attr[some][:2])
        print("  enriched:", out_enriched[some][:2])
    if attr_clean:
        cls = next(iter(attr_clean))
        print("  clean_attr:", attr_clean[cls][:2])
    if enriched_clean:
        cls = next(iter(enriched_clean))
        print("  clean_enriched:", enriched_clean[cls][:2])


if __name__ == "__main__":
    main()
