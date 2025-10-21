#!/usr/bin/env python3
import argparse, csv, json, re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

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
            seen.add(k); out.append(s.strip())
    return out

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

# We’ll generate multiple lighting variants; don’t hard-wire a single one.
LIGHTING_VARIANTS = ["in bright daylight", "in moderate light", "in low light"]

def render_attr_phrase(attrs: Dict[str, List[str]]) -> str:
    parts = []
    # join fields only if we have values
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

    # Fallback: if we have no attrs for this class, still diversify with lighting
    if not enriched:
        for L in LIGHTING_VARIANTS:
            enriched.append(f"a close-up photo of {CLASS} mushroom, {L}")
            enriched.append(f"a natural habitat photo of {CLASS} fungus, {L}")

    return dedup_keep_order(plain), dedup_keep_order(enriched)

def main():
    ap = argparse.ArgumentParser(description="Build per-class text prompts (plain & enriched).")
    ap.add_argument("--csv", default="splits/train.csv", help="CSV with image paths and class labels")
    ap.add_argument("--csv_image_col", default="filepath")
    ap.add_argument("--csv_label_col", default="label")
    ap.add_argument("--labels_tsv", default="", help="Optional: numeric->name map; leave empty to skip")
    ap.add_argument("--attributes", default="data/prompts/blip_attributes.json")
    ap.add_argument("--out_plain", default="data/prompts/class_prompts_plain.json")
    ap.add_argument("--out_enriched", default="data/prompts/class_prompts_enriched.json")
    ap.add_argument("--top_k_per_field", type=int, default=2)
    args = ap.parse_args()

    # read attributes db
    attr_db = json.load(open(args.attributes, "r", encoding="utf-8"))

    # read CSV + validate header
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in [args.csv_image_col, args.csv_label_col] if c not in header]
        if missing:
            raise ValueError(f"CSV missing columns {missing}. Header={header}")
        rows = list(reader)

    # optional label mapping
    labmap = read_labels_tsv(args.labels_tsv) if args.labels_tsv else {}

    # build class -> list[relative_image_path]
    class_to_images: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        img = (r[args.csv_image_col] or "").strip()
        lbl = (r[args.csv_label_col] or "").strip()
        if not img or not lbl:
            continue
        # map numeric label to name if mapping provided
        lbl_name = labmap.get(lbl, lbl)
        class_to_images[lbl_name].append(img)

    # aggregate attributes per class
    top_attrs = aggregate_class_attributes(class_to_images, attr_db, top_k_per_field=args.top_k_per_field)

    # build prompts
    out_plain: Dict[str, List[str]] = {}
    out_enriched: Dict[str, List[str]] = {}
    for cls in sorted(class_to_images.keys()):
        p_plain, p_enriched = build_prompts_for_class(cls, top_attrs.get(cls, {}))
        # ensure at least 4 prompts/class in plain; enriched could be larger
        if len(p_plain) < 4:
            p_plain = (p_plain + [t.format(CLASS=titleize(cls)) for t in PLAIN_TEMPLATES])[:4]
        out_plain[cls] = p_plain
        out_enriched[cls] = p_enriched[:12]  # cap for efficiency

    # write
    for path, obj in [(args.out_plain, out_plain), (args.out_enriched, out_enriched)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    print("[done]")
    print(" classes:", len(out_plain))
    some = next(iter(out_plain)) if out_plain else None
    if some:
        print(" example class:", some)
        print("  plain:", out_plain[some][:2])
        print("  enriched:", out_enriched[some][:2])

if __name__ == "__main__":
    main()

