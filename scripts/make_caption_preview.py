#!/usr/bin/env python3
import os, sys, json, csv, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import Counter, defaultdict
from src.utils.clean_caption import clean  # uses your cleaner

def main():
    ap = argparse.ArgumentParser(
        description="Preview BLIP captions vs cleaned tokens/lighting."
    )
    ap.add_argument("--captions", default="data/prompts/blip_captions.json",
                    help="Path to BLIP captions JSON")
    ap.add_argument("--out", default="results/examples/caption_preview_clean.csv",
                    help="Output CSV path")
    ap.add_argument("--limit", type=int, default=200,
                    help="Max rows to write (0 = all)")
    ap.add_argument("--per_image_captions", type=int, default=3,
                    help="How many BLIP captions to consider per image (1-3)")
    ap.add_argument("--class_contains", default="",
                    help="Optional: only include classes whose name contains this substring (case-insensitive)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.captions, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    class_filter = args.class_contains.lower().strip()

    for rel, meta in data.items():
        if not isinstance(meta, dict):
            continue

        # class name is the first folder in "Class/image.jpg"
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        cls = parts[0]
        if class_filter and class_filter not in cls.lower():
            continue

        caps = meta.get("captions", [])[: max(1, min(3, args.per_image_captions))]
        if not caps:
            continue

        # concatenate raw captions
        raw_concat = " ".join(caps)

        # aggregate cleaned tokens and pick dominant lighting across the considered captions
        kept_tokens = []
        lighting_counts = Counter()
        for ctext in caps:
            c = clean(ctext)
            kept_tokens.extend(c.attrs or [])
            if c.lighting:
                lighting_counts[c.lighting] += 1

        # deduplicate tokens, keep stable order by first occurrence
        seen = set()
        kept_tokens_dedup = []
        for t in kept_tokens:
            if t not in seen:
                seen.add(t)
                kept_tokens_dedup.append(t)

        lighting = lighting_counts.most_common(1)[0][0] if lighting_counts else ""

        rows.append({
            "image_id": rel,
            "class": cls,
            "raw_caption": raw_concat,
            "kept_tokens": ", ".join(kept_tokens_dedup),
            "lighting": lighting,
        })

        if args.limit and len(rows) >= args.limit:
            break

    # write CSV
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "class", "raw_caption", "kept_tokens", "lighting"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[preview] wrote {len(rows)} rows → {args.out}")

if __name__ == "__main__":
    main()
