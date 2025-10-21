#!/usr/bin/env python3
# BLIP caption bank with lighting phrase + multi-prefix prompting
import os, json, argparse, random
from typing import List, Dict
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

DEFAULT_PREFIXES = [
    "a detailed photo of",
    "close up macro shot of",
    "forest mushroom with",
    "this mushroom has",
    "scientific description of",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

def is_image_file(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)

def brightness_words(pil_image: Image.Image) -> str:
    """Return a short lighting phrase from average luma."""
    arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    Y = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).mean()
    if Y < 0.30:
        return "in low light"
    if Y < 0.55:
        return "in moderate light"
    return "in bright daylight"

def dedup_keep_order(texts: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in texts:
        k = t.strip().lower()
        if k not in seen and len(k) > 0:
            seen.add(k)
            out.append(t.strip())
    return out

def generate_k_captions(image: Image.Image,
                        processor: BlipProcessor,
                        model: BlipForConditionalGeneration,
                        device: torch.device,
                        prefixes: List[str],
                        k: int,
                        beam: int,
                        max_tokens: int,
                        lighting_phrase: str) -> List[str]:
    caps = []
    for i in range(k):
        pref = prefixes[i % len(prefixes)].strip()
        inputs = processor(images=image, text=pref, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=beam,
                early_stopping=True,
            )
        dec = processor.decode(out[0], skip_special_tokens=True).strip()

        # ensure the prefix context is preserved (helps CLIP later)
        if pref.lower() in dec.lower():
            caption = dec
        else:
            caption = f"{pref} {dec}".strip()

        # make very short captions a bit more descriptive
        if len(caption.split()) < 5:
            caption = f"{caption}, showing detailed mushroom features"

        # append lighting phrase
        if lighting_phrase and lighting_phrase.lower() not in caption.lower():
            caption = f"{caption} {lighting_phrase}"

        caps.append(caption)
    return dedup_keep_order(caps)

def collect_image_paths(root: str) -> List[str]:
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if is_image_file(fn):
                paths.append(os.path.join(dp, fn))
    return paths

def main():
    ap = argparse.ArgumentParser(description="Build BLIP caption bank with lighting phrases.")
    ap.add_argument("--root", required=True, help="Dataset root (e.g., data/raw)")
    ap.add_argument("--out", default="data/prompts/blip_captions.json", help="Output JSON path")
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-base", help="BLIP captioning model name")
    ap.add_argument("--limit", type=int, default=0, help="Max number of images to process (0 = all)")
    ap.add_argument("--k", type=int, default=3, help="Captions per image")
    ap.add_argument("--beam", type=int, default=5, help="Beam width for generation")
    ap.add_argument("--max_tokens", type=int, default=75, help="Max new tokens per caption")
    ap.add_argument("--prefixes", type=str, default=",".join(DEFAULT_PREFIXES),
                    help="Comma-separated prompt prefixes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    print(f"[info] loading BLIP model: {args.model}")
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    print(f"[info] prefixes ({len(prefixes)}): {prefixes}")

    all_imgs = collect_image_paths(args.root)
    if args.limit and args.limit > 0:
        all_imgs = all_imgs[: args.limit]
    total = len(all_imgs)
    print(f"[info] images to process: {total}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_dict: Dict[str, Dict] = {}

    errors = 0
    for img_path in tqdm(all_imgs, desc="BLIPing", unit="img"):
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                lighting = brightness_words(im)
                caps = generate_k_captions(
                    image=im,
                    processor=processor,
                    model=model,
                    device=device,
                    prefixes=prefixes,
                    k=args.k,
                    beam=args.beam,
                    max_tokens=args.max_tokens,
                    lighting_phrase=lighting,
                )
            rel = os.path.relpath(img_path, args.root)
            out_dict[rel] = {"captions": caps, "lighting": lighting}
        except Exception as e:
            errors += 1
            print(f"[warn] failed on {img_path}: {e}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)

    print("\n[done]")
    print(f"  processed: {len(out_dict)} images")
    print(f"  errors   : {errors}")
    print(f"  saved to : {args.out}")

    # quick preview
    for i, (k, v) in enumerate(list(out_dict.items())[:3]):
        print(f"  {i+1}. {k}")
        print("     →", v["captions"][0] if v["captions"] else "<no caption>")

if __name__ == "__main__":
    main()
