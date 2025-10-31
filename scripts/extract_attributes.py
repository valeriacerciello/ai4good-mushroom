#!/usr/bin/env python3
# Parse simple mushroom attributes from BLIP captions using tiny lexicons.
import argparse, json, re
from collections import Counter
from typing import Dict, List

def norm(t: str) -> str:
    t = t.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

COLOR_TERMS = [
    ("golden brown","golden-brown"),("dark brown","dark-brown"),("light brown","light-brown"),
    ("brown","brown"),("white","white"),("cream","cream"),("beige","beige"),("gray","gray"),
    ("black","black"),("yellow","yellow"),("orange","orange"),("red","red"),("pink","pink"),
    ("purple","purple"),("olive","olive"),("tan","tan")
]
SURFACE_TERMS = ["smooth","scaly","sticky","slimy","dry","shiny","velvety","hairy","fibrous","wrinkled"]
CAP_SHAPE_TERMS = [("cup","cup-shaped"),("funnel","funnel-shaped"),("bell","bell-shaped"),
                   ("flat","flat"),("convex","convex"),("concave","concave"),("umbonate","umbonate")]
HABITAT_TERMS = [("forest","forest"),("woodland","forest"),("moss","moss"),
                 ("leaf litter","leaf litter"),("leaves","leaf litter"),
                 ("grass","grass"),("soil","soil"),("ground","soil"),
                 ("log","dead wood"),("stump","dead wood"),("tree trunk","dead wood"),
                 ("bark","dead wood"),("decaying wood","dead wood"),("on a tree","dead wood")]
HYMENOPHORE_TERMS = [("gills","gills"),("pores","pores"),("tubes","pores"),("spines","spines")]
ADJECTIVE_TERMS = ["pale","crowded","distant","free","attached","adnate","adnexed","decurrent","sinuate"]

def find_terms(text: str, terms) -> List[str]:
    found = []
    for item in terms:
        pat, alias = (item if isinstance(item, tuple) else (item, item))
        if " " in pat:
            if pat in text: found.append(alias)
        else:
            if re.search(rf"\b{re.escape(pat)}\b", text): found.append(alias)
    return found

def aggregate_attrs(captions: List[str]) -> Dict[str,str]:
    votes = {k: Counter() for k in ["cap_color","surface","cap_shape","habitat","hymenophore","hymen_adj"]}
    for c in captions:
        t = norm(c)
        for x in find_terms(t, COLOR_TERMS):       votes["cap_color"][x] += 1
        for x in find_terms(t, SURFACE_TERMS):     votes["surface"][x] += 1
        for x in find_terms(t, CAP_SHAPE_TERMS):   votes["cap_shape"][x] += 1
        for x in find_terms(t, HABITAT_TERMS):     votes["habitat"][x] += 1
        for x in find_terms(t, HYMENOPHORE_TERMS): votes["hymenophore"][x] += 1
        for x in find_terms(t, ADJECTIVE_TERMS):   votes["hymen_adj"][x] += 1
    out = {}
    for k, ctr in votes.items():
        if ctr:
            out[k] = sorted(ctr.items(), key=lambda kv:(-kv[1], kv[0]))[0][0]
    return out

def main():
    ap = argparse.ArgumentParser(description="Extract attributes from BLIP captions.")
    ap.add_argument("--captions", default="data/prompts/blip_captions.json")
    ap.add_argument("--out", default="data/prompts/blip_attributes.json")
    args = ap.parse_args()

    data = json.load(open(args.captions, "r", encoding="utf-8"))
    result = {}
    cover = Counter()
    for rel, meta in data.items():
        caps = meta.get("captions", [])
        attrs = aggregate_attrs(caps)
        result[rel] = {"attributes": attrs, "lighting": meta.get("lighting","")}
        for k in ["cap_color","surface","cap_shape","habitat","hymenophore","hymen_adj"]:
            if k in attrs: cover[k] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    n = max(1, len(data))
    print("[done] saved:", args.out)
    print(f"[stats] images: {len(data)}")
    for k in ["cap_color","surface","cap_shape","habitat","hymenophore","hymen_adj"]:
        print(f"[stats] {k} coverage: {cover[k]}/{len(data)} = {cover[k]*100.0/n:.1f}%")
    any_attr = sum(1 for v in result.values() if v["attributes"])
    print(f"[stats] any attribute coverage: {any_attr}/{len(data)} = {any_attr*100.0/n:.1f}%")

if __name__ == "__main__":
    main()
