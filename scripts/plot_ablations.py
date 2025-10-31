#!/usr/bin/env python3
import argparse, json, os
import matplotlib.pyplot as plt
import numpy as np

def tops(path): 
    d=json.load(open(path)); 
    return {s:m["top1"] for s,m in d["splits"].items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", default="results/metrics/clip_plain.json")
    ap.add_argument("--attr", default="results/metrics/clip_attr.json")
    ap.add_argument("--enriched", default="results/metrics/clip_enriched.json")
    ap.add_argument("--out", default="results/figs/ablations.png")
    ap.add_argument("--splits", nargs="+", default=["val","test"])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    P, A, E = tops(args.plain), tops(args.attr), tops(args.enriched)
    x = np.arange(len(args.splits)); w = 0.25

    plt.figure(figsize=(6,4))
    plt.bar(x - w, [P.get(s, np.nan) for s in args.splits], w, label="PLAIN")
    plt.bar(x,     [A.get(s, np.nan) for s in args.splits], w, label="+ATTR")
    plt.bar(x + w, [E.get(s, np.nan) for s in args.splits], w, label="+ATTR+LIGHT")
    plt.xticks(x, args.splits)
    plt.ylabel("Top-1 accuracy")
    plt.title("Ablations: attributes & lighting")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Saved →", args.out)

if __name__ == "__main__":
    main()
