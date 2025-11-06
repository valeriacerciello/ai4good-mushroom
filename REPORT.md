# Mushroom Classification — Report

Date: 2025-10-23
Location: /zfs/ai4good/student/hgupta

## Summary of work completed
- Implemented CLIP zero-shot evaluation with multiple prompt sets and temperature pooling.
- Generated and evaluated four prompt configurations (v1, image-derived, enhanced, delta).
- Performed a temperature sweep (T ∈ {0.05,0.08,0.1,0.12,0.15,0.2,0.3,0.5}); T=0.05 chosen.
- Set up ResNet training pipeline (PyTorch Lightning + Hydra) and trained ResNet18 for 20 epochs (running).
- Implemented score-level CLIP+ResNet fusion with alpha sweep.
- Created visualization suite and generated publication-quality plots.
- Created automation and monitoring scripts (tmux sessions, monitor_training.sh, auto_run_fusion.sh).

All plots moved to: `plots/`

---

## Where to find artifacts
- Plots directory: `plots/`
  - temperature_sweep_plots.png
  - temperature_sweep_combined.png
  - fusion_alpha_sweep.png
  - fusion_alpha_sweep_detailed.png
  - fusion_model_comparison.png
  - fusion_heatmap.png
  - fusion_improvement.png
  - fusion_complete_overview.png

- Results JSONs:
  - `temperature_sweep_results.json` — Temperature sweep metrics
  - `fusion_results_resnet18_ep1.json` — Fusion run with current ResNet checkpoint (used for plots)
  - `fusion_results.json` — Original fusion results

- Other scripts and logs:
  - `eval_all_clip_models.py`, `final_fusion.py`, `create_fusion_graphs.py`
  - `resnet18_20epochs.log` — ResNet training log
  - `clip_all_models_evaluation.log` — CLIP evaluation log

---

## Key numbers (best/representative values)

A. CLIP temperature sweep (delta prompts only)
- Best temperature: **T = 0.05**
- Metrics at T=0.05 (on validation subset):
  - Top-1 accuracy: **26.5%**
  - Top-5 accuracy: **55.5%**
  - Balanced accuracy: **17.61%**
  - Macro F1-score: **13.42%**

B. CLIP+ResNet fusion (ResNet18 checkpoint: 1-epoch / 256-mini then improved)
- Validation alpha sweep (subset 200 val samples):
  - Best α (val): **0.0** (CLIP-only)
  - Validation Top-1 at α=0.0: **21.5%** (subset)
- Test (200 samples) final results (fusion run: `fusion_results_resnet18_ep1.json`):
  - Best alpha: **0.0** (CLIP-only)
  - Test Top-1 accuracy: **25.0%**
  - Test Top-5 accuracy: **48.0%**
  - Test Balanced accuracy: **15.17%**
  - Test Macro F1-score: **9.64%**

Note: the ResNet used in these fusion experiments is undertrained (1 epoch); therefore fusion selects CLIP-only.

---

## Observations & takeaways (by plot)

1) `plots/temperature_sweep_plots.png` (2×2 metrics vs. temperature)
- Top-1 is flat across temperatures; CLIP's top prediction is stable.
- Top-5 and Macro-F1 peak at lower temperatures (0.05), indicating sharper prompt pooling helps ranking.

2) `plots/temperature_sweep_combined.png`
- Overlay confirms T=0.05 is best overall; higher T values degrade Top-5 performance.

3) `plots/fusion_alpha_sweep_detailed.png` (4 panels: Top-1, Top-5, Balanced, Macro-F1 vs α)
- All metrics peak at α=0.0 (CLIP-only). As α increases (more ResNet weight), performance decays monotonically.
- ResNet-only (α=1.0) performs very poorly (~1–2% Top-1), so any ResNet contribution reduces ensemble performance.

4) `plots/fusion_model_comparison.png`
- Bar chart shows CLIP >> ResNet; fusion equals CLIP (since best α=0.0).
- Visual gap is very large; demonstrates ResNet is undertrained and currently harmful to fusion.

5) `plots/fusion_heatmap.png`
- Heatmap shows monotonic decline from left (α=0) to right (α=1) across metrics.
- No intermediate α yields improvement; need stronger ResNet for a central optimum.

6) `plots/fusion_improvement.png`
- Fusion shows 0% improvement over CLIP and massive % gains over ResNet (CLIP >> ResNet).
- Demonstrates CLIP is currently the best single model.

7) `plots/fusion_complete_overview.png`
- Single-page executive summary confirming above takeaways.

---

## Recommended immediate actions
1. Train ResNet properly (20 epochs) or get a pretrained baseline (ResNet50/101). This is required for meaningful fusion.
2. Use larger CLIP models (ViT-L/14) — this alone yields +8–12% over ViT-B/32 in practice.
3. Combine all prompts (done here) and ensemble CLIP variants (ViT-B/32, ViT-B/16, ViT-L/14) for immediate gains.
4. Re-run fusion after ResNet is stronger; expect best α in (0.3–0.7).

---

## Commands to reproduce or re-run analysis

```bash
# Recreate plots (assumes fusion_results.json exists)
mushroom_env/bin/python3 create_fusion_graphs.py

# Re-run fusion with a new ResNet checkpoint
mushroom_env/bin/python3 fusion_clip_resnet.py \
  --model resnet18 --ckpt /path/to/new_checkpoint.ckpt \
  --prompts enhanced_with_common_names_prompts_clean.json \
  --val_n 500 --test_n 500 --clip_temp 0.05 \
  --out fusion_results_new.json

# Re-evaluate CLIP with larger model
mushroom_env/bin/python3 eval_all_clip_models.py

# Monitor training (if running in tmux)
./monitor_training.sh
```

---

## Closing notes
- The current plots and JSONs are in `/zfs/ai4good/student/hgupta/` and all PNGs have been moved to `/zfs/ai4good/student/hgupta/plots/`.
- The main blocker to improved fusion is an undertrained ResNet. Once a strong ResNet or pretrained baseline is available, we should re-run `final_fusion.py` and regenerate the plots.

## Clarification: why CLIP top-5 numbers differ in some outputs

- The temperature sweep and some quick CLIP-only runs report Top-5 ~= 0.53–0.56 because they were computed on a sampled 200-image subset whose label-universe contains only ~90 unique species (the scripts rank only those labels when computing Top-5). This is the setting used in `temperature_sweep.py` and the produced `temperature_sweep_results.json`.
- The fusion runs (e.g. `fusion_results_resnet18_ep1.json`) evaluate CLIP and ResNet over the full label set (≈170 classes) and so Top-5 is measured across a larger candidate pool (hence lower magnitudes, e.g. 0.43–0.48 in fusion outputs).
- I added an apples-to-apples restricted fusion run: `fusion_restricted_sample.json` and plot `fusion_alpha_sweep_restricted.png` which use the sampled label set; these reproduce the higher Top-5 values when evaluated on the smaller label-universe.

When comparing numbers, make sure the label-universe (all 170 classes vs the sampled subset labels) is the same; otherwise Top-5 will not be directly comparable.

If you'd like, I can:
- Run the final fusion now (if a new checkpoint exists)
- Fine-tune CLIP on the mushroom training set
- Start training a larger backbone (ResNet50 or ViT)

