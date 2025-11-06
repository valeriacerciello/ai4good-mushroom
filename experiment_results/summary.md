# Experiment results (consolidated)

This folder contains the consolidated experiment outputs for CLIP + ResNet fusion runs.

Files included
- `prototype_knn_results.json` — CLIP prototype (centroid) baseline (zero-shot on train centroids). Key metrics below.
- `fusion_results_delta_full.json` — ResNet + CLIP (prompt) alpha-fusion baseline and LR-stacker run (stacker failed; use alpha baseline).
- `calibration_tta_results.json` — Temperature scaling + alpha sweep calibration run.
- `pt_fs_test.json` — Small smoke-test run of the new few-shot CLIP + ResNet fusion (shots=2) saved from `/tmp`.

Key metrics (selected)

- Prototype KNN (zero-shot centroids)
  - val Top-1: 0.51, val Top-5: 0.76
  - test Top-1: 0.465, test Top-5: 0.783

- ResNet + CLIP (prompt) alpha-fusion (from `fusion_results_delta_full.json`)
  - Best alpha (val): 0.10
  - Test metrics @ alpha=0.10: Top-1 = 0.5942, Top-5 = 0.8685, Balanced Acc = 0.5563, Macro-F1 = 0.5266

- Calibration + TTA (temperature scaling)
  - T_clip = 1.0, T_res = 1.0 (fit on val)
  - Best alpha: 0.10
  - Test Top-5 ≈ 0.8683 (very similar to alpha baseline)

- Few-shot CLIP prototypes smoke test (tiny run: `shots=2`, `train_limit=200`, `val_n=8`, `test_n=8`)
  - Best alpha (val): 0.0
  - Test @ best alpha: Top-1 = 0.0, Top-5 = 0.125  (note: very small sample sizes; this is only a smoke check that the pipeline runs)

Notes
- I extended `run_prototype_knn.py` to support `--shots` (few-shot prototypes) and optional ResNet fusion via `--ckpt` and `--model`.
- The small smoke test confirms the code path; for meaningful few-shot evaluation use larger `train_limit` and `test_n` and provide a ResNet checkpoint via `--ckpt`.
- The best robust baseline so far remains the simple alpha-sweep fusion (alpha ~ 0.10) combining CLIP(prompt) + ResNet.

How to run the few-shot CLIP + ResNet fusion (example)

```
# using the project's venv python (recommended)
/zfs/ai4good/student/hgupta/mushroom_env/bin/python3 run_prototype_knn.py \
  --train_limit 5000 --val_n 200 --test_n 2000 --shots 5 \
  --ckpt ./logs/mushroom/train/runs/.../checkpoints/last.ckpt \
  --model resnet18 --out experiment_results/prototype_knn_fewshot_shots5.json --device cuda
```

Next steps
- Run a full few-shot sweep (shots ∈ {1,2,5,10}) with a proper ResNet checkpoint and larger validation/test sizes.
- Optionally, re-run the LR stacker training with a larger/stratified holdout (not the 200-example val) to avoid the poor stacker result seen earlier.
- Try prompt-tuning (CoOp) to improve CLIP embeddings before forming prototypes.

If you'd like, I can start the full few-shot sweep on the cluster (GPU recommended for CLIP/ResNet) and write the results into this folder.
# Experiment summary — CLIP + ResNet fusion (delta_prompts.json)

Date: 2025-10-24
Workspace: /zfs/ai4good/student/hgupta

## Experiments included
All experiments used `delta_prompts.json` (prompt pool) and CLIP ViT-B/32 unless noted.
Results and logs are copied into this folder.

Files in this folder:
- `prototype_knn_results.json`, `prototype_knn.log`
- `fusion_results_delta_full.json`, `fusion_stack.log` (stacking + alpha baseline)
- `calibration_tta_results.json`, `calibration_tta.log`

---

## 1) Prototype KNN (CLIP image-prototype centroid classifier)
- Setup: build per-class centroids from train (limit=5000), classify val (200) and test (2000) by cosine similarity (softmax temp=0.01).
- Val metrics:
  - Top‑1 = 0.51
  - Top‑5 = 0.76
  - Balanced Acc = 0.4089
  - Macro F1 = 0.3092
- Test metrics:
  - Top‑1 = 0.465
  - Top‑5 = 0.783
  - Balanced Acc = 0.4401
  - Macro F1 = 0.3947
- Notes: quick, low-effort baseline that performed reasonably well.

---

## 2) Learned stacking (Logistic Regression) + alpha baseline
- Setup: compute CLIP_probs and ResNet_probs (ResNet ckpt: `./logs/mushroom/train/runs/2025-10-23_13-55-22/checkpoints/last.ckpt`), train LR on val predictions, evaluate on full test (15614).
- Stacked LR test metrics (training issue suspected):
  - Top‑1 = 0.002
  - Top‑5 = 0.013
  - Balanced Acc = 0.006
  - Macro F1 = 0.00007
- Alpha baseline (sweep on val → applied to test):
  - Chosen α = 0.10 (by validation Top‑5)
  - Test metrics at α=0.10:
    - Top‑1 = 0.5942
    - Top‑5 = 0.8685
    - Balanced Acc = 0.5563
    - Macro F1 = 0.5266
- Notes: the simple α sweep produced strong results; the naive LR stacker failed due to class/sample imbalance or label alignment issues (see warnings in logs). Recommend fixing stacker training split or using a larger training set for the stacker.

---

## 3) Calibration + simple TTA (temperature scaling on val)
- Setup: fit scalar temperature for CLIP and ResNet on val (200) then sweep α on scaled probs; evaluated on full test (15614).
- Fitted temperatures: T_clip = 1.0, T_res = 1.0 (no change in this run)
- Chosen α = 0.10 (by validation Top‑5)
- Test metrics at chosen α:
  - Top‑1 = 0.5941
  - Top‑5 = 0.8683
  - Balanced Acc = 0.5563
  - Macro F1 = 0.5266
- Notes: calibration did not change temperatures here (T=1.0); alpha sweep matched the fusion baseline.

---

## Quick comparison (test Top‑5)
- Prototype KNN (subset test 2k): 0.783
- Alpha-fusion baseline (full test): 0.8685
- Stacked LR: 0.013 (problem - discard until fixed)
- Calibration run: 0.8683

## Recommended next actions
1. Use the alpha-fusion baseline (α=0.10) as the current best result.
2. Fix stacking training: create a larger/stratified training split for the stacker (e.g., hold out 1k from train) and re-train; apply class weighting or regularization.
3. Run CoOp prompt tuning on GPU to further improve CLIP and re-run fusion.


Saved artifacts (in this folder):

Generated by automation on 2025-10-24.

---

## 4) Few-shot CLIP prototypes sweep (new)

I ran a few-shot sweep using `run_prototype_knn.py` with `shots` = 1,2,5,10, `train_limit`=5000, `val_n`=200 and `test_n`=15614, fusing with the ResNet checkpoint `logs/mushroom/train/runs/2025-10-23_13-55-22/checkpoints/last.ckpt`.

Summary (best alpha chosen on val, test metrics reported):

- shots = 1
  - best alpha = 0.6
  - test: Top‑1 = 0.8535, Top‑5 = 0.9302, Balanced Acc = 0.8072, Macro‑F1 = 0.8111

- shots = 2
  - best alpha = 0.6
  - test: Top‑1 = 0.8552, Top‑5 = 0.9367, Balanced Acc = 0.8076, Macro‑F1 = 0.8125

- shots = 5
  - best alpha = 0.5
  - test: Top‑1 = 0.8563, Top‑5 = 0.9478, Balanced Acc = 0.8099, Macro‑F1 = 0.8144

- shots = 10
  - best alpha = 0.5
  - test: Top‑1 = 0.8560, Top‑5 = 0.9518, Balanced Acc = 0.8097, Macro‑F1 = 0.8141

Notes:
- These few-shot runs substantially improve performance when fusing CLIP prototypes with the ResNet checkpoint. The best-performing shot counts here are 5–10 with alpha around 0.5–0.6.
- All result JSONs for these runs are in this folder as `prototype_knn_fewshot_shots{1,2,5,10}.json`.

If you want, I can now:
- Run a denser sweep over shots and alphas, or
- Re-run these few-shot experiments but compute CLIP prototypes from prompt-tuned embeddings (CoOp) or with randomized per-class sampling (instead of deterministic first-N), or
- Produce plots comparing Top‑1/Top‑5 vs shots and alpha.
