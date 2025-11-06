## Summary of recent actions

Date: 2025-10-23

This file records the interactive work performed in this workspace during the recent session. It is written so you (or collaborators) can quickly see what was run, where results live, and what to run next.

### What I ran

- Restarted a proper ResNet18 training (20 epochs) using the Hydra experiment `baseline`.
  - Command (launched via nohup):
    - `src/train.py experiment=baseline model=resnet18 trainer.max_epochs=20 ...`
  - Training log: `resnet18_20epochs.log`
  - Training PID (started): 206918 (running in background at time of recording)

- Ran CLIP prompt temperature and CLIP-only experiments earlier (T=0.05 chosen). Artifacts from earlier runs live in `temperature_sweep_results.json` and `plots/`.

- Implemented and ran an intermediate fusion using an existing checkpoint:
  - Checkpoint used: `./logs/mushroom/train/runs/2025-10-22_11-03-32/checkpoints/last.ckpt`
  - Wrapper used: `run_intermediate_fusion_and_plots.sh` (uses venv python)
  - Outputs: `fusion_results_intermediate.json`, plots copied to `plots/`

- Created and executed a prompt diagnostic + pruning pipeline:
  - Script: `prompt_diagnostic_and_prune.py` (launched inside tmux session `prompt_diag`)
  - Purpose: score each prompt by discriminativeness on the validation images, keep top K=8 prompts per class
  - Outputs written:
    - `prompt_discriminativeness.csv` — per-prompt discriminativeness scores
    - `delta_prompts.pruned.json` — pruned prompts (top 8 per class)
    - `fusion_results_full_pruned.json` — fusion results on the full test set using pruned prompts
    - `prompt_diag.log` — full diagnostic run log

### Key numeric results (intermediate)

- Intermediate fusion (using old checkpoint, restricted runs earlier):
  - Example (restricted sample): val α=0.0, test top5≈0.51

- Fusion on full test set with pruned prompts (most recent):
  - Best alpha (val): 0.0 (CLIP-only)
  - Test metrics: top1 = 0.185, top5 = 0.341, balanced_acc = 0.108, macro_f1 = 0.075

### Important file locations

- Training log: `resnet18_20epochs.log`
- ResNet checkpoints: `logs/mushroom/train/runs/<timestamp>/checkpoints/`
- Prompt file (original): `delta_prompts.json`
- Pruned prompt file: `delta_prompts.pruned.json`
- Prompt diagnostic outputs: `prompt_discriminativeness.csv`, `prompt_diag.log`
- Fusion results (intermediate): `fusion_results_intermediate.json`
- Fusion results (full, pruned): `fusion_results_full_pruned.json`
- Plots directory: `plots/` (contains fusion and temperature-sweep figures)

### Commands used (examples)

Start ResNet training (background):

```bash
cd ai4good-mushroom/external/ecovision_mushroom
nohup /zfs/ai4good/student/hgupta/mushroom_env/bin/python3 src/train.py experiment=baseline model=resnet18 logger=csv trainer.max_epochs=20 trainer.accelerator=cpu trainer.precision=32 > /zfs/ai4good/student/hgupta/resnet18_20epochs.log 2>&1 & echo $!
```

Run intermediate fusion+plots on an existing checkpoint (wrapper):

```bash
/zfs/ai4good/student/hgupta/run_intermediate_fusion_and_plots.sh
```

Run prompt diagnostic & pruning (launched in tmux):

```bash
tmux new-session -d -s prompt_diag '/zfs/ai4good/student/hgupta/mushroom_env/bin/python3 /zfs/ai4good/student/hgupta/prompt_diagnostic_and_prune.py --prune_k 8 > /zfs/ai4good/student/hgupta/prompt_diag.log 2>&1'
```

Generate plots from a fusion results file:

```bash
python3 create_fusion_graphs.py --input_json fusion_results_full_pruned.json --out_dir plots/
```

### Recommendations / next steps

1. Let the current ResNet training finish (monitor via `tail -f resnet18_20epochs.log`) and then run a final fusion on the final checkpoint. The monitor script `monitor_resnet_completion.sh` can help notify when training stops.
2. If you have access to a GPU, consider restarting training with `experiment=small_gpu` or `mini` to speed up and improve ResNet quality.
3. Optionally run fusion on multiple checkpoints (continuous watcher) to produce intermediate plots over time.
4. Review `prompt_discriminativeness.csv` and decide whether to accept the pruned file or adjust `K` and re-run pruning.

### Contact / trace

This summary was written automatically by the run agent at 2025-10-23 and saved to this file. If you want a shorter report or to add specific metrics to a `REPORT.md`, tell me and I will create/update it.
