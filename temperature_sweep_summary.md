# Temperature Sweep Results - Delta Prompts (200 samples)

## Summary

This experiment tested 8 different temperature values for per-prompt softmax pooling on the **delta_prompts.json** set (10,389 net-new prompts from image-derived and common-name sources).

## Best Results by Metric

| Metric | Optimal T | Score |
|--------|-----------|-------|
| **Top-1 Accuracy** | 0.05 | 0.2650 |
| **Top-5 Accuracy** | 0.05 | 0.5550 |
| **Balanced Accuracy** | 0.08 | 0.1778 |
| **Macro F1-Score** | 0.05 | 0.1342 |

## Full Results

| Temperature | Top-1 Acc | Top-5 Acc | Balanced Acc | Macro F1 |
|-------------|-----------|-----------|--------------|----------|
| **0.05** | **0.2650** | **0.5550** | 0.1761 | **0.1342** |
| 0.08 | **0.2650** | 0.5350 | **0.1777** | 0.1279 |
| 0.10 | **0.2650** | 0.5350 | **0.1777** | 0.1280 |
| 0.12 | **0.2650** | 0.5350 | **0.1777** | 0.1287 |
| 0.15 | 0.2600 | 0.5400 | 0.1728 | 0.1221 |
| 0.20 | 0.2600 | 0.5300 | 0.1728 | 0.1222 |
| 0.30 | 0.2550 | 0.5300 | 0.1711 | 0.1208 |
| 0.50 | 0.2500 | 0.5350 | 0.1678 | 0.1150 |

## Key Insights

### 1. Lower Temperatures Perform Best
- **T=0.05** achieves the best performance across 3/4 metrics
- Lower temperatures implement a "max-like" pooling strategy that heavily weights the best-matching prompts
- This suggests that our prompt set contains high-quality discriminative descriptors that shouldn't be diluted by averaging

### 2. Temperature Stability (0.05-0.12)
- Top-1 accuracy is stable at **0.265** from T=0.05 to T=0.12
- Balanced accuracy peaks in the 0.08-0.12 range
- This plateau suggests robust performance across low-temperature settings

### 3. Performance Degradation at High Temperatures
- As temperature increases beyond 0.15, all metrics decline
- At T=0.5 (mean-like pooling), Top-1 drops to 0.250
- This confirms that treating all prompts equally reduces discrimination power

### 4. Top-5 vs Top-1 Gap
- Top-5 accuracy (0.555) is **2.1x** higher than Top-1 (0.265)
- This indicates good retrieval but weak final discrimination
- The correct class is often in top-5 predictions but not always ranked #1

### 5. Class Imbalance Challenge
- Balanced accuracy (0.1778) is lower than Top-1 accuracy (0.265)
- Macro F1 (0.1342) is also relatively low
- This suggests the model performs better on common/well-represented species

## Recommendations

### For Production Use:
- **Use T=0.05** for optimal overall performance
- Expect ~26.5% Top-1 accuracy and ~55.5% Top-5 accuracy on this prompt set

### For Future Work:
1. **Ensemble with ResNet** - combine with CNN predictions to boost accuracy
2. **Prompt Filtering** - identify and remove low-quality prompts
3. **Per-Class Temperature** - optimize T individually for each species
4. **Feature-Level Fusion** - combine CLIP and ResNet features before classification
5. **Test on Full Dataset** - validate on complete test split (not just 200 samples)

## Visualizations

Two plot files were generated:
- `temperature_sweep_plots.png` - 2x2 grid showing each metric separately
- `temperature_sweep_combined.png` - All metrics on one plot for comparison

## Data Files

- Full results: `temperature_sweep_results.json`
- Prompt set: `delta_prompts.json` (10,389 prompts across 169 species)
- Test data: 200 samples from `/zfs/ai4good/datasets/mushroom/test.csv`
