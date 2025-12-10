# import_n_config

This directory contains centralized configuration and import management for the few-shot learning pipeline used in the mushroom classification project. 

## Directory Contents

### Core Files

- **`constants.py`** - Centralized constants and configuration parameters
  - Model backbone options and hyperparameter sweeps
  - Directory paths for data, results, and cache
  - Final model hyperparameters
  - Computational settings (device, thread counts, batch sizes)

- **`shared_setup.py`** - Shared imports and common dependencies
  - Core libraries (numpy, pandas, torch, scikit-learn)
  - Open CLIP integration
  - Utility functions (tqdm, etc.)
  - Type hints and general purpose imports

- **`train_setup.py`** - Training script imports and configuration
  - Imports from shared setup and constants
  - Specialized constants for model training
  - Re-exports all globals for convenient importing

- **`eval_setup.py`** - Evaluation script imports and configuration
  - Imports from shared setup and constants
  - Model evaluation specific constants
  - Re-exports all globals for convenient importing

- **`hyper_setup.py`** - Hyperparameter tuning script imports and configuration
  - PyTorch optimization settings (matmul precision, cudnn benchmarking)
  - Thread pool configuration for parallel processing
  - Hyperparameter sweep constants
  - Re-exports all globals for convenient importing

## Key Constants

### Paths
- **WORK_ENV**: Root directory for code and results
- **DATA_ROOT**: Mushroom dataset location
- **RESULTS_DIR**: Output directory for results and trained models
- **DEFAULT_CACHE_DIR**: Feature cache directory

### Hyperparameter Grids
- **SHOTS**: Few-shot counts to test `[0, 1, 5, 10, 20, 50, 100]`
- **ALPHAS**: Linear probe weight combinations `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`
- **PROMPT_SET**: Prompt templates to evaluate `["ensemble", "v1", "names", "delta"]`
- **LR_GRID**: Learning rates to sweep `[1e-3, 3e-3, 3e-2, 3e-1]`
- **WD_GRID**: Weight decay values `[0, 1e-4, 5e-4]`

### Final Model Configuration
- **FINAL_BACKBONE**: Selected backbone model
- **FINAL_SHOTS**: Number of few-shots for final training
- **FINAL_PROMPT_SET**: Prompt template for final model
- **FINAL_LR**: Final model learning rate
- **FINAL_WD**: Final model weight decay
- **FINAL_EPOCHS**: Training epochs (200)
- **FINAL_BATCH_SIZE**: Batch size (512)

## Configuration Adjustments

To modify the pipeline configuration:

1. **Paths**: Update directory variables at the top of `constants.py`
2. **Model backbone**: Toggle `BACKBONE_TOGGLE` in `constants.py` (0 or 1)
3. **Computational settings**: Modify thread counts and CUDA settings in `hyper_setup.py`

## Dependencies

All required dependencies are imported through `shared_setup.py`:
- PyTorch
- NumPy, Pandas
- scikit-learn
- open_clip (via open_clip_torch)
- PIL
- tqdm
