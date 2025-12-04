### Hypertuning Engine
This folder includes:
- `few_shot_hyper_test.py` – full sweeps
- `few_shot_temp.py` – temperature experiments
- `few_shot_alpha_cast.py` – alpha sweeps
- `train_best_model.py` – final training
- `eval_final_model.py` – final evaluation

Outputs:
- sweep JSONs
- global CSVs
- per-class results
- `best_alpha.json`
- `final_model.pt`