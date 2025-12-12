### Prompt Files

Each JSON file maps class names to lists of prompts.

Prompt sets include:
- `class_prompts_plain.json`
- `class_prompts_attr.json`
- `class_prompts_enriched.json`
- Cleaned versions based on BLIP captions

Used in:
- `eval_zero_shot.py`
- `test_few_shots_overall.py`
- `few_shot_hyper_test.py`
- `train_best_model.py`