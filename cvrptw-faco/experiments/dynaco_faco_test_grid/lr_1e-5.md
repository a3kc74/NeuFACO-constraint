# neural_DyNACO PPO Experiment Report

- Branch: `neural_DyNACO`
- Commit: `5266c82`
- Best epoch: `1`
- Best selected metric: `5.807880`
- Validation mode: `faco_test`; selection mode: `faco_test`

## Command

```powershell
uv run .\train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name lr_1e-5 --report_path experiments\dynaco_faco_test_grid\lr_1e-5.md --lr 1e-5 --ants 32 --train_H 1 --train_mini_H 4 --ppo_clip 0.1
```

## Per-Epoch Metrics

| epoch | train_best_cost | train_mean_cost | faco_test_best_T |
| --- | --- | --- | --- |
| 0 | nan | nan | 5.840701 |
| 1 | 12.772299 | 13.308625 | 5.807880 |
| 2 | 12.762995 | 13.313799 | 5.816594 |
| 3 | 12.833898 | 13.362457 | 5.864123 |
| 4 | 12.672993 | 13.289513 | 5.838334 |
| 5 | 12.641138 | 13.165879 | 5.841975 |
