# neural_DyNACO PPO Experiment Report

- Branch: `neural_DyNACO`
- Commit: `5266c82`
- Best epoch: `2`
- Best selected metric: `5.822951`
- Validation mode: `faco_test`; selection mode: `faco_test`

## Command

```powershell
uv run .\train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name ants64_miniH8_lr1e-5 --report_path experiments\dynaco_faco_test_grid\ants64_miniH8_lr1e-5.md --lr 1e-5 --ants 64 --train_H 1 --train_mini_H 8 --ppo_clip 0.1
```

## Per-Epoch Metrics

| epoch | train_best_cost | train_mean_cost | faco_test_best_T |
| --- | --- | --- | --- |
| 0 | nan | nan | 5.840701 |
| 1 | 12.324030 | 12.875615 | 5.831363 |
| 2 | 12.415032 | 12.897528 | 5.822951 |
| 3 | 12.497382 | 12.989553 | 5.824015 |
| 4 | 12.390736 | 12.970858 | 5.849116 |
| 5 | 12.349308 | 12.953464 | 5.838544 |
