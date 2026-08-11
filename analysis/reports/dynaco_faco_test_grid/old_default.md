# neural_DyNACO PPO Experiment Report

- Branch: `neural_DyNACO`
- Commit: `5266c82`
- Best epoch: `1`
- Best selected metric: `5.807880`
- Validation mode: `faco_test`; selection mode: `faco_test`

## Command

```powershell
uv run .\train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name old_default --report_path experiments\dynaco_faco_test_grid\old_default.md --lr 5e-6 --ants 32 --train_H 1 --train_mini_H 4 --ppo_clip 0.1
```

## Per-Epoch Metrics

| epoch | train_best_cost | train_mean_cost | faco_test_best_T |
| --- | --- | --- | --- |
| 0 | nan | nan | 5.840701 |
| 1 | 12.753110 | 13.286332 | 5.807880 |
| 2 | 12.814298 | 13.371267 | 5.811824 |
| 3 | 12.800796 | 13.364528 | 5.835847 |
| 4 | 12.723959 | 13.312877 | 5.825606 |
| 5 | 12.642584 | 13.301152 | 5.828083 |
