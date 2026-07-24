# neural_DyNACO PPO Experiment Report

- Branch: `neural_DyNACO`
- Commit: `5266c82`
- Best epoch: `3`
- Best selected metric: `5.808170`
- Validation mode: `faco_test`; selection mode: `faco_test`

## Command

```powershell
uv run .\train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name miniH_8 --report_path experiments\dynaco_faco_test_grid\miniH_8.md --lr 5e-6 --ants 32 --train_H 1 --train_mini_H 8 --ppo_clip 0.1
```

## Per-Epoch Metrics

| epoch | train_best_cost | train_mean_cost | faco_test_best_T |
| --- | --- | --- | --- |
| 0 | nan | nan | 5.840701 |
| 1 | 12.545699 | 13.045603 | 5.816179 |
| 2 | 12.526932 | 13.014640 | 5.831825 |
| 3 | 12.565195 | 13.043954 | 5.808170 |
| 4 | 12.496563 | 13.005659 | 5.819589 |
| 5 | 12.427886 | 12.951421 | 5.830055 |
