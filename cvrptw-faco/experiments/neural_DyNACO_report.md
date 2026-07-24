# neural_DyNACO PPO Experiment Report

- Branch: `neural_DyNACO`
- Commit: `5266c82`
- Best epoch: `1`
- Best selected metric: `5.844298`
- Validation mode: `both`; selection mode: `faco_test_ib`

## Command

```powershell
uv run .\train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --val_infer_mode both --report_path experiments\neural_DyNACO_report.md --disable_wandb
```

## Per-Epoch Metrics

| epoch | train_best_cost | train_mean_cost | faco_test_best_T | faco_test_ib_best_T |
| --- | --- | --- | --- | --- |
| 0 | nan | nan | 5.840701 | 5.846393 |
| 1 | 12.753110 | 13.286332 | 5.807880 | 5.844298 |
| 2 | 12.814298 | 13.371267 | 5.811824 | 5.845087 |
| 3 | 12.800796 | 13.364528 | 5.835847 | 5.864362 |
| 4 | 12.723959 | 13.312877 | 5.825606 | 5.862887 |
| 5 | 12.642584 | 13.301152 | 5.828083 | 5.865832 |
