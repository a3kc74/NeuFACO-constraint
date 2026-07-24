# DyNACO FACO Test Grid Summary

- Winner by final epoch: `miniH_10_lr1e-5` with `faco_test_best_T=5.816833`
- Winner improvement vs epoch 0: `0.023868`
- Validation mode: `faco_test` only
- Objective: lowest final epoch `faco_test_best_T`

## Results

| rank | name | epoch0 | final_epoch | final | improvement | best_epoch | best | runtime_sec | ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | miniH_10_lr1e-5 | 5.840701 | 5 | 5.816833 | 0.023868 | 3 | 5.813355 | 0.000000 | True |
| 2 | old_default | 5.840701 | 5 | 5.828083 | 0.012618 | 1 | 5.807880 | 0.000000 | True |
| 3 | miniH_8 | 5.840701 | 5 | 5.830055 | 0.010646 | 3 | 5.808170 | 0.000000 | True |
| 4 | ants_64 | 5.840701 | 5 | 5.833382 | 0.007319 | 1 | 5.805103 | 0.000000 | True |
| 5 | clip_0.2_entropy | 5.840701 | 5 | 5.836961 | 0.003740 | 2 | 5.805375 | 0.000000 | True |
| 6 | ants64_miniH8_lr1e-5 | 5.840701 | 5 | 5.838544 | 0.002157 | 2 | 5.822951 | 0.000000 | True |
| 7 | lr_1e-5 | 5.840701 | 5 | 5.841975 | -0.001274 | 1 | 5.807880 | 0.000000 | True |
| 8 | lr_2e-5 | 5.840701 | 5 | 5.863178 | -0.022477 | 2 | 5.804920 | 0.000000 | True |

## Commands

### old_default

```powershell
uv run python train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name old_default --report_path experiments\dynaco_faco_test_grid\old_default.md --lr 5e-6 --ants 32 --train_H 1 --train_mini_H 4 --ppo_clip 0.1
```

### lr_1e-5

```powershell
uv run python train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name lr_1e-5 --report_path experiments\dynaco_faco_test_grid\lr_1e-5.md --lr 1e-5 --ants 32 --train_H 1 --train_mini_H 4 --ppo_clip 0.1
```

### lr_2e-5

```powershell
uv run python train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name lr_2e-5 --report_path experiments\dynaco_faco_test_grid\lr_2e-5.md --lr 2e-5 --ants 32 --train_H 1 --train_mini_H 4 --ppo_clip 0.1
```

### miniH_8

```powershell
uv run python train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name miniH_8 --report_path experiments\dynaco_faco_test_grid\miniH_8.md --lr 5e-6 --ants 32 --train_H 1 --train_mini_H 8 --ppo_clip 0.1
```

### miniH_10_lr1e-5

```powershell
uv run python train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name miniH_10_lr1e-5 --report_path experiments\dynaco_faco_test_grid\miniH_10_lr1e-5.md --lr 1e-5 --ants 32 --train_H 1 --train_mini_H 10 --ppo_clip 0.1
```

### ants_64

```powershell
uv run python train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name ants_64 --report_path experiments\dynaco_faco_test_grid\ants_64.md --lr 5e-6 --ants 64 --train_H 1 --train_mini_H 4 --ppo_clip 0.1
```

### ants64_miniH8_lr1e-5

```powershell
uv run python train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name ants64_miniH8_lr1e-5 --report_path experiments\dynaco_faco_test_grid\ants64_miniH8_lr1e-5.md --lr 1e-5 --ants 64 --train_H 1 --train_mini_H 8 --ppo_clip 0.1
```

### clip_0.2_entropy

```powershell
uv run python train_dynaco_ppo.py 100 --threads 8 --batch_size 1 --val_size 8 --val_ants 100 --min_new_edges 8 --val_n_iter 10 --val_mini_H 10 --smooth_mmas --extend_ls --use_local_search --epochs 5 --steps 20 --disable_wandb --val_infer_mode faco_test --select_metric_mode faco_test --run_name clip_0.2_entropy --report_path experiments\dynaco_faco_test_grid\clip_0.2_entropy.md --lr 1e-5 --ants 32 --train_H 1 --train_mini_H 4 --ppo_clip 0.2 --entropy_coeff 0.01
```

