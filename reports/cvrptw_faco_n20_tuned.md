# CVRPTW FACO Tuned Benchmark - n=20

Generated on 2026-08-11 from the NeuFACO-constraint repository root.

## Configuration

| Field | Value |
|---|---|
| Problem | CVRPTW |
| Problem scale | `n_nodes=20` customers + depot |
| Instances | `8` |
| Data source | RL4CO CVRPTW `.npz` in `data/cvrptw` |
| Metric | RL4CO `CVRPTWEnv.get_reward()` on the global best route at each outer `T` |
| Solver | FACO, no neural prior |
| `n_ants` | `100` |
| `n_iter` | `10` |
| `mini_H` | `10` |
| `smooth_mmas` | `True` |
| `extend_ls` | `True` |
| `threads` | `8` |
| `seed` | `0` |
| Average inference time | `0.9882s` |
| Result file | `pretrained/cvrptw/20/faco/test_result_ckptnone-rl4co-cvrptw20-ninst8-nants100-niter10-miniH10-threads8-seed0-faco.txt` |

## Command

```powershell
uv run python test.py --method faco -- 20 --data_source rl4co --data_dir data\cvrptw --size 8 --rl4co_root "C:\Users\PHAM ANH KHOI\Projects\rl4co" -i 10 --mini_H 10 --smooth_mmas --extend_ls --threads 8
```

## Results

| T | Avg cost | Avg diversity |
|---:|---:|---:|
| 1 | 1013.1520 | 0.4124 |
| 2 | 1010.2554 | 0.4215 |
| 3 | 1010.2554 | 0.4147 |
| 4 | 1010.2554 | 0.4306 |
| 5 | 1010.2554 | 0.4235 |
| 6 | 1010.2554 | 0.4263 |
| 7 | 1010.2554 | 0.4289 |
| 8 | 1009.4924 | 0.4261 |
| 9 | 1009.4924 | 0.4259 |
| 10 | 1009.4924 | 0.4208 |

## Notes

- The command uses `--smooth_mmas --extend_ls` as two separate flags.
- FACO now evaluates the RL4CO metric only on the global best route once per outer `T`, not on every ant in every mini-iteration.
