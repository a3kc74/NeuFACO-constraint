# CVRPTW Benchmark Summary - Size 8

Generated on 2026-08-11 from the NeuFACO-constraint repository root.

## Runs

| Run | Data / metric source | Solver config | Instances | Device | Avg inference time | Final T=10 cost | Final T=10 diversity | Result file |
|---|---|---|---:|---|---:|---:|---:|---|
| FACO + RL4CO metric | RL4CO CVRPTW `.npz` in `data/cvrptw` evaluated with `CVRPTWEnv.get_reward()` | `n_ants=100`, `n_iter=10`, `mini_H=1`, `seed=0` | 8 | CPU | 12.0763s | 989.8103 | 0.4327 | `pretrained/cvrptw/20/faco/test_result_ckptnone-rl4co-cvrptw20-ninst8-nants100-niter10-miniH1-threadsdefault-seed0-faco.txt` |
| GFACS ACO no-model + RL4CO metric | Same RL4CO CVRPTW `.npz` in `data/cvrptw`, evaluated with `CVRPTWEnv.get_reward()` | `ACO=AS`, `n_ants=100`, `n_iter=10`, `seed=0`, `model=none` | 8 | CPU | 1.2943s | 983.7913 | 0.5267 | `pretrained/cvrptw/20/no_model/test_result_ckptnone-rl4co-cvrptw20-ninst8-nants100-niter10-seed0.txt` |

## Notes

- Both runs now use the same RL4CO CVRPTW instances and strict RL4CO reward/feasibility metric.
- The earlier `inf` result came from a mismatch in the GFACS PyVRP local-search wrapper: it used `duration_matrix=np.zeros_like(distances)`, so PyVRP time-window feasibility ignored travel time. This is fixed by using `duration_matrix=distances` in `solvers/pyvrp_local_search.py`.
- PyVRP still prints warnings while retrying infeasible local-search candidates, but final benchmark costs are finite under RL4CO validation.

## Commands

```powershell
.\.venv\Scripts\python.exe test.py --method faco -- 20 --data_source rl4co --data_dir data\cvrptw --size 8 --rl4co_root "C:\Users\PHAM ANH KHOI\Projects\rl4co"
```

```powershell
# Bootstrap pre-imports torch to avoid an intermittent Windows c10.dll initialization issue when gfacs_test.py is executed by runpy.
.\.venv\Scripts\python.exe -c "import runpy, sys, torch; sys.argv=['evaluation/gfacs_test.py','20','--size','8','--data_source','rl4co','--data_dir','data/cvrptw','--rl4co_root',r'C:\Users\PHAM ANH KHOI\Projects\rl4co']; runpy.run_path('evaluation/gfacs_test.py', run_name='__main__')"
```

## Per-Iteration Results

| T | FACO + RL4CO cost | FACO + RL4CO diversity | GFACS ACO no-model + RL4CO cost | GFACS ACO no-model + RL4CO diversity |
|---:|---:|---:|---:|---:|
| 1 | 1050.6055 | 0.4950 | 987.8161 | 0.6332 |
| 2 | 1012.9884 | 0.4429 | 983.8271 | 0.6190 |
| 3 | 1005.1906 | 0.4609 | 983.8271 | 0.6056 |
| 4 | 998.0026 | 0.4456 | 983.8271 | 0.5923 |
| 5 | 991.0071 | 0.4317 | 983.7913 | 0.5805 |
| 6 | 990.2441 | 0.4245 | 983.7913 | 0.5649 |
| 7 | 989.8103 | 0.4353 | 983.7913 | 0.5544 |
| 8 | 989.8103 | 0.4370 | 983.7913 | 0.5433 |
| 9 | 989.8103 | 0.4361 | 983.7913 | 0.5362 |
| 10 | 989.8103 | 0.4327 | 983.7913 | 0.5267 |

## Data Files

- RL4CO CVRPTW files used by both runs: `data/cvrptw/cvrptw20_test_seed1234.npz`, `data/cvrptw/cvrptw20_val_seed4321.npz`.
