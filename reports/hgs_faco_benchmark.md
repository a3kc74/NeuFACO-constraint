# HGS-style FACO Benchmark Report

## Common Config
- Base command: `uv run test.py --method faco --problem CVRPTW -- 100 -i 100 --elite_source_period 10 --n_ants 100 -s 16 --smooth_mmas --extend_ls --threads 8 --log_period 10 --granular_mode 1`
- Dataset/run: `CVRPTW100`, `16` instances, `100` ants, `100` iterations, seed `0`, threads `8`.
- All benchmark versions include `--granular_mode 1` as requested.
- Baseline is FACO after granular refactor, without HGS deep/soft variants.

## Final Tried Versions
| Version | Change | Final cost | Improvement vs V0 | Diversity | Avg time/instance | Slowdown |
|---|---|---:|---:|---:|---:|---:|
| V0 | Granular mode 1 baseline | 5.716640 | +0.00% | 0.281993 | 3.897s | 1.00x |
| V1 | Optimized soft deep LS r=1 | 5.684886 | +0.56% | 0.285391 | 10.054s | 2.58x |
| V2 | Segment RELOCATE* + SWAP* + 2-OPT* r=1 | 5.673635 | +0.75% | 0.285441 | 11.576s | 2.97x |
| V3 | Segment RELOCATE* + SWAP* + 2-OPT* r=2 | 5.667048 | +0.87% | 0.271937 | 18.585s | 4.77x |
| V4 | Selective top-8 post-sample r=2 | 5.699694 | +0.30% | 0.284749 | 16.693s | 4.28x |
| V5 | Adaptive penalty top-8 r=2 | 5.712497 | +0.07% | 0.284425 | 21.090s | 5.41x |
| V6 | Soft cheap LS only | 6.078068 | -6.32% | 0.305420 | 3.640s | 0.93x |

## Conclusion
- Best current-code quality: `V3`, final cost `5.667048`, improvement `0.87%` vs V0.
- Adding deep `2-OPT*` improved best cost from `5.677016` to `5.667048`.
- Penalty probes (`TW=5/20/50`, `cap=50`) at 30 iterations did not beat default `TW=10, cap=10`.
- Adaptive penalty on post-sample top-k is implemented but did not improve quality here (`V5` only `+0.07%`).
- Best speed/quality compromise remains `V1`: `+0.56%` with `2.58x` slowdown.
- Best quality variant `V3` gives `+0.87%` with `4.77x` slowdown.
- Soft infeasible search inside cheap inter-route LS is harmful (`V6`), so it stays disabled by default.
- The requested `+5%` target is not reached. Best validated gain is `+0.87%`; implementation now includes HGS-style soft deep search, adaptive penalty hooks, RELOCATE*, segment RELOCATE*, SWAP*, and 2-OPT*.

## Implemented Design Mapping
- Ant construction remains hard-feasible; infeasible LS outputs are rolled back before return/best/deposit.
- Deep local search uses opt-in HGS-style penalized objective via `--hgs_soft_deep_ls`: `travel + lambda_Q * capacity_violation + lambda_TW * time_warp`.
- Adaptive penalty hooks are exposed via `--hgs_adaptive_penalty` and `--hgs_target_feasible`; current validated use is post-sample top-k.
- Selective intensification is opt-in via `--hgs_deep_ls`, running candidate-restricted `RELOCATE*`, segment `RELOCATE*` length 2/3, `SWAP*`, and `2-OPT*` after cheap RVND.
- Deep move scoring evaluates remove/insert/tail-exchange sequences without allocating/copying full route vectors in inner loops.
- Optional post-sample trigger `--hgs_deep_top_k` is implemented but did not outperform inline deep LS on this config.
- Soft cheap LS is separated behind `--hgs_soft_cheap_ls` after experiments showed it is not robust enough yet.
- Deep and inter-route soft stages have local rollback guards to avoid discarding preceding feasible cheap-RVND results.

## Validation
- Build: `uv run python cpp/faco/setup.py build_ext --inplace` passed.
- Tests: `uv run pytest tests/test_mfaco_cvrptw.py tests/test_faco_test_cli.py -q` passed with `20 passed`.

## Artifacts
- `V0`: `reports/hgs_faco_runs/v0_granular1_baseline/test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode1-threads8-seed0-faco.txt`
- `V1`: `reports/hgs_faco_runs/optimized_final_softdeep_r1_p10/test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode1-threads8-seed0-faco.txt`
- `V2`: `reports/hgs_faco_runs/deep2opt_final_softdeep_r1_p10/test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode1-threads8-seed0-faco.txt`
- `V3`: `reports/hgs_faco_runs/deep2opt_final_softdeep_r2_p10/test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode1-threads8-seed0-faco.txt`
- `V4`: `reports/hgs_faco_runs/selective_top8_r2_p10/test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode1-threads8-seed0-faco.txt`
- `V5`: `reports/hgs_faco_runs/adaptive_top8_r2_p10/test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode1-threads8-seed0-faco.txt`
- `V6`: `reports/hgs_faco_runs/final_v4_softcheap_only_p10_rollback/test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode1-threads8-seed0-faco.txt`
