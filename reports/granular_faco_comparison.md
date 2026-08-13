# FACO Granular KNN Comparison

- Command base: `uv run test.py --method faco --problem CVRPTW -- 100 -i 100 --elite_source_period 10 --n_ants 100 -s 16 --smooth_mmas --extend_ls --threads 8 --log_period 10`
- Dataset/source: `gfacs`, CVRPTW100, `16` instances, seed `0`.
- Granular weights: `wait=0.2`, `time_warp=1.0`, `slack=0.2`.
- Best final cost: mode `1` (Static spatio-temporal KNN) = `5.716640`.
- Fastest: mode `1` (Static spatio-temporal KNN) = `3.887313s` average inference time.

## Summary

| Mode | Config | Avg Time (s) | Final Cost | Delta Cost vs 0 | Final Diversity | Delta Time vs 0 | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | Euclidean KNN | 3.982538 | 5.717993 | +0.000000 | 0.291012 | +0.000000 | baseline |
| 1 | Static spatio-temporal KNN | 3.887313 | 5.716640 | -0.001353 | 0.281993 | -0.095225 | better cost |
| 2 | Static TW KNN + temporal heuristic | 4.054828 | 5.740324 | +0.022331 | 0.295103 | +0.072289 | worse cost |
| 3 | Dynamic slack rerank | 3.930182 | 5.738781 | +0.020788 | 0.290969 | -0.052356 | worse cost |

## Cost Curve

| T | Mode 0 | Mode 1 | Mode 2 | Mode 3 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.793379 | 5.790847 | 5.812898 | 5.812436 |
| 2 | 5.748236 | 5.754246 | 5.776464 | 5.785709 |
| 3 | 5.740204 | 5.744059 | 5.770931 | 5.761428 |
| 4 | 5.735658 | 5.738859 | 5.759213 | 5.749283 |
| 5 | 5.733060 | 5.736082 | 5.754546 | 5.745109 |
| 6 | 5.729429 | 5.729852 | 5.749043 | 5.743259 |
| 7 | 5.724087 | 5.724032 | 5.744841 | 5.741529 |
| 8 | 5.722751 | 5.723518 | 5.741214 | 5.740493 |
| 9 | 5.718691 | 5.717219 | 5.740773 | 5.738983 |
| 10 | 5.717993 | 5.716640 | 5.740324 | 5.738781 |

## Diversity Curve

| T | Mode 0 | Mode 1 | Mode 2 | Mode 3 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.283531 | 0.281808 | 0.289547 | 0.281633 |
| 2 | 0.283350 | 0.279969 | 0.290666 | 0.285616 |
| 3 | 0.280948 | 0.275916 | 0.283409 | 0.284245 |
| 4 | 0.282648 | 0.273443 | 0.293195 | 0.280554 |
| 5 | 0.288421 | 0.277672 | 0.293241 | 0.286624 |
| 6 | 0.286534 | 0.278428 | 0.293635 | 0.287839 |
| 7 | 0.287147 | 0.279703 | 0.291983 | 0.289601 |
| 8 | 0.291311 | 0.279971 | 0.292795 | 0.289864 |
| 9 | 0.284045 | 0.276974 | 0.288592 | 0.291494 |
| 10 | 0.291012 | 0.281993 | 0.295103 | 0.290969 |

## Interpretation

- Mode 1 (static spatio-temporal KNN) is slightly better than Euclidean baseline on final cost and also slightly faster in this run.
- Mode 2 (temporal heuristic damping) is worse on cost here, suggesting the heuristic penalty is too strong or misaligned with FACO pheromone/savings.
- Mode 3 (dynamic slack rerank) is also worse than modes 0/1 with current `slack=0.2`; it may need a smaller weight or rerank-only over a larger static pool.
- Best candidate for follow-up is mode 1 as the new default candidate-list builder, then tune weights before re-enabling heuristic/slack variants.

## Result Files

- Mode 0: `reports\granular_faco_cli_runs\mode0_euclidean\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode0-threads8-seed0-faco.txt`
- Mode 1: `reports\granular_faco_cli_runs\mode1_static_tw\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode1-threads8-seed0-faco.txt`
- Mode 2: `reports\granular_faco_cli_runs\mode2_static_tw_heuristic\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode2-threads8-seed0-faco.txt`
- Mode 3: `reports\granular_faco_cli_runs\mode3_dynamic_slack\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode3-threads8-seed0-faco.txt`
