# FACO Mode 3 Slack Weight Gridsearch

- Command base: `uv run test.py --method faco --problem CVRPTW -- 100 -i 100 --elite_source_period 10 --n_ants 100 -s 16 --smooth_mmas --extend_ls --threads 8 --log_period 10`
- Grid: `--granular_mode 3 --granular_slack_weight {0.0,0.02,0.05,0.1,0.15,0.2,0.3,0.5}`.
- Dataset/source: `gfacs`, CVRPTW100, `16` instances, seed `0`, `100` ants, `100` iterations.
- Static TW weights kept default: `granular_wait_weight=0.2`, `granular_time_warp_weight=1.0`.
- Best grid weight: `0.3` with final cost `5.717744` and average time `3.929053s`.

## Summary

| Slack Weight | Avg Time (s) | Final Cost | Final Diversity | Delta Cost vs Mode 0 | Delta Cost vs Mode 1 | Delta Time vs Mode 0 | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 3.905895 | 5.740324 | 0.295103 | +0.022331 | +0.023684 | -0.076643 | worse |
| 0.02 | 3.916085 | 5.723837 | 0.286671 | +0.005844 | +0.007197 | -0.066453 | better than 0.2 |
| 0.05 | 3.935341 | 5.744223 | 0.288622 | +0.026230 | +0.027583 | -0.047197 | worse |
| 0.1 | 3.906368 | 5.739146 | 0.286932 | +0.021153 | +0.022506 | -0.076170 | worse |
| 0.15 | 3.994644 | 5.741302 | 0.289206 | +0.023309 | +0.024662 | +0.012106 | worse |
| 0.2 | 3.934311 | 5.738781 | 0.290969 | +0.020788 | +0.022141 | -0.048227 | worse |
| 0.3 | 3.929053 | 5.717744 | 0.281476 | -0.000249 | +0.001104 | -0.053485 | best grid |
| 0.5 | 3.923776 | 5.729491 | 0.289162 | +0.011498 | +0.012851 | -0.058762 | better than 0.2 |

## Ranking By Final Cost

| Rank | Slack Weight | Final Cost | Avg Time (s) | Final Diversity |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3 | 5.717744 | 3.929053 | 0.281476 |
| 2 | 0.02 | 5.723837 | 3.916085 | 0.286671 |
| 3 | 0.5 | 5.729491 | 3.923776 | 0.289162 |
| 4 | 0.2 | 5.738781 | 3.934311 | 0.290969 |
| 5 | 0.1 | 5.739146 | 3.906368 | 0.286932 |
| 6 | 0 | 5.740324 | 3.905895 | 0.295103 |
| 7 | 0.15 | 5.741302 | 3.994644 | 0.289206 |
| 8 | 0.05 | 5.744223 | 3.935341 | 0.288622 |

## Cost Curve

| T | w=0 | w=0.02 | w=0.05 | w=0.1 | w=0.15 | w=0.2 | w=0.3 | w=0.5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.812898 | 5.807099 | 5.820366 | 5.807719 | 5.811824 | 5.812436 | 5.797569 | 5.822650 |
| 2 | 5.776464 | 5.772011 | 5.788976 | 5.792386 | 5.784720 | 5.785709 | 5.765445 | 5.779708 |
| 3 | 5.770931 | 5.765604 | 5.779412 | 5.779662 | 5.778720 | 5.761428 | 5.751664 | 5.764519 |
| 4 | 5.759213 | 5.743925 | 5.770734 | 5.766591 | 5.763289 | 5.749283 | 5.739446 | 5.754035 |
| 5 | 5.754546 | 5.734502 | 5.761143 | 5.759922 | 5.752165 | 5.745109 | 5.731699 | 5.745202 |
| 6 | 5.749043 | 5.731296 | 5.753726 | 5.754724 | 5.744963 | 5.743259 | 5.722923 | 5.740643 |
| 7 | 5.744841 | 5.727609 | 5.748441 | 5.752568 | 5.743668 | 5.741529 | 5.719866 | 5.734797 |
| 8 | 5.741214 | 5.726738 | 5.745883 | 5.752267 | 5.743144 | 5.740493 | 5.718675 | 5.732810 |
| 9 | 5.740773 | 5.725807 | 5.744829 | 5.752016 | 5.742281 | 5.738983 | 5.717744 | 5.731274 |
| 10 | 5.740324 | 5.723837 | 5.744223 | 5.739146 | 5.741302 | 5.738781 | 5.717744 | 5.729491 |

## Diversity Curve

| T | w=0 | w=0.02 | w=0.05 | w=0.1 | w=0.15 | w=0.2 | w=0.3 | w=0.5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.289547 | 0.285653 | 0.275282 | 0.272813 | 0.278866 | 0.281633 | 0.285877 | 0.289092 |
| 2 | 0.290666 | 0.287284 | 0.286529 | 0.268107 | 0.277872 | 0.285616 | 0.293989 | 0.293048 |
| 3 | 0.283409 | 0.282948 | 0.285207 | 0.283352 | 0.276763 | 0.284245 | 0.289499 | 0.291673 |
| 4 | 0.293195 | 0.286467 | 0.285387 | 0.287523 | 0.285790 | 0.280554 | 0.288545 | 0.283760 |
| 5 | 0.293241 | 0.289692 | 0.293804 | 0.288778 | 0.286953 | 0.286624 | 0.287302 | 0.286968 |
| 6 | 0.293635 | 0.291205 | 0.288943 | 0.287761 | 0.285345 | 0.287839 | 0.295304 | 0.290909 |
| 7 | 0.291983 | 0.285351 | 0.282254 | 0.285250 | 0.284161 | 0.289601 | 0.288117 | 0.293153 |
| 8 | 0.292795 | 0.288306 | 0.287390 | 0.284677 | 0.285754 | 0.289864 | 0.291941 | 0.295534 |
| 9 | 0.288592 | 0.286303 | 0.287462 | 0.285718 | 0.284291 | 0.291494 | 0.279130 | 0.294294 |
| 10 | 0.295103 | 0.286671 | 0.288622 | 0.286932 | 0.289206 | 0.290969 | 0.281476 | 0.289162 |

## Interpretation

- `slack_weight=0.3` is the best tested Mode 3 setting: final cost `5.717744`, slightly better than Mode 0 Euclidean baseline `5.717993` by `-0.000249`.
- `slack_weight=0.3` is still slightly worse than Mode 1 static spatio-temporal KNN `5.716640` by `+0.001104`, so Mode 1 remains the safest default from these runs.
- Very small damping (`0.02`) is the second-best Mode 3 result, while `0.05`, `0.10`, `0.15`, and the old `0.20` are inconsistent/worse on this seed.
- `slack_weight=0.0` under Mode 3 matches the Mode 2-style temporal heuristic path in practice, confirming the slack component is the differentiator for this grid.
- Runtime differences are small; tuning mainly affects solution quality, not speed, because damping only changes probabilities over already-generated candidates.

## Recommendation

- If keeping current Mode 3 implementation, set experimental `granular_slack_weight=0.3` instead of `0.2`.
- If choosing production default from the full comparison, keep Mode 1 first; use Mode 3 `0.3` only as an exploration variant needing more seeds/instances.
- Next useful test: rerun weights around `0.25,0.30,0.35,0.40` across multiple seeds to see whether `0.3` is robust or seed-specific.

## Result Files

- `w=0`: `reports\granular_mode3_slack_grid\slack_0p0\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode3-threads8-seed0-faco.txt`
- `w=0.02`: `reports\granular_mode3_slack_grid\slack_0p02\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode3-threads8-seed0-faco.txt`
- `w=0.05`: `reports\granular_mode3_slack_grid\slack_0p05\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode3-threads8-seed0-faco.txt`
- `w=0.1`: `reports\granular_mode3_slack_grid\slack_0p1\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode3-threads8-seed0-faco.txt`
- `w=0.15`: `reports\granular_mode3_slack_grid\slack_0p15\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode3-threads8-seed0-faco.txt`
- `w=0.2`: `reports\granular_mode3_slack_grid\slack_0p2\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode3-threads8-seed0-faco.txt`
- `w=0.3`: `reports\granular_mode3_slack_grid\slack_0p3\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode3-threads8-seed0-faco.txt`
- `w=0.5`: `reports\granular_mode3_slack_grid\slack_0p5\test_result_ckptnone-gfacs-cvrptw100-ninst16-nants100-niter100-miniH1-logperiod10-eliteperiod10-gmode3-threads8-seed0-faco.txt`
