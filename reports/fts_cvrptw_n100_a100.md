# FTS CVRPTW Benchmark

- Generated: `2026-08-13 20:42:09`
- Nodes: `100`
- Ants: `100`
- Samples: `5`
- Instance seed: `100`
- Solver seed: `1234`
- Threads: `1`

## Results

| Metric | Baseline scan | FTS checks | Speedup / Delta |
| --- | ---: | ---: | ---: |
| wall_time | 0.637085 | 0.719264 | 0.885746 |
| time_ant | 0.352787 | 0.437080 | 0.807146 |
| time_ls | 0.000000 | 0.000000 | n/a |
| time_split | 0.000000 | 0.000000 | n/a |
| best_cost | 14.899634 | 13.358946 | -1.540689 |
| mean_cost | 16.416594 | 15.029635 | -1.386959 |
| infeasible | 0.000000 | 0.000000 | 0.000000 |

## Notes

- Baseline uses `use_fts_checks=False`; FTS uses `use_fts_checks=True` with the same binary.
- FTS v1 targets ant perturbation insert checks and inter-route LS one-node relocate checks only.
- `time_ls` and `time_split` are currently not instrumented separately in the backend, so this report uses `time_ant` and wall time for efficiency.
- `infeasible` counts independently verified sampled routes; expected value is `0`.
