# FTS CVRPTW Benchmark

- Generated: `2026-08-13 20:56:21`
- Nodes: `100`
- Ants: `100`
- Samples: `20`
- Instance seed: `100`
- Solver seed: `1234`
- Threads: `1`

## Results

| Metric | Baseline scan | FTS checks | Speedup / Delta |
| --- | ---: | ---: | ---: |
| wall_time | 2.733488 | 2.306348 | 1.185202 |
| time_ant | 1.599486 | 1.161141 | 1.377512 |
| time_ls | 0.000000 | 0.000000 | n/a |
| time_split | 0.000000 | 0.000000 | n/a |
| fts_checks | 0.000000 | 1746664.000000 | 1746664.000000 |
| fts_fallback_scans | 1760779.000000 | 1658557.000000 | -102222.000000 |
| best_cost | 13.195335 | 13.195335 | 0.000000 |
| mean_cost | 14.992135 | 14.992135 | 0.000000 |
| infeasible | 0.000000 | 0.000000 | 0.000000 |

## Notes

- Baseline uses `use_fts_checks=False`; FTS uses `use_fts_checks=True` with the same binary.
- FTS uses metadata fast-paths for ant relocation insert/remove checks plus optimized in-place fallback simulation; remaining LS move cases keep scan fallback because all-move FTS scans were slower on this workload.
- `time_ls` and `time_split` are currently not instrumented separately in the backend, so this report uses `time_ant` and wall time for efficiency.
- `infeasible` counts independently verified sampled routes; expected value is `0`.
