# FACO Diversity Experiment Results

## diverse_elite_archive

- Command: `uv run .\test_diverse_elite_archive.py 1000 -i 10 --mini_H 10 --n_ants 50 -s 8 --smooth_mmas --cand_list_size 32 --extend_ls --threads 8`
- Versions tested: `[{"elite_cost_tolerance": 1.03, "elite_k": 4, "elite_min_diversity": 0.1}, {"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15}, {"elite_cost_tolerance": 1.1, "elite_k": 12, "elite_min_diversity": 0.2}]`
- Best config: `{"elite_cost_tolerance": 1.03, "elite_k": 4, "elite_min_diversity": 0.1}`
- Final avg cost: `45.415535`
- Final avg diversity: `0.065547`
- Avg time: `11.353728` seconds
- Raw results: `cvrptw-faco/diversity_results/diverse_elite_archive.json`, `cvrptw-faco/diversity_results/diverse_elite_archive.csv`

## topk_source_archive

- Command: `uv run .\test_topk_source_archive.py 1000 -i 10 --mini_H 10 --n_ants 50 -s 8 --smooth_mmas --cand_list_size 32 --extend_ls --threads 8`
- Versions tested: `[{"top_k": 4}, {"top_k": 8}, {"top_k": 12}]`
- Best config: `{"top_k": 4}`
- Final avg cost: `45.353977`
- Final avg diversity: `0.063081`
- Avg time: `6.753710` seconds
- Raw results: `cvrptw-faco/diversity_results/topk_source_archive.json`, `cvrptw-faco/diversity_results/topk_source_archive.csv`

## exploration_schedule

- Command: `uv run .\test_exploration_schedule.py 1000 -i 10 --mini_H 10 --n_ants 50 -s 8 --smooth_mmas --cand_list_size 32 --extend_ls --threads 8`
- Versions tested: `[{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "late_global_prob": 0.5}, {"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "late_global_prob": 0.7}, {"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "late_global_prob": 0.9}]`
- Best config: `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "late_global_prob": 0.9}`
- Final avg cost: `45.702614`
- Final avg diversity: `0.061380`
- Avg time: `20.153594` seconds
- Raw results: `cvrptw-faco/diversity_results/exploration_schedule.json`, `cvrptw-faco/diversity_results/exploration_schedule.csv`

## multisource_sampling

- Command: `uv run .\test_multisource_sampling.py 1000 -i 10 --mini_H 10 --n_ants 50 -s 8 --smooth_mmas --cand_list_size 32 --extend_ls --threads 8`
- Versions tested: `[{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "source_count": 2}, {"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "source_count": 4}]`
- Best config: `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "source_count": 2}`
- Final avg cost: `45.250885`
- Final avg diversity: `0.182913`
- Avg time: `28.944272` seconds
- Raw results: `cvrptw-faco/diversity_results/multisource_sampling.json`, `cvrptw-faco/diversity_results/multisource_sampling.csv`

## rank_topk_pheromone

- Command: `uv run .\test_rank_topk_pheromone.py 1000 -i 10 --mini_H 10 --n_ants 50 -s 8 --smooth_mmas --cand_list_size 32 --extend_ls --threads 8`
- Versions tested: `[{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "pheromone_top_k": 3}, {"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "pheromone_top_k": 5}]`
- Best config: `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "pheromone_top_k": 3}`
- Final avg cost: `45.967525`
- Final avg diversity: `0.070906`
- Avg time: `18.098780` seconds
- Raw results: `cvrptw-faco/diversity_results/rank_topk_pheromone.json`, `cvrptw-faco/diversity_results/rank_topk_pheromone.csv`

## gfacs_rank_smoothing

- Command: `uv run .\test_gfacs_rank_smoothing.py 1000 -i 10 --mini_H 10 --n_ants 50 -s 8 --smooth_mmas --cand_list_size 32 --extend_ls --threads 8`
- Versions tested: `[{"elite_cost_tolerance": 1.05, "elite_k": 5, "elite_min_diversity": 0.1, "smoothing_thres": 1}, {"elite_cost_tolerance": 1.08, "elite_k": 8, "elite_min_diversity": 0.15, "smoothing_thres": 2}, {"elite_cost_tolerance": 1.1, "elite_k": 10, "elite_min_diversity": 0.2, "smoothing_thres": 2}]`
- Best config: `{"elite_cost_tolerance": 1.05, "elite_k": 5, "elite_min_diversity": 0.1, "smoothing_thres": 1}`
- Final avg cost: `45.550468`
- Final avg diversity: `0.064078`
- Avg time: `12.729652` seconds
- Raw results: `cvrptw-faco/diversity_results/gfacs_rank_smoothing.json`, `cvrptw-faco/diversity_results/gfacs_rank_smoothing.csv`

## Overall Winner

| Method | Best config | Final avg cost | Final avg diversity | Avg time |
|---|---|---:|---:|---:|
| diverse_elite_archive | `{"elite_cost_tolerance": 1.03, "elite_k": 4, "elite_min_diversity": 0.1}` | 45.415535 | 0.065547 | 11.353728 |
| topk_source_archive | `{"top_k": 4}` | 45.353977 | 0.063081 | 6.753710 |
| exploration_schedule | `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "late_global_prob": 0.9}` | 45.702614 | 0.061380 | 20.153594 |
| multisource_sampling | `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "source_count": 2}` | 45.250885 | 0.182913 | 28.944272 |
| rank_topk_pheromone | `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "pheromone_top_k": 3}` | 45.967525 | 0.070906 | 18.098780 |
| gfacs_rank_smoothing | `{"elite_cost_tolerance": 1.05, "elite_k": 5, "elite_min_diversity": 0.1, "smoothing_thres": 1}` | 45.550468 | 0.064078 | 12.729652 |

**Best-cost version to keep:** `multisource_sampling` with config `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "source_count": 2}`.
Run it directly with `uv run .\test_best_diversity.py 1000 -i 10 --mini_H 10 --n_ants 50 -s 8 --smooth_mmas --cand_list_size 32 --extend_ls --threads 8`.
Final avg cost `45.250885`, final avg diversity `0.182913`, avg time `28.944272` seconds.
## best_diversity

- Command: `.\test_best_diversity.py 100 -i 10 --mini_H 10 --n_ants 100 -s 8 --smooth_mmas --cand_list_size 32 --extend_ls`
- Versions tested: `[{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "source_count": 2}]`
- Best config: `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "source_count": 2}`
- Final avg cost: `5.854128`
- Final avg diversity: `0.282898`
- Avg time: `5.986067` seconds
- Raw results: `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.json`, `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.csv`

## best_diversity

- Command: `.\test_best_diversity.py 100 -i 10 --mini_H 10 --n_ants 100 -s 8 --smooth_mmas --extend_ls`
- Versions tested: `[{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "source_count": 2}]`
- Best config: `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.15, "source_count": 2}`
- Final avg cost: `5.848192`
- Final avg diversity: `0.295654`
- Avg time: `5.034854` seconds
- Raw results: `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.json`, `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.csv`

## best_diversity

- Command: `.\test_best_diversity.py 100 -i 10 --mini_H 10 --n_ants 100 -s 8 --smooth_mmas --extend_ls`
- Versions tested: `[{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.25, "source_count": 2}]`
- Best config: `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.25, "source_count": 2}`
- Final avg cost: `5.833182`
- Final avg diversity: `0.325041`
- Avg time: `5.344974` seconds
- Raw results: `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.json`, `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.csv`

## best_diversity

- Command: `.\test_best_diversity.py 100 -i 10 --mini_H 10 --n_ants 100 -s 8 --smooth_mmas --extend_ls`
- Versions tested: `[{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.25, "source_count": 4}]`
- Best config: `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.25, "source_count": 4}`
- Final avg cost: `5.841739`
- Final avg diversity: `0.361723`
- Avg time: `14.606411` seconds
- Raw results: `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.json`, `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.csv`

## best_diversity

- Command: `.\test_best_diversity.py 100 -i 10 --mini_H 10 --n_ants 100 -s 8 --smooth_mmas --extend_ls`
- Versions tested: `[{"elite_cost_tolerance": 1.03, "elite_k": 8, "elite_min_diversity": 0.1, "source_count": 2}]`
- Best config: `{"elite_cost_tolerance": 1.03, "elite_k": 8, "elite_min_diversity": 0.1, "source_count": 2}`
- Final avg cost: `5.838957`
- Final avg diversity: `0.289012`
- Avg time: `5.140502` seconds
- Raw results: `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.json`, `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.csv`

## best_diversity

- Command: `.\test_best_diversity.py 1000 -i 10 --mini_H 10 --n_ants 50 -s 8 --smooth_mmas --cand_list_size 32 --extend_ls`
- Versions tested: `[{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.25, "source_count": 2}]`
- Best config: `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.25, "source_count": 2}`
- Final avg cost: `45.263092`
- Final avg diversity: `0.353263`
- Avg time: `30.390033` seconds
- Raw results: `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.json`, `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.csv`

## best_diversity

- Command: `.\test_best_diversity.py 100 -i 10 --mini_H 10 --n_ants 100 -s 8 --smooth_mmas --extend_ls`
- Versions tested: `[{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.25, "source_count": 2}]`
- Best config: `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.25, "source_count": 2}`
- Final avg cost: `5.833182`
- Final avg diversity: `0.325041`
- Avg time: `6.682705` seconds
- Raw results: `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.json`, `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.csv`

## best_diversity

- Command: `.\test_best_diversity.py 100 -i 10 --mini_H 10 --n_ants 100 -s 8 --smooth_mmas --extend_ls`
- Versions tested: `[{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.25, "source_count": 2}]`
- Best config: `{"elite_cost_tolerance": 1.05, "elite_k": 8, "elite_min_diversity": 0.25, "source_count": 2}`
- Final avg cost: `5.833182`
- Final avg diversity: `0.325041`
- Avg time: `5.608479` seconds
- Raw results: `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.json`, `C:/Users/PHAM ANH KHOI/Projects/NeuFACO-constraint/cvrptw-faco/diversity_results/best_diversity.csv`

