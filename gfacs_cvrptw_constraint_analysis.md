# GFACS-CVRPTW Constraint Handling Analysis

This note analyzes how `cvrptw-gfacs` handles capacity and time-window constraints in three places:

1. ACO construction masks.
2. PyVRP-backed local search operators.
3. Training and guided exploration.

The main implementation files are:

- `cvrptw-gfacs/aco.py`
- `cvrptw-gfacs/pyvrp_local_search.py`
- `cvrptw-gfacs/train.py`
- `cvrptw-gfacs/utils.py`

## Summary

GFACS handles CVRPTW constraints in two different layers:

- During neural-guided ACO construction, constraints are hard masks. The next-node probability is multiplied by visit, capacity, and time-window masks before sampling.
- During local search, the repository delegates all move feasibility and penalized evaluation to PyVRP. It registers every default PyVRP node and route operator, then uses `CostEvaluator(load_penalty=..., tw_penalty=...)` and `Solution.is_feasible()` to reject infeasible results unless explicitly allowed during perturbation.
- During training, the raw ACO samples are already mask-feasible. If guided exploration is enabled, local-search-improved paths are replayed through the same ACO masking logic to compute off-policy log probabilities; paths that fail the masks are filtered out.

One important limitation: local-search costs used by GFACS after PyVRP improvement are recomputed as pure travel distance in `ACO.gen_path_costs()`. Capacity and time-window violations are not part of the returned training objective; instead, infeasible local-search outputs are rejected or filtered.

## Data Representation

### Capacity

- `utils.gen_instance()` creates integer customer demands and normalizes them by the benchmark capacity: `demands / get_capacity(n, tam)`.
- The depot demand is prepended as zero.
- In VRPTW mode, all customer demands are set to zero, so the capacity mask becomes non-binding.
- The ACO class uses `CAPACITY = 1.0`, so normalized demand sums are compared against `1.0`.

Relevant code:

- `cvrptw-gfacs/utils.py:49` creates random demands.
- `cvrptw-gfacs/utils.py:51` normalizes demands, or zeros them for `--vrptw`.
- `cvrptw-gfacs/utils.py:52` prepends depot demand zero.
- `cvrptw-gfacs/aco.py:14` defines normalized capacity as `1.0`.

### Time Windows

- `utils.gen_instance()` creates node time windows `[early, late]`.
- The window generation tries to ensure each customer can be reached from the depot and can return to depot before the depot closes.
- The depot window is reset to `[0, n]`, making the depot close time large relative to normalized travel times.
- `service_time` defaults to `0.0`, but the ACO time-window update supports nonzero service time.

Relevant code:

- `cvrptw-gfacs/utils.py:63` computes the time-window upper bound using return-to-depot distance.
- `cvrptw-gfacs/utils.py:70` and `cvrptw-gfacs/utils.py:71` sample early/late times.
- `cvrptw-gfacs/utils.py:78` and `cvrptw-gfacs/utils.py:79` set the depot window.
- `cvrptw-gfacs/utils.py:81` stores windows as `[min_time, max_time]`.

### Neural Input

The GNN receives constraint information directly in node features:

```text
x = [demand, window_start, window_end]
```

Relevant code:

- `cvrptw-gfacs/utils.py:119` concatenates demand and windows into `pyg_data.x`.

## ACO Construction Masks

The constructive ACO path generator maintains three masks:

- `visit_mask`: prevents revisiting customers while allowing depot revisits.
- `capacity_mask`: prevents choosing customers whose demand exceeds remaining capacity.
- `time_window_mask`: prevents choosing customers that violate customer due times or prevent return to depot before depot close.

The final sampling distribution is:

```python
dist = (dist ** invtemp) * visit_mask * capacity_mask * time_window_mask
```

Then it is normalized and sampled with `torch.distributions.Categorical`.

Relevant code:

- `cvrptw-gfacs/aco.py:249` initializes `visit_mask`.
- `cvrptw-gfacs/aco.py:251` initializes `capacity_mask`.
- `cvrptw-gfacs/aco.py:252` initializes `time_window_mask`.
- `cvrptw-gfacs/aco.py:270` calls `pick_move()` with all masks.
- `cvrptw-gfacs/aco.py:320` multiplies the sampling distribution by all masks.

### Visit Mask

The visit mask is not a CVRPTW-specific constraint, but it interacts with depot revisits:

- The chosen node is masked out.
- Depot `0` is generally allowed to be revisited.
- Depot is temporarily disallowed immediately after visiting depot if there are still unvisited customers, preventing repeated `0 -> 0` moves before completion.

Relevant code:

- `cvrptw-gfacs/aco.py:327` defines `update_visit_mask()`.
- `cvrptw-gfacs/aco.py:328` masks chosen nodes.
- `cvrptw-gfacs/aco.py:329` re-allows depot revisits.
- `cvrptw-gfacs/aco.py:330` applies the depot-repeat exception.

### Capacity Mask

Capacity is enforced as a hard construction mask:

1. If the current node is the depot, used capacity is reset to zero.
2. The demand of the selected current node is added to used capacity.
3. Remaining capacity is `self.capacity - used_capacity`.
4. Any candidate whose demand is greater than remaining capacity is masked out.

This means ACO cannot select a customer that would exceed route capacity. If no customer fits, the depot remains the feasible action as long as visiting depot is allowed by the visit/time masks.

Relevant code:

- `cvrptw-gfacs/aco.py:333` defines `update_capacity_mask()`.
- `cvrptw-gfacs/aco.py:345` resets used capacity at depot.
- `cvrptw-gfacs/aco.py:346` adds selected-node demand.
- `cvrptw-gfacs/aco.py:348` computes remaining capacity.
- `cvrptw-gfacs/aco.py:351` masks over-capacity candidates.

### Time-Window Mask

Time windows are also enforced as a hard construction mask:

1. Current route time is advanced by travel time from previous node to current node.
2. If the vehicle arrives before the current node opens, it waits until `window_start`.
3. Service time is added.
4. If the current node is the depot, current time is reset to zero.
5. For every candidate, arrival time from the current node is computed.
6. A candidate is masked if arrival exceeds the candidate's `window_end`.
7. A candidate is also masked if serving it and returning to depot would exceed the depot's `window_end`.

The implementation checks customer due-time feasibility before applying candidate waiting/service time, and separately checks whether candidate service plus return-to-depot fits within the depot close time.

Relevant code:

- `cvrptw-gfacs/aco.py:355` defines `update_time_window_mask()`.
- `cvrptw-gfacs/aco.py:368` advances time by travel.
- `cvrptw-gfacs/aco.py:369` waits until the selected node opens.
- `cvrptw-gfacs/aco.py:370` adds service time.
- `cvrptw-gfacs/aco.py:373` resets time at depot.
- `cvrptw-gfacs/aco.py:378` computes candidate arrival time.
- `cvrptw-gfacs/aco.py:381` masks candidates whose arrival exceeds their close time.
- `cvrptw-gfacs/aco.py:383` computes candidate start time with waiting.
- `cvrptw-gfacs/aco.py:384` adds service plus return-to-depot travel.
- `cvrptw-gfacs/aco.py:388` masks candidates that cannot return to depot before depot close.

## Path Cost Semantics

`ACO.gen_path_costs()` computes only travel distance along the depot-separated route. It does not add load penalties, time-window penalties, waiting time, service time, or vehicle-count penalties.

This is valid only because construction masks and local-search feasibility checks are expected to keep solutions feasible. If infeasible paths reach this function, their objective still appears as pure distance.

Relevant code:

- `cvrptw-gfacs/aco.py:229` defines `gen_path_costs()`.
- `cvrptw-gfacs/aco.py:232` sums pairwise distances only.

## Local Search Overview

GFACS does not implement individual local-search moves itself. It delegates local search to PyVRP:

- Converts the GFACS instance to `pyvrp.ProblemData`.
- Converts each ACO path to a PyVRP `Solution`.
- Creates a PyVRP `LocalSearch` object.
- Registers every default PyVRP node operator from `NODE_OPERATORS`.
- Registers every default PyVRP route operator from `ROUTE_OPERATORS`.
- Runs `LocalSearch.search(solution, CostEvaluator(...))`.
- Returns the original path if no feasible improved solution is found, unless infeasible output is explicitly allowed.

Relevant code:

- `cvrptw-gfacs/pyvrp_local_search.py:7` imports PyVRP core classes.
- `cvrptw-gfacs/pyvrp_local_search.py:8` imports `NODE_OPERATORS`, `ROUTE_OPERATORS`, and `LocalSearch`.
- `cvrptw-gfacs/pyvrp_local_search.py:39` defines `make_search_operator()`.
- `cvrptw-gfacs/pyvrp_local_search.py:43` registers all PyVRP node operators.
- `cvrptw-gfacs/pyvrp_local_search.py:45` registers all PyVRP route operators.
- `cvrptw-gfacs/pyvrp_local_search.py:53` creates the `CostEvaluator`.
- `cvrptw-gfacs/pyvrp_local_search.py:54` runs PyVRP local search.
- `cvrptw-gfacs/pyvrp_local_search.py:56` checks `improved_solution.is_feasible()`.
- `cvrptw-gfacs/pyvrp_local_search.py:96` returns the original path if local search cannot find a feasible solution.

### Local-Search ProblemData Constraints

PyVRP receives both capacity and time-window information:

- Client delivery demand is set from normalized GFACS demand rescaled by capacity `600`.
- Each client receives `tw_early` and `tw_late`.
- The depot receives `tw_late`.
- Vehicle capacity is `600`.
- The vehicle count is `len(positions) - 1`, meaning up to one route per customer is available.
- The distance matrix is scaled by `10**4` and used as PyVRP distance.
- The duration matrix is set to zeros, so time-window evaluation in PyVRP depends on how PyVRP interprets distance versus duration. In this wrapper, GFACS provides distance but not explicit travel duration to PyVRP.

Relevant code:

- `cvrptw-gfacs/pyvrp_local_search.py:12` scales positions.
- `cvrptw-gfacs/pyvrp_local_search.py:13` scales windows.
- `cvrptw-gfacs/pyvrp_local_search.py:14` scales distances.
- `cvrptw-gfacs/pyvrp_local_search.py:16` sets PyVRP capacity to `600`.
- `cvrptw-gfacs/pyvrp_local_search.py:17` rescales normalized demands.
- `cvrptw-gfacs/pyvrp_local_search.py:21` creates clients with demand and time windows.
- `cvrptw-gfacs/pyvrp_local_search.py:24` creates the depot with a late time.
- `cvrptw-gfacs/pyvrp_local_search.py:26` creates a vehicle type with capacity `600`.
- `cvrptw-gfacs/pyvrp_local_search.py:28` passes the distance matrix.
- `cvrptw-gfacs/pyvrp_local_search.py:29` passes a zero duration matrix.

## Constraint Handling in Each Local-Search Operator Category

The repository does not define custom relocate, swap, 2-opt, or route-exchange code. Instead, every operator in PyVRP's `NODE_OPERATORS` and `ROUTE_OPERATORS` is added to the `LocalSearch` object.

Therefore, constraint handling is common across all local-search operators in this codebase:

| Operator category | What GFACS registers | Capacity handling | Time-window handling | Acceptance handling |
|---|---|---|---|---|
| Node operators | Every PyVRP default in `NODE_OPERATORS` | Handled inside PyVRP using vehicle capacity, client deliveries, and `load_penalty` | Handled inside PyVRP using client/depot time windows and `tw_penalty` | Search result must pass `Solution.is_feasible()` unless infeasible output is allowed |
| Route operators | Every PyVRP default in `ROUTE_OPERATORS` | Same global PyVRP cost/feasibility mechanism | Same global PyVRP cost/feasibility mechanism | Same feasibility gate |
| Heuristic-distance perturbation NLS | Same PyVRP operators, but with `self.heuristic_dist` as the distance matrix | Penalties are divided by 10 and infeasible output is allowed | Penalties are divided by 10 and infeasible output is allowed | Followed by a second feasibility-enforcing local-search pass |
| Final distance local search | Same PyVRP operators with actual distance matrix | Uses configured `load_penalty`; infeasible result rejected | Uses configured `tw_penalty`; infeasible result rejected | Only feasible improved paths are used |

Because PyVRP was not installed in the current environment, the exact class names contained in `NODE_OPERATORS` and `ROUTE_OPERATORS` could not be introspected locally. The repository itself only exposes these two operator groups, not the per-operator list.

### Local-Search Penalty Escalation

If PyVRP returns an infeasible solution, GFACS increases both penalties and retries recursively:

- `load_penalty *= 10`
- `tw_penalty *= 10`

This applies to every registered PyVRP local-search operator because penalties are provided globally through `CostEvaluator`.

Relevant code:

- `cvrptw-gfacs/pyvrp_local_search.py:56` checks feasibility.
- `cvrptw-gfacs/pyvrp_local_search.py:62` increases load penalty.
- `cvrptw-gfacs/pyvrp_local_search.py:63` increases time-window penalty.
- `cvrptw-gfacs/pyvrp_local_search.py:64` retries local search.

### Local Search During ACO/NLS

`ACO.local_search()` performs a two-stage pattern:

1. Run PyVRP local search with real distances and `allow_infeasible=False`.
2. Optionally run NLS perturbation loops:
   - Search using the learned heuristic-derived distance matrix with weaker penalties and `allow_infeasible=True`.
   - Then repair/improve using real distances with `allow_infeasible=False`.
   - Keep only paths whose pure distance cost improves the best path.

Relevant code:

- `cvrptw-gfacs/aco.py:416` defines `local_search()`.
- `cvrptw-gfacs/aco.py:419` calls `pyvrp_batched_local_search`.
- `cvrptw-gfacs/aco.py:430` reads cost-evaluator parameters.
- `cvrptw-gfacs/aco.py:435` weakens load penalty for heuristic perturbation.
- `cvrptw-gfacs/aco.py:437` weakens time-window penalty for heuristic perturbation.
- `cvrptw-gfacs/aco.py:439` runs feasible real-distance local search.
- `cvrptw-gfacs/aco.py:447` runs heuristic-distance perturbation with infeasible output allowed.
- `cvrptw-gfacs/aco.py:453` repairs with real-distance local search.
- `cvrptw-gfacs/aco.py:460` accepts only pure-distance improvements.

## Training Constraint Handling

### Normal ACO Training Samples

Training calls `aco.sample()`, which uses the construction masks described above. Therefore raw training paths should satisfy:

- no repeated customers,
- capacity feasibility,
- customer time-window feasibility,
- ability to return to depot before depot close.

The loss then uses pure route distance as the energy/cost.

Relevant code:

- `cvrptw-gfacs/train.py:74` constructs `ACO` with demands and windows.
- `cvrptw-gfacs/train.py:86` samples masked ACO paths.
- `cvrptw-gfacs/train.py:87` uses pure-distance costs from `aco.sample()`.
- `cvrptw-gfacs/train.py:101` uses sampled-path log probabilities in trajectory-balance loss.
- `cvrptw-gfacs/train.py:102` uses weighted advantages from costs.

### Guided Exploration with Local Search

If guided exploration is enabled:

1. Local search improves the sampled paths.
2. Costs for the improved paths are recomputed with `ACO.gen_path_costs()`, again pure distance.
3. The improved paths are replayed through `aco.gen_path(require_prob=True, paths=paths_ls)` to compute off-policy log probabilities under the current masked ACO policy.
4. If a local-search path violates the ACO capacity/time-window masks during replay, that ant is removed from the LS loss.

Relevant code:

- `cvrptw-gfacs/train.py:90` checks `guided_exploration`.
- `cvrptw-gfacs/train.py:91` runs local search.
- `cvrptw-gfacs/train.py:92` recomputes local-search path costs as pure distance.
- `cvrptw-gfacs/train.py:95` blends LS advantages and original advantages.
- `cvrptw-gfacs/train.py:110` replays local-search paths through masked `gen_path()`.
- `cvrptw-gfacs/train.py:115` filters infeasible replay paths.
- `cvrptw-gfacs/train.py:121` computes LS forward flow.
- `cvrptw-gfacs/train.py:122` computes LS backward flow.

The replay filtering logic lives in `ACO.gen_path()`:

- `cvrptw-gfacs/aco.py:281` detects infeasible guided paths by zero mask availability.
- `cvrptw-gfacs/aco.py:286` builds `is_feasible` from capacity and time-window mask availability.
- `cvrptw-gfacs/aco.py:296` filters paths/log-probs.
- `cvrptw-gfacs/aco.py:311` returns `feasible_idx`.

Note: the code uses Python boolean `and` and set-like `|` over tensors in this filtering block. This may deserve review if guided exploration filtering behaves unexpectedly.

### Training Hyperparameters Related to Constraints

The CLI exposes local-search penalties:

- `--load_penalty`, default `20`
- `--tw_penalty`, default `20`

These are passed to PyVRP local search through `local_search_params`, not to ACO construction masks. ACO masks remain hard constraints regardless of these values.

Relevant code:

- `cvrptw-gfacs/train.py:421` defines `--load_penalty`.
- `cvrptw-gfacs/train.py:422` defines `--tw_penalty`.
- `cvrptw-gfacs/train.py:448` passes both into `cost_evaluator_params`.

## Testing / Inference Constraint Handling

Testing follows the same ACO and local-search mechanisms:

- ACO uses demand/window masks during construction.
- PyVRP local search receives demand/window data and uses penalties/feasibility checks.
- Result quality is reported as pure travel distance.

Relevant code:

- `cvrptw-gfacs/test.py:24` constructs `ACO` with demands, windows, positions, and local-search params.
- `cvrptw-gfacs/test.py:37` enables local search.
- `cvrptw-gfacs/test.py:181` passes `load_penalty` and `tw_penalty` to local search.

## Key Takeaways

- Capacity in ACO is a hard mask based on normalized demand and remaining route capacity.
- Time windows in ACO are hard masks based on arrival due-time and return-to-depot feasibility.
- The neural model sees demand and time-window data as node features, but it does not directly enforce constraints; masks enforce them during sampling.
- Local search is entirely PyVRP-backed; the repository applies all default PyVRP node and route operators.
- Local-search constraint handling is global through PyVRP `ProblemData`, `CostEvaluator`, and `Solution.is_feasible()`, not per custom operator in this repo.
- Training uses pure distance costs, relying on masks/local-search feasibility rather than penalty-augmented objective values.
- Guided exploration replays local-search paths through the same ACO mask logic and drops paths that cannot be replayed feasibly.

