# `infer_instance` FACO-CVRPTW Call Flow

This document traces how `infer_instance()` in `cvrptw-faco/faco_test.py` drives the Python wrapper, pybind11 layer, and low-level C++ functions in `cvrptw-faco/src/mfaco_train.cpp`.

The focus is function order plus inputs/outputs. Implementation details are intentionally summarized.

## Entry Point

### `infer_instance(...)`

Location: `cvrptw-faco/faco_test.py:75`

Inputs:

- `demands`: customer demand tensor/array, depot included.
- `positions`: node coordinates, depot at index `0`.
- `windows`: time windows with shape `(n, 2)`, `[ready_time, due_time]`.
- `n_ants`: number of ants sampled per iteration.
- `n_iter`: number of FACO iterations.
- `seed`: random seed.
- `cand_list_size`: nearest-neighbor candidate list size.
- `backup_list_size`: backup nearest-neighbor list size.
- `min_new_edges`: stopping threshold for new/cross-route edges during sampling.
- `decay`: pheromone decay parameter.
- `alpha`: pheromone exponent.
- `p_best`: MMAS trail-limit parameter.
- `use_local_search`: whether to run local search after sampling.
- `disable_heuristic`: whether to disable heuristic values.
- `extend_ls`: local-search option.
- `smooth_mmas`: whether to use smooth MMAS trail updates.
- `fixed_steps`: fixed sampling steps; `0` means use adaptive stopping.
- `nls`: neural/local-search option flag.
- `T_nls`: NLS iteration count.

Outputs:

- `results`: tensor of best-so-far costs per iteration, shape `(n_iter,)`.
- `diversities`: tensor of route diversity per iteration, shape `(n_iter,)`.
- `elapsed`: wall-clock runtime in seconds.

High-level loop:

```text
construct MFACO_CVRPTW
seed RNG
for each iteration:
    sample n_ants routes from C++
    choose lowest-cost route in Python
    update C++ pheromone/source route with that route
    record best-so-far and diversity
return results, diversities, elapsed
```

## Phase 1: Solver Construction

### 1. `MFACO_CVRPTW.__init__(...)`

Location: `cvrptw-faco/faco.py:727`

Called by:

- `infer_instance()` when it creates `solver = MFACO_CVRPTW(...)`.

Inputs:

- `coords`: Python/Torch/NumPy coordinates.
- `demand`: Python/Torch/NumPy demands.
- `windows`: Python/Torch/NumPy time windows.
- `capacity`: vehicle capacity, passed as `1.0` in `faco_test.py`.
- Solver hyperparameters from `infer_instance()`.

Outputs/state:

- Creates `self._cpp`, a pybind11 `faco_opt.MFACO_CVRPTW` object.
- Creates Torch copies of C++ pheromone, heuristic, and nearest-neighbor arrays:
  - `_pheromone_sparse`
  - `_h_sparse_torch`
  - `_nn_torch`

Notes:

- Converts inputs to contiguous `float32` NumPy arrays.
- Forces `demand_np[0] = 0.0` for the depot.

### 2. `PyMFACO_CVRPTW::PyMFACO_CVRPTW(...)`

Location: `cvrptw-faco/src/binding.cpp:550`

Called by:

- `faco_opt.MFACO_CVRPTW(...)` from Python.

Inputs:

- `coords`: NumPy array, shape `(n, 2)`, `float32`.
- `demand`: NumPy array, shape `(n,)`, `float32`.
- `windows`: NumPy array, shape `(n, 2)`, `float32`.
- `capacity`: float.
- FACO parameters: `n_ants`, `cand_list_size`, `backup_list_size`, `min_new_edges`, `decay`, `alpha`, `p_best`, `use_local_search`, `disable_heuristic`, `extend_ls`, `smooth_mmas`, `fixed_steps`, `nls`, `T_nls`.

Outputs/state:

- Calls the base `PyMFACO_CVRP` constructor.
- Validates `windows` shape.
- Calls `solver->set_time_windows(...)` on the underlying C++ `MFACO_CVRP` solver.

### 3. `PyMFACO_CVRP::PyMFACO_CVRP(...)`

Location: `cvrptw-faco/src/binding.cpp:358`

Called by:

- `PyMFACO_CVRPTW` base constructor.

Inputs:

- `coords`: NumPy array, shape `(n, 2)`.
- `demand`: NumPy array, shape `(n,)`.
- `capacity`: float.
- FACO parameters.

Outputs/state:

- Creates `solver = std::make_unique<MFACO_CVRP>(...)`.

### 4. `MFACO_CVRP::MFACO_CVRP(...)`

Location: `cvrptw-faco/src/mfaco_train.cpp:1244`

Called by:

- `PyMFACO_CVRP` constructor.

Inputs:

- `coords_ptr`: raw pointer to `(n, 2)` coordinate data.
- `demand_ptr`: raw pointer to `(n,)` demand data.
- `n_`: number of nodes including depot.
- `capacity_`: vehicle capacity.
- `n_ants_`: number of ants.
- FACO parameters.

Outputs/state:

- Initializes the C++ solver object.
- Sets problem dimensions:
  - `n`: nodes including depot.
  - `m = n - 1`: customers.
  - `k`: candidate-list size.
  - `bl`: backup-list size.
- Copies coordinates and demands into C++ vectors.
- Builds nearest-neighbor lists, heuristic arrays, depot distances, initial solution, trail limits, and pheromone.

Constructor-internal C++ calls:

```text
MFACO_CVRP::build_nn_lists()
MFACO_CVRP::build_heuristic()
MFACO_CVRP::build_d0()
MFACO_CVRP::build_initial_solution()
MFACO_CVRP::calc_trail_limits_cl(...) or calc_trail_limits_smooth(...)
Xoshiro128Plus::seed(42)
```

### 5. `MFACO_CVRP::build_nn_lists()`

Location: `cvrptw-faco/src/mfaco_train.cpp:1416`

Inputs:

- Uses solver state:
  - `coords`
  - `n`
  - `k`
  - `bl`

Outputs/state:

- Fills:
  - `nn_list`: sparse nearest-neighbor candidate list, shape `(n, k)`.
  - `backup_list`: backup nearest-neighbor list, shape `(n, bl)`.

Important lower-level dependency:

- Uses `KDTree` from `cvrptw-faco/src/kd_tree.h`.

### 6. `KDTree` functions used by `build_nn_lists()`

Location: `cvrptw-faco/src/kd_tree.h`

Main calls:

- `KDTree::KDTree(points, round_distances)`
  - Input: vector of `Vec2d` points and distance-rounding flag.
  - Output/state: KD-tree object.

- `KDTree::nn_bottom_up(point_idx)`
  - Input: point index.
  - Output: nearest-neighbor point index.

- `KDTree::delete_point(point_idx)` / related removal logic if used by neighbor extraction.
  - Input: point index to exclude from future nearest-neighbor calls.
  - Output/state: marks/removes that point inside the temporary KD-tree copy.

### 7. `MFACO_CVRP::build_heuristic()`

Location: `cvrptw-faco/src/mfaco_train.cpp:1489`

Inputs:

- Uses solver state:
  - `nn_list`
  - `coords`
  - `disable_heuristic`

Outputs/state:

- Fills `heuristic_sparse`, shape `(n, k)`.
- If heuristic is disabled, fills it with `1.0`.

Lower-level calls:

- `MFACO_CVRP::dist(u, v)`
  - Input: node indices `u`, `v`.
  - Output: Euclidean distance.

### 8. `MFACO_CVRP::build_d0()`

Location: `cvrptw-faco/src/mfaco_train.cpp:1508`

Inputs:

- Uses `coords` and `n`.

Outputs/state:

- Fills `d0[v] = dist(0, v)` for every node.

### 9. `MFACO_CVRP::build_initial_solution()`

Location: `cvrptw-faco/src/mfaco_train.cpp:1517`

Inputs:

- Uses solver state:
  - `nn_list`
  - `demand_int`
  - `capacity_int`
  - `coords`

Outputs/state:

- Builds an initial depot-separated CVRP route.
- Sets:
  - `source_route`
  - `best_route`
  - `source_cost`
  - `best_cost`
  - initial `tau_min`
  - initial `tau_max`

Notes:

- This initial route is capacity-aware.
- For CVRPTW, it is built before `set_time_windows()`, so time windows are not used in this initial construction.

Lower-level calls:

- `MFACO_CVRP::dist(u, v)`
  - Input: two node indices.
  - Output: Euclidean distance.

### 10. `MFACO_CVRP::calc_trail_limits_cl(solution_cost)`

Location: `cvrptw-faco/src/mfaco_train.cpp:1631`

Inputs:

- `solution_cost`: current solution cost.

Outputs:

- Pair `{tau_min, tau_max}` for classic MMAS-style bounds.

### 11. `MFACO_CVRP::calc_trail_limits_smooth(solution_cost)`

Location: `cvrptw-faco/src/mfaco_train.cpp:1642`

Inputs:

- `solution_cost`: current solution cost; currently not central to the smooth formula.

Outputs:

- Pair `{tau_min, tau_max}` for smooth MMAS-style bounds.

### 12. `MFACO_CVRP::set_time_windows(windows_ptr)`

Location: `cvrptw-faco/src/mfaco_train.cpp:1315`

Called by:

- `PyMFACO_CVRPTW::PyMFACO_CVRPTW(...)` after the base CVRP solver is already constructed.

Inputs:

- `windows_ptr`: raw pointer to `(n, 2)` time-window data.

Outputs/state:

- Sets `has_time_windows = true`.
- Fills:
  - `ready_time`
  - `due_time`

## Phase 2: Seeding

### 13. `MFACO_CVRPTW.seed_rng(seed)` / `PyMFACO_CVRP::seed_rng(seed)`

Locations:

- Python call: `cvrptw-faco/faco_test.py:122`
- Python wrapper: `cvrptw-faco/faco.py:662`
- Binding method: `cvrptw-faco/src/binding.cpp:426`
- C++ method: `cvrptw-faco/src/mfaco_train.cpp:1313`

Inputs:

- `seed`: integer seed.

Outputs/state:

- Seeds C++ solver RNG.

Lower-level call:

- `Xoshiro128Plus::seed(seed)`
  - Input: unsigned 64-bit seed.
  - Output/state: initialized RNG state.

## Phase 3: Iteration Loop

Location: `cvrptw-faco/faco_test.py:130`

For each iteration `t` in `range(n_iter)`, `infer_instance()` calls:

```text
solver.sample(prior=None)
choose best route in Python
solver.update_pheromone(best_route, best_cost)
route_diversity(routes)
```

## Phase 3A: Sampling Routes

### 14. `MFACO_CVRP.sample(...)` Python method inherited by `MFACO_CVRPTW`

Location: `cvrptw-faco/faco.py:666`

Called by:

- `costs, routes, *_ = solver.sample(prior=None)` in `infer_instance()`.

Inputs:

- `invtemp`: API compatibility parameter; unused by CVRP path here.
- `require_prob`: default `False` in `infer_instance()`.
- `prior`: `None` in `infer_instance()`.
- `parallel_traced`: default `False`.
- `return_decoded`: default `False`.

Outputs:

Tuple:

```text
(costs, routes, decoded, logps, traces, costs_raw, routes_raw, new_edges_count, survival)
```

In `infer_instance()`, only `costs` and `routes` are used.

### 15. `PyMFACO_CVRP::sample(require_prob, prior, parallel_traced, return_decoded)`

Location: `cvrptw-faco/src/binding.cpp:429`

Inputs:

- `require_prob`: bool.
- `prior`: `None` or NumPy array with shape `(n, k)`.
- `parallel_traced`: bool.
- `return_decoded`: bool.

Outputs:

- Python tuple containing:
  - `costs`: NumPy float array, shape `(n_ants,)`.
  - `routes`: Python list of NumPy int arrays, one depot-separated route per ant.
  - `decoded`: either routes or `None`, depending on `return_decoded`.
  - `logps`: NumPy float array, shape `(n_ants,)`.
  - `traces`: `MFACOTrace` object or `None`.
  - `costs_raw`: raw-cost array/list depending on traced mode.
  - `routes_raw`: raw route list.
  - `new_edges_count`: NumPy int array, shape `(n_ants,)`.
  - `survival`: NumPy float array, shape `(n_ants,)`.

C++ call:

```text
solver->sample(require_prob, prior_ptr, result, parallel_traced)
```

### 16. `MFACO_CVRP::sample(require_prob, prior_ptr, result, parallel_traced)`

Location: `cvrptw-faco/src/mfaco_train.cpp:1729`

Inputs:

- `require_prob`: bool; `False` from `infer_instance()`.
- `prior_ptr`: raw pointer to prior matrix or `nullptr`; `nullptr` from `infer_instance()`.
- `result`: mutable `SampleResult` output container.
- `parallel_traced`: bool; `False` from default Python call.

Outputs/state:

- Fills `result` with sampled ant data:
  - `result.costs`
  - `result.routes`
  - `result.decoded_routes`
  - `result.new_edges_count`
  - `result.edge_survival`
  - optionally `result.costs_raw`, `result.routes_raw`, `result.logps`, `result.traces`

Main internal calls in the `infer_instance()` default path:

```text
MFACO_CVRP::compute_probmat(prior_ptr, probmat)
Xoshiro128Plus::next_uint(...) for start nodes
Xoshiro128Plus::next_u32(...) for per-ant seeds
MFACO_CVRP::sample_ant_direct(...)
MFACO_CVRP::dist(...) via route-cost calculation
```

Because `infer_instance()` calls `sample(prior=None)` with default `require_prob=False`, the non-traced fast branch is used.

### 17. `MFACO_CVRP::compute_probmat(prior_ptr, probmat)`

Location: `cvrptw-faco/src/mfaco_train.cpp:1651`

Inputs:

- `prior_ptr`: pointer to `(n, k)` prior logits or `nullptr`.
- `probmat`: mutable vector output.

Outputs:

- Fills `probmat`, shape `(n * k)`, with positive transition weights derived from:
  - pheromone values
  - heuristic values, unless disabled
  - optional prior logits

### 18. `Xoshiro128Plus::next_uint(max_exclusive)`

Location: `cvrptw-faco/src/mfaco_train.h:63`

Inputs:

- `max_exclusive`: upper bound.

Outputs:

- Uniform random integer in `[0, max_exclusive)`.

Used for:

- Choosing ant start nodes.

### 19. `Xoshiro128Plus::next_u32()`

Location: `cvrptw-faco/src/mfaco_train.h:52`

Inputs:

- None beyond RNG state.

Outputs:

- Random `uint32_t`.

Used for:

- Creating per-ant seeds.

### 20. `MFACO_CVRP::sample_ant_direct(...)`

Location: `cvrptw-faco/src/mfaco_train.cpp:3544`

Called by:

- `MFACO_CVRP::sample(...)` in the default non-traced branch.

Inputs:

- `probmat`: pointer to transition probability/weight matrix, shape `(n, k)`.
- `start_node`: customer index chosen for the ant.
- `route_out`: mutable output route vector.
- `new_edges_out`: mutable output integer.
- `checklist`: mutable working list used by local search.
- `rng`: ant-local RNG.
- `prior`: prior pointer or `nullptr`.

Outputs:

- Return value: final route cost as `float`.
- Mutates:
  - `route_out`: depot-separated route, eventually using repeated `0` depots.
  - `new_edges_out`: number of new/cross edges counted by the sampler.
  - `checklist`: nodes/edges touched for local search.

Main internal calls and helpers:

```text
MFACO_CVRP::initial_routes_from_perm(source_route)
MFACO_CVRP::select_next_node(...) or equivalent inline candidate selection logic
Xoshiro128Plus::next_float()
MFACO_CVRP::enforce_time_windows(route_out)
MFACO_CVRP::inter_route_ls_optimized(...) if local search is enabled
MFACO_CVRP::intra_route_ls(...) if local search is enabled
MFACO_CVRP::route_time_feasible(route_out) after local search if time windows are active
MFACO_CVRP::dist(...) for final cost
```

Notes:

- The function internally represents route starts/ends with artificial depot nodes like `n + route_id`, then flattens back to repeated depot `0` before returning.
- If local search makes a time-windowed route infeasible, the route is rolled back to the pre-local-search route.

### 21. `MFACO_CVRP::initial_routes_from_perm(solution)`

Location: `cvrptw-faco/src/mfaco_train.cpp:2029`

Inputs:

- `solution`: depot-separated route vector, usually `source_route`.

Outputs:

- Vector of individual routes, each shaped like `[0, customer..., 0]`.

Used by:

- `sample_ant_direct(...)` to initialize its linked-list route representation from the current source solution.

### 22. `MFACO_CVRP::select_next_node(...)`

Location: `cvrptw-faco/src/mfaco_train.cpp:3987`

Inputs:

- `curr`: current node, possibly artificial depot node.
- `curr_route`: current route index.
- `probmat_row`: pointer to probability row for current lookup node.
- `visited`: customer visited flags.
- `node_route`: customer-to-route mapping.
- `route_loads`: current loads per route.
- `rng`: random number generator.
- `out_pick_j`: mutable output candidate index.
- `out_valid_mask`: mutable output candidate bit mask.

Outputs:

- Tuple:
  - `chosen`: selected next node.
  - `is_stoch`: whether selection was stochastic.
  - `log_prob`: log probability of selected move if stochastic.
- Mutates:
  - `out_pick_j`
  - `out_valid_mask`

Used by:

- The traced sampler directly.
- The non-traced sampler has equivalent candidate-selection logic in its route-construction loop.

### 23. `Xoshiro128Plus::next_float()`

Location: `cvrptw-faco/src/mfaco_train.h:80`

Inputs:

- None beyond RNG state.

Outputs:

- Uniform random float in `[0, 1)`.

Used for:

- Stochastic candidate selection.

### 24. `MFACO_CVRP::enforce_time_windows(route)`

Location: `cvrptw-faco/src/mfaco_train.cpp:1363`

Inputs:

- `route`: mutable depot-separated route vector.

Outputs/state:

- Mutates `route` by rebuilding it and inserting depot `0` where needed.
- If time windows are disabled, returns without changes.

Internal calls:

- `MFACO_CVRP::can_append_tw(prev, node, route_time)`
- `MFACO_CVRP::dist(u, v)`

### 25. `MFACO_CVRP::can_append_tw(prev, node, route_time)`

Location: `cvrptw-faco/src/mfaco_train.cpp:1328`

Inputs:

- `prev`: previous node index.
- `node`: candidate next node index.
- `route_time`: current accumulated route time.

Outputs:

- `true` if appending `node` after `prev` respects node due time and can still return to depot by depot due time.
- `false` otherwise.

### 26. `MFACO_CVRP::inter_route_ls_optimized(route, positions, checklist, in_checklist)`

Location: `cvrptw-faco/src/mfaco_train.cpp:2307`

Inputs:

- `route`: mutable depot-separated route.
- `positions`: mutable node-position vector.
- `checklist`: nodes/edges to consider.
- `in_checklist`: flags for checklist membership.

Outputs:

- Return value: improvement delta/cost value as `float`.
- Mutates `route` if local-search moves are accepted.

Used only when:

- `use_local_search == true`
- `checklist` is non-empty.

### 27. `MFACO_CVRP::intra_route_ls(route, checklist)`

Location: `cvrptw-faco/src/mfaco_train.cpp:2116`

Inputs:

- `route`: mutable depot-separated route.
- `checklist`: nodes/edges to consider.

Outputs:

- Return value: improvement delta/cost value as `float`.
- Mutates `route` if intra-route moves are accepted.

Used only when:

- `use_local_search == true`
- `checklist` is non-empty.

### 28. `MFACO_CVRP::route_time_feasible(route)`

Location: `cvrptw-faco/src/mfaco_train.cpp:1339`

Inputs:

- `route`: depot-separated route vector.

Outputs:

- `true` if route timing respects time windows.
- `false` otherwise.

Used after local search:

- If time windows are active and local search makes the route infeasible, the sampler restores the route saved before local search.

### 29. `MFACO_CVRP::dist(u, v)`

Location: `cvrptw-faco/src/mfaco_train.cpp:1407`

Inputs:

- `u`: node index.
- `v`: node index.

Outputs:

- Euclidean distance between nodes `u` and `v`.

Used throughout:

- heuristic construction
- initial solution construction
- route construction
- time-window checks
- cost computation
- local search

### 30. Optional traced branch: `MFACO_CVRP::sample_ant_direct_traced(...)`

Location: `cvrptw-faco/src/mfaco_train.cpp:4133`

Not used by default in `infer_instance()` because `require_prob=False`.

Inputs:

- `probmat`: transition matrix pointer.
- `start_node`: customer start node.
- `route_out`: mutable final route output.
- `route_raw_out`: mutable raw route output before final local-search result.
- `cost_raw_out`: mutable raw-cost output.
- `new_edges_out`: mutable new-edge count output.
- `checklist`: mutable local-search checklist.
- `trace`: mutable `MFACOTrace` output.
- `rng`: RNG.
- `logp_sum`: mutable log-probability sum output.
- `surv_out`: mutable edge-survival output.
- `prior`: prior pointer or `nullptr`.

Outputs:

- Return value: final route cost.
- Mutates all output references above.

Used when:

- Python calls `sample(require_prob=True, ...)`.

## Phase 3B: Python Best-Ant Selection

After `solver.sample(...)`, `infer_instance()` does this in Python:

```python
costs_np = np.asarray(costs, dtype=np.float32)
best_idx = int(np.argmin(costs_np))
best_cost = float(costs_np[best_idx])
best_route = np.asarray(routes[best_idx], dtype=np.int32)
```

Inputs:

- `costs`: sampled C++ costs from all ants.
- `routes`: sampled C++ routes from all ants.

Outputs:

- `best_idx`: index of best ant this iteration.
- `best_cost`: lowest sampled route cost this iteration.
- `best_route`: depot-separated route for the best ant.

Then:

```python
if best_cost < best_so_far:
    best_so_far = best_cost
```

Output/state:

- Updates Python-side `best_so_far` scalar.

## Phase 3C: Updating Pheromone and Source Route

### 31. `MFACO_CVRP.update_pheromone(best_route, best_cost)` Python method

Location: `cvrptw-faco/faco.py:695`

Called by:

- `infer_instance()` after selecting the best route of the current iteration.

Inputs:

- `best_route`: NumPy int route array, depot-separated with repeated `0`.
- `best_cost`: float cost of that route.

Outputs/state:

- Calls C++ `update_pheromone_from_route(...)`.
- If Torch sync is enabled, calls `sync_pheromone_to_torch()` afterward.

### 32. `PyMFACO_CVRP::update_pheromone_from_route(best_route, best_cost)`

Location: `cvrptw-faco/src/binding.cpp:526`

Inputs:

- `best_route`: NumPy int array, 1D.
- `best_cost`: float.

Outputs/state:

- Converts route to `std::vector<int32_t>`.
- Calls `solver->update_pheromone(route_vec, best_cost)`.

### 33. `MFACO_CVRP::update_pheromone(best_route_in, new_best_cost)`

Location: `cvrptw-faco/src/mfaco_train.cpp:1948`

Inputs:

- `best_route_in`: depot-separated route vector.
- `new_best_cost`: route cost.

Outputs/state:

- If `new_best_cost < best_cost`, updates:
  - `best_cost`
  - `best_route`
- Recomputes trail bounds:
  - `tau_min`
  - `tau_max`
- Evaporates and reinforces pheromone on edges belonging to `best_route_in`.
- Updates source solution unconditionally:
  - `source_route = best_route_in`
  - `source_cost = new_best_cost`

Internal calls:

```text
MFACO_CVRP::calc_trail_limits_cl(best_cost) or calc_trail_limits_smooth(best_cost)
```

Notes:

- This function trusts the route passed from Python.
- It does not run a final CVRPTW feasibility check before assigning `best_route` or `source_route`.

### 34. `MFACO_CVRPTW.sync_pheromone_to_torch()`

Location: `cvrptw-faco/faco.py:796`

Inputs:

- No direct inputs.
- Reads C++ `pheromone_sparse_np` property.

Outputs/state:

- Copies C++ pheromone values into Python/Torch tensor `_pheromone_sparse`.

Used when:

- `enable_torch_sync=True`, which is default in `MFACO_CVRPTW.__init__()`.

## Phase 3D: Route Diversity

### 35. `route_diversity(routes)`

Location: `cvrptw-faco/faco_test.py:54`

Inputs:

- `routes`: list of sampled depot-separated routes from the current iteration.

Outputs:

- Float diversity score for the current sampled route set.

Used by:

- `infer_instance()` to fill `diversities[t]`.

## End of Iteration

At the end of each loop iteration:

```python
results[t] = best_so_far
diversities[t] = route_diversity(routes)
```

Inputs:

- `best_so_far`: best cost seen over all previous iterations including current.
- `routes`: current iteration sampled routes.

Outputs/state:

- Writes one element of `results`.
- Writes one element of `diversities`.

## Final Return

After all iterations:

```python
elapsed = time.time() - start
return results, diversities, elapsed
```

Outputs:

- `results`: best-so-far curve.
- `diversities`: route diversity curve.
- `elapsed`: total runtime.

## Full Default Call Sequence

This is the default path used by `infer_instance()` with `solver.sample(prior=None)` and default `require_prob=False`.

```text
infer_instance(...)
  MFACO_CVRPTW.__init__(...)
    faco_opt.MFACO_CVRPTW(...)
      PyMFACO_CVRPTW::PyMFACO_CVRPTW(...)
        PyMFACO_CVRP::PyMFACO_CVRP(...)
          MFACO_CVRP::MFACO_CVRP(...)
            MFACO_CVRP::build_nn_lists()
              KDTree::KDTree(...)
              KDTree::nn_bottom_up(...)
            MFACO_CVRP::build_heuristic()
              MFACO_CVRP::dist(...)
            MFACO_CVRP::build_d0()
              MFACO_CVRP::dist(...)
            MFACO_CVRP::build_initial_solution()
              MFACO_CVRP::dist(...)
            MFACO_CVRP::calc_trail_limits_cl(...) or calc_trail_limits_smooth(...)
            Xoshiro128Plus::seed(42)
        MFACO_CVRP::set_time_windows(...)
  solver.seed_rng(seed)
    PyMFACO_CVRP::seed_rng(seed)
      MFACO_CVRP::seed_rng(seed)
        Xoshiro128Plus::seed(seed)
  repeat n_iter times:
    solver.sample(prior=None)
      PyMFACO_CVRP::sample(False, None, False, False)
        MFACO_CVRP::sample(False, nullptr, result, False)
          MFACO_CVRP::compute_probmat(nullptr, probmat)
          Xoshiro128Plus::next_uint(...)
          Xoshiro128Plus::next_u32(...)
          parallel for each ant:
            Xoshiro128Plus::seed(ant_seed)
            MFACO_CVRP::sample_ant_direct(...)
              MFACO_CVRP::initial_routes_from_perm(source_route)
              candidate selection / route relocation
                Xoshiro128Plus::next_float()
                MFACO_CVRP::dist(...)
              MFACO_CVRP::enforce_time_windows(route_out)
                MFACO_CVRP::can_append_tw(...)
                  MFACO_CVRP::dist(...)
              if local search enabled:
                MFACO_CVRP::inter_route_ls_optimized(...)
                MFACO_CVRP::intra_route_ls(...)
                MFACO_CVRP::route_time_feasible(...)
              MFACO_CVRP::dist(...) for final cost
          route_cost_euclid lambda
            MFACO_CVRP::dist(...)
    Python np.argmin(costs)
    solver.update_pheromone(best_route, best_cost)
      PyMFACO_CVRP::update_pheromone_from_route(...)
        MFACO_CVRP::update_pheromone(...)
          MFACO_CVRP::calc_trail_limits_cl(...) or calc_trail_limits_smooth(...)
      MFACO_CVRPTW.sync_pheromone_to_torch()
    route_diversity(routes)
return results, diversities, elapsed
```

## Conditional Functions Not Used in Default `infer_instance()` Path

The following functions are part of the same FACO-CVRPTW backend but are not reached by the default `infer_instance()` call unless options change.

### `MFACO_CVRP::sample_ant_direct_traced(...)`

Location: `cvrptw-faco/src/mfaco_train.cpp:4133`

Used when:

- `sample(require_prob=True, ...)` is called.

Input/output:

- Same broad role as `sample_ant_direct`, but also outputs traces, raw route, raw cost, log probability, and survival metrics.

### `MFACOTraceBatch` / `PyMFACOTrace`

Locations:

- C++ structure: `cvrptw-faco/src/mfaco_train.h:125`
- Python wrapper: `cvrptw-faco/src/binding.cpp:34`

Used when:

- Traced sampling is requested.

Input/output:

- Stores batched ant decision traces and exposes them to Python as NumPy views or Python trace dictionaries.

### `MFACO_CVRP::reset_timings()` and `get_timings()`

Locations:

- C++ reset: `cvrptw-faco/src/mfaco_train.cpp:1400`
- Binding/Python access: `cvrptw-faco/src/binding.cpp:540`

Used when:

- Python explicitly requests timing reset or timing metrics.

Input/output:

- `reset_timings()` input: none; output/state: resets timing counters.
- `get_timings()` input: none; output: dictionary with `time_ant`, `time_ls`, and `time_split`.
