# FACO Implementation Flow for `cvrptw-faco`

This document explains how the FACO code in `cvrptw-faco` is wired, how a CVRP solution is sampled and improved, and where to modify the code if you want to extend FACO toward constrained VRP variants such as CVRPTW or VRPTW.

## High-Level Architecture

`cvrptw-faco` is a Python wrapper around a C++/OpenMP backend.

```text
Python caller
  |
  v
cvrptw-faco/faco.py
  |
  v
pybind11 bridge: cvrptw-faco/src/binding.cpp
  |
  v
C++ solver: cvrptw-faco/src/mfaco_train.cpp + mfaco_train.h
  |
  v
KD-tree nearest-neighbor support: cvrptw-faco/src/kd_tree.h
```

There are two solver families in the codebase:

- `MFACO_TSP`: TSP-oriented FACO wrapper and C++ solver.
- `MFACO_CVRP`: CVRP-oriented FACO wrapper and C++ solver.
- `ACO_TSP` / `ACO_CVRP`: more standard constructive MMAS-style implementations kept in the same backend.

For modifying FACO for constrained VRP, focus on `MFACO_CVRP`, not `ACO_CVRP` unless you specifically want the simpler baseline.

## Python Entry Point

The main Python class is `MFACO_CVRP` in `cvrptw-faco/faco.py`.

Flow:

1. User creates `MFACO_CVRP(coords, demand, capacity, n_ants, ...)`.
2. Python converts tensors/arrays to contiguous NumPy `float32` arrays.
3. Python calls `faco_opt.MFACO_CVRP(...)`, which is the pybind11 C++ class.
4. Python keeps torch copies of sparse pheromone, heuristic, and nearest-neighbor buffers for training/integration code.
5. User calls `solver.sample(...)`.
6. Python forwards `require_prob`, optional neural `prior`, and tracing flags into C++.
7. C++ returns costs, depot-separated routes, decoded routes, log probabilities, traces, raw routes, new-edge counts, and edge-survival stats.
8. User calls `update_pheromone(best_route, best_cost)` to push the selected solution back into the FACO pheromone state.

Important Python methods:

- `cvrptw-faco/faco.py:546` — `MFACO_CVRP` class.
- `cvrptw-faco/faco.py:553` — constructor and parameter handoff to C++.
- `cvrptw-faco/faco.py:664` — `sample(...)` wrapper.
- `cvrptw-faco/faco.py:693` — `update_pheromone(...)` wrapper.

## Pybind Bridge

The bridge is `cvrptw-faco/src/binding.cpp`.

Flow:

1. `PyMFACO_CVRP` owns `std::unique_ptr<MFACO_CVRP> solver`.
2. Its constructor validates input array shapes and creates the real C++ solver.
3. `sample(...)` calls `solver->sample(...)` and converts C++ vectors into NumPy arrays / Python objects.
4. `update_pheromone_from_route(...)` converts a Python route array to `std::vector<int32_t>` and calls the C++ pheromone update.
5. It exposes knobs such as `use_relocate`, `use_swap`, and `use_2opt_star` for local-search ablations.

Important bridge locations:

- `cvrptw-faco/src/binding.cpp:352` — `PyMFACO_CVRP` wrapper class.
- `cvrptw-faco/src/binding.cpp:356` — wrapper constructor.
- `cvrptw-faco/src/binding.cpp:526` — route-based pheromone update handoff.
- `cvrptw-faco/src/binding.cpp:930` — pybind class registration for `MFACO_CVRP`.
- `cvrptw-faco/src/binding.cpp:965` — exposed `sample(...)` method.
- `cvrptw-faco/src/binding.cpp:968` — exposed `update_pheromone_from_route(...)` method.

If you add new input data such as time windows, service times, vehicle limits, penalties, or constraint flags, you must update both `faco.py` and `binding.cpp` before C++ can see those values.

## C++ Solver Initialization

The core solver is `MFACO_CVRP` in `cvrptw-faco/src/mfaco_train.h` and `cvrptw-faco/src/mfaco_train.cpp`.

Constructor flow:

1. Store problem size: `n` includes depot `0`, `m = n - 1` customers.
2. Store ant count and candidate-list sizes.
3. Store FACO/MMAS hyperparameters: `rho`, `alpha`, `p_best`, `min_new_edges`, local-search flags, smoothing flags.
4. Store CVRP data: `coords`, `demand`, `capacity`.
5. Convert demand/capacity to scaled integers using `DEMAND_SCALE` to avoid floating-capacity drift.
6. Build nearest-neighbor candidate lists and backup lists.
7. Build sparse heuristic values, usually inverse distance.
8. Precompute distance-to-depot array `d0`.
9. Build an initial feasible CVRP solution.
10. Compute MMAS trail bounds `tau_min` and `tau_max`.
11. Initialize sparse pheromone matrix to `tau_max`.

Important C++ locations:

- `cvrptw-faco/src/mfaco_train.h:460` — `MFACO_CVRP` public state and constructor declaration.
- `cvrptw-faco/src/mfaco_train.cpp:1244` — `MFACO_CVRP` constructor.
- `cvrptw-faco/src/mfaco_train.cpp:1331` — nearest-neighbor list construction.
- `cvrptw-faco/src/mfaco_train.cpp:1405` — sparse heuristic construction.
- `cvrptw-faco/src/mfaco_train.cpp:1424` — depot-distance precomputation.
- `cvrptw-faco/src/mfaco_train.cpp:1433` — initial solution construction.

## Sampling Flow

`MFACO_CVRP::sample(...)` is the iteration-level ant sampler.

Flow:

1. Build a sparse probability matrix from pheromone, heuristic, and optional neural prior.
2. Choose start nodes for ants.
3. For each ant, call either the fast sampler or traced sampler.
4. The ant sampler constructs a depot-separated CVRP route.
5. If local search is enabled, the route is improved.
6. The route is canonicalized to start and end with depot `0`.
7. True route cost is recomputed as Euclidean path length over the depot-separated route.
8. Results are returned to Python.

Important methods:

- `cvrptw-faco/src/mfaco_train.cpp:1567` — `compute_probmat(...)`.
- `cvrptw-faco/src/mfaco_train.cpp:1647` — local route-cost lambda used during sampling output.
- `cvrptw-faco/src/mfaco_train.cpp:1673` — sampling begins using computed probabilities.
- `cvrptw-faco/src/mfaco_train.cpp:1842` — fast-mode call to `sample_ant_direct(...)`.
- `cvrptw-faco/src/mfaco_train.cpp:3460` — `sample_ant_direct(...)` implementation.
- `cvrptw-faco/src/mfaco_train.cpp:4041` — `sample_ant_direct_traced(...)` implementation.

## Probability Model

`compute_probmat(...)` creates the sparse transition scores over each node's candidate list.

Conceptually:

```text
score(u, v) = pheromone(u, v)^alpha * heuristic(u, v) * exp(optional_prior(u, v))
```

The exact implementation uses logs/exponentials for stability. `prior` is where a neural model can bias FACO toward promising edges.

Touch this area if you want to:

- Change the static heuristic from inverse distance to a constraint-aware heuristic.
- Add penalties for time-window slack, lateness risk, capacity pressure, or depot-return urgency.
- Change how a neural prior combines with ACO pheromone.

Primary file: `cvrptw-faco/src/mfaco_train.cpp:1567`.

## Current CVRP Constraint Logic

The current `MFACO_CVRP` code handles capacity constraints, not full CVRPTW time windows.

Current CVRP-specific assumptions:

- Depot is node `0`.
- Each customer must be visited once.
- Vehicle capacity is enforced while building routes.
- A route uses depot separators: for example `[0, 3, 7, 0, 2, 5, 0]`.
- Cost is Euclidean travel distance only.
- Feasibility is mainly about capacity and visiting every customer.

Capacity appears in several places:

- Constructor stores `capacity` and `capacity_int`.
- `demand` and `demand_int` are stored per node.
- Ant construction tracks route load and decides whether a customer can be inserted/continued.
- Local search checks loads before relocating/swapping/crossing customers between routes.

Important capacity/local-search areas:

- `cvrptw-faco/src/mfaco_train.h:491` — `capacity`, `capacity_int`, `demand`, and `demand_int` fields.
- `cvrptw-faco/src/mfaco_train.cpp:3460` — route construction with capacity/split behavior.
- `cvrptw-faco/src/mfaco_train.cpp:3718` — depot split/end-route handling in direct sampling.
- `cvrptw-faco/src/mfaco_train.cpp:3876` — local search call after construction.
- `cvrptw-faco/src/mfaco_train.cpp:2223` — inter-route local search.
- `cvrptw-faco/src/mfaco_train.h:650` — local-search move declarations.

## Pheromone Update Flow

After sampling, training/evaluation code selects a best route and calls `update_pheromone(...)`.

Flow:

1. Python passes a route into `MFACO_CVRP.update_pheromone(...)`.
2. Pybind converts the route to `std::vector<int32_t>`.
3. C++ evaporates pheromone using `rho`.
4. C++ deposits pheromone along the best route edges.
5. Trail values are clipped to `[tau_min, tau_max]` for MMAS behavior.
6. `source_route`, `best_route`, `source_cost`, and `best_cost` are updated if appropriate.

Important locations:

- `cvrptw-faco/faco.py:693` — Python `update_pheromone(...)`.
- `cvrptw-faco/src/binding.cpp:526` — pybind conversion and call.
- `cvrptw-faco/src/mfaco_train.cpp:1864` — C++ route-based pheromone update.

## Where to Modify for CVRPTW / VRPTW

To support CVRPTW or VRPTW correctly, you need to add time-window state and make the construction, cost, feasibility, local search, and API agree on it.

### 1. Add Problem Data

Add these fields to `MFACO_CVRP` or create a new class such as `MFACO_CVRPTW`:

```cpp
std::vector<float> ready_time;
std::vector<float> due_time;
std::vector<float> service_time;
float max_route_duration; // optional
```

Files to touch:

- `cvrptw-faco/faco.py:553` — accept `time_windows` / `ready_time` / `due_time` / `service_time` in Python.
- `cvrptw-faco/src/binding.cpp:356` — accept and validate new NumPy arrays.
- `cvrptw-faco/src/mfaco_train.h:460` — add fields and constructor parameters.
- `cvrptw-faco/src/mfaco_train.cpp:1244` — copy arrays and validate dimensions.

Recommendation: create `MFACO_CVRPTW` rather than overloading `MFACO_CVRP` too heavily if you want to preserve CVRP behavior.

### 2. Replace Feasibility Checks

For CVRPTW/VRPTW, a candidate customer is feasible only if the vehicle can arrive before its due time after travel and waiting:

```text
arrival = current_time + travel_time(current, candidate)
start_service = max(arrival, ready_time[candidate])
finish_service = start_service + service_time[candidate]
feasible_time = finish_service <= due_time[candidate]
```

If returning to depot must also respect depot close time:

```text
finish_service + travel_time(candidate, depot) <= due_time[depot]
```

Files to touch:

- `cvrptw-faco/src/mfaco_train.cpp:3460` — `sample_ant_direct(...)` candidate filtering.
- `cvrptw-faco/src/mfaco_train.cpp:4041` — `sample_ant_direct_traced(...)` candidate filtering and trace masks.
- `cvrptw-faco/src/mfaco_train.cpp:3718` — route split/depot close behavior.

This is the most important modification point.

### 3. Track Route State During Construction

Current CVRP construction mainly needs current node, visited flags, and load. CVRPTW also needs current route time.

Add/update state like:

```cpp
float current_time;
float current_load;
```

When moving to customer `v`:

```cpp
current_time = max(current_time + dist(curr, v), ready_time[v]) + service_time[v];
current_load += demand[v];
```

When closing a route at depot:

```cpp
current_time = 0.0f; // or depot ready time, depending on model
current_load = 0.0f;
curr = 0;
```

Touch the same construction methods:

- `cvrptw-faco/src/mfaco_train.cpp:3460`.
- `cvrptw-faco/src/mfaco_train.cpp:4041`.

### 4. Update Cost Function

Right now, cost is Euclidean travel length. For CVRPTW, decide whether objective is:

- total travel distance only, with hard time-window feasibility;
- total duration including waiting/service;
- distance plus penalty for time-window violation;
- number of vehicles first, then distance.

Files to touch:

- `cvrptw-faco/src/mfaco_train.cpp:1647` — route cost used when returning sampled results.
- `cvrptw-faco/src/mfaco_train.cpp:3889` — final direct-sampler cost recomputation.
- `cvrptw-faco/src/mfaco_train.cpp:4399` — traced-sampler final cost recomputation.
- `cvrptw-faco/src/mfaco_train.cpp:1864` — pheromone update receives and compares costs.

Recommendation: create helper functions such as:

```cpp
bool route_feasible_tw(const std::vector<int32_t>& route) const;
float route_cost_tw(const std::vector<int32_t>& route) const;
```

Then call those everywhere instead of duplicating logic.

### 5. Make Local Search Constraint-Aware

Current inter-route local search checks capacity/load deltas. For CVRPTW, every move must also preserve time-window feasibility for affected routes.

Moves that need changes:

- relocate customer between routes;
- swap customers between routes;
- 2-opt-star between routes;
- any intra-route reversal or reconnect move.

Files to touch:

- `cvrptw-faco/src/mfaco_train.cpp:2223` — inter-route LS driver.
- `cvrptw-faco/src/mfaco_train.h:650` — move declarations.
- `cvrptw-faco/src/mfaco_train.cpp:2355` — relocate logic area.
- `cvrptw-faco/src/mfaco_train.cpp:2422` — swap logic area.
- `cvrptw-faco/src/mfaco_train.cpp:2460` — 2-opt-star logic area.

For a first implementation, the safest approach is:

1. Generate the tentative changed route(s).
2. Recompute feasibility and cost for only those route(s).
3. Accept the move only if all affected routes satisfy capacity and time windows.

This is slower than incremental time-window delta evaluation, but much easier to get correct.

### 6. Make Heuristic Constraint-Aware

Current heuristic is mostly inverse distance. For CVRPTW, inverse distance alone may choose customers that cause time infeasibility.

Possible heuristic additions:

```text
eta(u, v) = 1 / (distance(u, v) + lambda_wait * wait_time + lambda_due * lateness_risk + epsilon)
```

Where:

```text
wait_time = max(0, ready_time[v] - arrival)
lateness_risk = max(0, arrival - due_time[v])
```

Files to touch:

- `cvrptw-faco/src/mfaco_train.cpp:1405` — `build_heuristic()`.
- `cvrptw-faco/src/mfaco_train.cpp:1567` — probability matrix if the heuristic needs dynamic state.

Important distinction:

- Static heuristic can use distance and node-level window width.
- Dynamic heuristic needs current route time, so it belongs in candidate selection inside `sample_ant_direct(...)`, not just `build_heuristic()`.

### 7. Update Trace Masks and Training Signals

If training code uses `require_prob=True` or traces, the valid mask must represent time-window feasibility, not only unvisited/capacity feasibility.

Files to touch:

- `cvrptw-faco/src/mfaco_train.cpp:4041` — traced sampler.
- `cvrptw-faco/src/mfaco_train.cpp:4216` — split/end-route handling in traced sampler.
- `cvrptw-faco/src/binding.cpp:965` — returned trace object shape stays same, but semantics change.

### 8. Rebuild the C++ Extension

After C++ changes, rebuild the extension from `cvrptw-faco/src`:

```powershell
cd cvrptw-faco\src
..\..\.venv\Scripts\python.exe setup.py build_ext --inplace
```

On Windows, the existing `setup.py` may need MSVC/OpenMP flag adjustments because it currently uses GCC-style `-fopenmp` outside macOS. If build fails on Windows, update `setup.py` to use `/openmp` and MSVC-compatible flags.

## Recommended Implementation Strategy

For CVRPTW, do this incrementally:

1. Add API fields for `ready_time`, `due_time`, and `service_time`.
2. Add helper functions for time-window feasibility and route cost.
3. Disable local search temporarily and make `sample_ant_direct(...)` produce feasible CVRPTW routes.
4. Mirror the same changes into `sample_ant_direct_traced(...)`.
5. Re-enable local search only after adding route-level feasibility checks to each move.
6. Add constraint-aware heuristic/prior behavior after correctness is stable.
7. Only then optimize feasibility checks with incremental delta calculations.

## Quick Modification Map

| Goal | Primary Files |
|---|---|
| Add new problem inputs | `faco.py`, `binding.cpp`, `mfaco_train.h`, `mfaco_train.cpp` |
| Change construction feasibility | `mfaco_train.cpp:3460`, `mfaco_train.cpp:4041` |
| Change route objective/cost | `mfaco_train.cpp:1647`, `mfaco_train.cpp:3889`, `mfaco_train.cpp:4399` |
| Change pheromone update behavior | `mfaco_train.cpp:1864` |
| Change static heuristic | `mfaco_train.cpp:1405` |
| Change dynamic transition scoring | `mfaco_train.cpp:1567`, `mfaco_train.cpp:3460` |
| Change local search constraints | `mfaco_train.cpp:2223` and LS move helpers |
| Expose new Python knobs | `faco.py:553`, `binding.cpp:930` |

## Most Important Places to Touch

If your goal is specifically “make FACO solve CVRPTW/VRPTW”, start here:

1. `cvrptw-faco/src/mfaco_train.h:460` — add time-window/service fields and helper declarations.
2. `cvrptw-faco/src/mfaco_train.cpp:1244` — copy and validate new data.
3. `cvrptw-faco/src/mfaco_train.cpp:3460` — enforce time-window feasibility in construction.
4. `cvrptw-faco/src/mfaco_train.cpp:4041` — enforce the same feasibility in traced construction.
5. `cvrptw-faco/src/mfaco_train.cpp:1647` — replace plain Euclidean cost with chosen constrained objective.
6. `cvrptw-faco/src/mfaco_train.cpp:2223` — update local search to reject infeasible constrained moves.
7. `cvrptw-faco/faco.py:553` and `cvrptw-faco/src/binding.cpp:356` — expose the new data from Python.

## Practical Advice

Do not only change the heuristic. A better heuristic can bias choices, but it does not guarantee feasibility. For CVRPTW/VRPTW, correctness must be enforced in construction and local search.

Do not only add penalties to cost unless you intentionally want a soft-constraint solver. For standard CVRPTW, routes violating capacity or time windows should be impossible or rejected, not merely expensive.

If you want the fastest path to a working constrained solver, create a new `MFACO_CVRPTW` class by copying `MFACO_CVRP` and changing only the constrained parts. After it works, refactor shared CVRP/CVRPTW helpers if needed.
