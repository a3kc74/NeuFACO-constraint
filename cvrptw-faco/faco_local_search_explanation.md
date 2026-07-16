# FACO Local Search Integration for CVRP/CVRPTW

This document explains how local search is currently plugged into the `cvrptw-faco` FACO algorithm and which local-search operators are being used.

The current implementation is CVRP/CVRPTW-focused after refactoring. The relevant C++ logic is in:

- `cvrptw-faco/src/mfaco_train.cpp`
- `cvrptw-faco/src/mfaco_train.h`
- `cvrptw-faco/src/binding.cpp`
- `cvrptw-faco/faco.py`

## High-Level FACO Flow

For each FACO iteration, Python calls:

```python
costs, routes, *_ = solver.sample(prior=None)
```

The C++ solver then samples `n_ants` candidate routes. Each ant follows this broad pipeline:

```text
current source_route
  -> build linked-list route representation
  -> perform relocation/split sampling steps
  -> flatten linked-list route back to depot-separated route
  -> enforce/repair CVRPTW constraints if needed
  -> run local search if enabled
  -> validate/rollback if local search breaks CVRPTW feasibility
  -> compute final route cost
```

So local search is not the main construction method. It is a post-processing/improvement phase after an ant has already modified the current `source_route`.

## How Local Search Is Enabled

The Python constructors expose:

```python
use_local_search: bool = True
extend_ls: bool = False
```

These are passed through:

```text
faco.py
  -> faco_opt.MFACO_CVRP / faco_opt.MFACO_CVRPTW
  -> binding.cpp
  -> MFACO_CVRP constructor
  -> C++ fields use_local_search and extend_ls
```

In C++, the solver stores:

```cpp
bool use_local_search;
bool extend_ls;
```

Local search only runs when:

```cpp
use_local_search && !checklist.empty()
```

So there are two requirements:

1. local search must be enabled, and
2. the ant must have touched/changed at least one relevant node or edge.

If `use_local_search = false`, the sampled route skips the local-search phase entirely.

## What The Checklist Is

FACO uses focused local search. It does not scan every possible move over every customer by default.

During ant sampling, the solver builds a `checklist` of nodes related to newly created or modified edges. Examples of nodes added to the checklist include:

- current node `u` when edge `(u, v)` is new relative to the source route
- selected customer `v`
- nodes around route split points
- nodes affected by relocation moves

Conceptually:

```text
checklist = nodes touched by the ant's route perturbation
```

This makes local search cheaper than a full exhaustive search because it concentrates effort around the part of the solution the ant changed.

## Where Local Search Runs In The Sampler

There are two ant sampling functions:

1. `sample_ant_direct(...)`
2. `sample_ant_direct_traced(...)`

Both use the same idea but differ in whether they record decision traces for log-probability replay.

### Non-Traced Sampling Path

The non-traced path is used by the benchmark default:

```python
solver.sample(prior=None)
```

with `require_prob=False`.

The C++ path is:

```text
MFACO_CVRP::sample(...)
  -> MFACO_CVRP::sample_ant_direct(...)
```

Inside `sample_ant_direct(...)`, after the route is sampled and flattened, local search is called like this:

```text
if use_local_search and checklist is not empty:
    inter_route_ls_optimized(route_out, pos_ls, checklist, in_checklist)
    intra_route_ls(route_out, checklist)
```

So the non-traced path applies:

```text
inter-route LS first
then intra-route LS
```

### Traced Sampling Path

The traced path is used when `require_prob=True`.

The C++ path is:

```text
MFACO_CVRP::sample(...)
  -> MFACO_CVRP::sample_ant_direct_traced(...)
```

Inside `sample_ant_direct_traced(...)`, local search is called like this:

```text
if use_local_search and checklist is not empty:
    intra_route_ls(route_out, checklist)
    inter_route_ls_optimized(route_out, pos_ls, checklist, in_checklist)
    intra_route_ls(route_out, checklist)
```

So the traced path applies:

```text
intra-route LS
then inter-route LS
then intra-route LS again
```

The extra intra-route pass after inter-route changes is useful because inter-route moves can create new within-route 2-opt opportunities.

## CVRPTW Feasibility Guard Around Local Search

For CVRPTW, local search can accidentally create a route that violates time windows. The implementation protects against this by saving the route before local search.

The logic is:

```text
route_before_ls = route_out
run local search
if route_out is not fully feasible:
    route_out = route_before_ls
```

After that, the code also has a safety fallback:

```text
if route_out is still not fully feasible:
    enforce_time_windows(route_out)
    if still infeasible:
        route_out = source_route
```

So local search is allowed to improve a route only if the final route remains feasible. For CVRPTW, feasibility means:

- route starts and ends at depot `0`
- every customer appears exactly once
- each route respects capacity
- each customer arrival respects due time
- each route can return to depot before depot due time

This is important: local search is distance-improvement oriented, but CVRPTW feasibility is checked afterward.

## Local Search Operators Used

The current implementation uses custom C++ local search, not PyVRP local search.

There are two main local-search functions:

```cpp
intra_route_ls(...)
inter_route_ls_optimized(...)
```

Together, these implement the operators currently used by FACO.

## Operator 1: Intra-Route 2-Opt

Function:

```cpp
MFACO_CVRP::intra_route_ls(std::vector<int32_t>& route,
                           std::vector<int32_t>& checklist)
```

### What It Does

`intra_route_ls(...)` improves the order of customers inside the same route.

It first parses the full depot-separated route:

```text
[0, a, b, c, 0, d, e, f, 0]
```

into separate customer sequences:

```text
route 1: [a, b, c]
route 2: [d, e, f]
```

Then it processes only nodes in `checklist`.

For each checklist node `a`, it tries nearest-neighbor-based 2-opt moves inside `a`'s route.

### 2-Opt Meaning

A 2-opt move removes two edges and reconnects the segment in the opposite order.

Example:

```text
before:
... a_prev -> a -> ... -> b_prev -> b ...

after:
... a_prev -> b_prev -> ... reversed ... -> a -> b ...
```

This is equivalent to reversing a route segment when doing so shortens travel distance.

### Candidate Restriction

The function does not compare `a` with every other customer. It uses the sparse nearest-neighbor list:

```cpp
nn_list[a * k + jj]
```

So it mainly tries 2-opt opportunities involving geographically nearby customers.

This keeps the local search fast.

### Acceptance Rule

The operator computes the distance improvement:

```text
improvement = old_edge_distance - new_edge_distance
```

If the best improvement is positive enough:

```cpp
max_diff > 1e-6
```

then it reverses the selected segment.

### Role Of `extend_ls`

When `extend_ls = true`, after a successful segment reversal, nodes around the changed segment are added back into the checklist.

This means local search can continue exploring newly affected neighborhoods.

When `extend_ls = false`, only the original checklist is processed.

## Operator 2: Inter-Route Relocate

Function:

```cpp
MFACO_CVRP::inter_route_ls_optimized(...)
```

Controlled by:

```cpp
use_relocate
```

Exposed to Python through pybind as:

```python
solver._cpp.use_relocate
```

or on the pybind object as property `use_relocate`.

### What It Does

Relocate moves one customer from its current position and inserts it after another customer, usually in a different route.

Conceptually:

```text
before:
route A: 0 ... prev_u -> u -> next_u ... 0
route B: 0 ... v -> next_v ... 0

after:
route A: 0 ... prev_u -> next_u ... 0
route B: 0 ... v -> u -> next_v ... 0
```

### Capacity Check

Before moving `u` into route `B`, the code checks:

```text
route_loads[target_route] + demand[u] <= capacity
```

So relocate is only allowed when the destination route remains capacity-feasible.

### Cost Check

The move is accepted only if the distance delta improves the solution.

The code compares removed edges and added edges. If the new route structure is shorter, it applies the move.

### Why It Matters

Relocate is important for CVRP/CVRPTW because route quality often depends on assigning each customer to a good vehicle route. A customer may be in a feasible route but better placed in another route.

Relocate changes both:

- customer-to-route assignment
- customer position inside the target route

## Operator 3: Inter-Route Swap

Function:

```cpp
MFACO_CVRP::inter_route_ls_optimized(...)
```

Controlled by:

```cpp
use_swap
```

Exposed to Python as property `use_swap`.

### What It Does

Swap exchanges two customers, usually from different routes.

Conceptually:

```text
before:
route A: ... prev_u -> u -> next_u ...
route B: ... prev_v -> v -> next_v ...

after:
route A: ... prev_u -> v -> next_u ...
route B: ... prev_v -> u -> next_v ...
```

### Capacity Check

For two different routes, the swap must keep both route loads within capacity.

The route loads change like:

```text
route A load = route A load - demand[u] + demand[v]
route B load = route B load - demand[v] + demand[u]
```

The move is valid only if both updated loads are within capacity.

### Cost Check

The implementation estimates the edge-distance delta around both swapped customers. It applies the swap only if the total route distance decreases.

### Why It Matters

Swap helps fix bad customer assignments without changing the number of routes. It can be useful when two customers are each better served in the other's route.

## Operator 4: 2-Opt-Star

Function:

```cpp
MFACO_CVRP::inter_route_ls_optimized(...)
```

Controlled by:

```cpp
use_2opt_star
```

Exposed to Python as property `use_2opt_star`.

### What It Does

2-opt-star is an inter-route edge exchange. Instead of reversing a segment inside one route, it exchanges route tails between two routes.

Conceptually:

```text
before:
route A: 0 ... u -> next_u ... 0
route B: 0 ... v -> next_v ... 0

after:
route A: 0 ... u -> next_v ... 0
route B: 0 ... v -> next_u ... 0
```

This swaps the suffixes after `u` and `v` between two routes.

### Capacity Check

Because tails are exchanged, the load of both resulting routes changes. The implementation checks that both resulting route loads remain within capacity.

### Cost Check

The operator compares the old crossing edges:

```text
u -> next_u
v -> next_v
```

against the new crossing edges:

```text
u -> next_v
v -> next_u
```

If the new connections reduce distance and capacity remains feasible, the move can be accepted.

### Why It Matters

2-opt-star is powerful for VRP because it can restructure two routes at once. It can correct poor route boundaries more effectively than a single-customer relocate.

## Operator Control Flags

The C++ solver has three operator flags:

```cpp
bool use_relocate;
bool use_swap;
bool use_2opt_star;
```

They are initialized to `true` in the constructor.

The pybind layer exposes them as mutable Python properties for both `MFACO_CVRP` and `MFACO_CVRPTW`:

```python
solver._cpp.use_relocate
solver._cpp.use_swap
solver._cpp.use_2opt_star
```

This allows ablation experiments, for example:

```python
solver._cpp.use_swap = False
solver._cpp.use_2opt_star = False
```

Then local search would use relocate but skip swap and 2-opt-star.

## Exact Local Search Order

### Default Non-Traced Benchmark Path

The default benchmark uses:

```python
solver.sample(prior=None)
```

which calls non-traced sampling.

The local-search order is:

```text
inter_route_ls_optimized
intra_route_ls
```

So it first tries route-assignment and inter-route structure improvements, then cleans up within-route ordering.

### Traced Path

When `require_prob=True`, the traced sampler uses:

```text
intra_route_ls
inter_route_ls_optimized
intra_route_ls
```

This first improves route interiors, then tries inter-route moves, then performs another intra-route cleanup pass.

## How Local Search Interacts With FACO Pheromone

Local search changes the route before cost is returned to Python.

The Python benchmark then does:

```python
best_idx = argmin(costs)
best_route = routes[best_idx]
solver.update_pheromone(best_route, best_cost)
```

So if local search improves a sampled ant route, the improved route is what competes to update pheromone and become the next `source_route`.

This means local search affects FACO in two ways:

1. immediate route cost is lower for the current iteration
2. pheromone and source-route updates are based on locally improved routes

So local search is part of the feedback loop, not just a final polish for reporting.

## How Local Search Interacts With CVRPTW

The operators themselves are mainly distance/capacity-oriented. CVRPTW time-window safety is handled by validation around local search.

The current flow is:

```text
save route_before_ls
run local search
if route is not fully CVRPTW-feasible:
    rollback to route_before_ls
```

Then a final safety check can repair or fall back:

```text
if still infeasible:
    enforce_time_windows(route)
    if still infeasible:
        route = source_route
```

So local search is allowed to change CVRPTW routes only when the final result passes full feasibility.

## Difference From GFACS Local Search

This FACO local search is custom C++ code.

It is not the same as GFACS's PyVRP local search path.

FACO local search:

```text
custom focused checklist-based LS
operators: intra-route 2-opt, relocate, swap, 2-opt-star
implemented in mfaco_train.cpp
```

GFACS local search:

```text
can call PyVRP batched local search
more general VRPTW-oriented local-search engine
```

This difference matters because FACO local search is faster and more focused, but may be less powerful than a full VRPTW local-search engine.

## Summary

Current FACO local search is plugged in after ant route sampling and before cost calculation.

It uses:

- checklist-focused local search
- inter-route relocate
- inter-route swap
- inter-route 2-opt-star
- intra-route 2-opt

The main algorithmic role is:

```text
ant creates a perturbed route
local search improves that route around changed nodes
feasibility validation protects CVRPTW constraints
improved route competes for pheromone/source-route update
```

So local search is a central part of FACO's exploitation step: it converts stochastic route perturbations into stronger candidate solutions before pheromone learning reinforces them.
