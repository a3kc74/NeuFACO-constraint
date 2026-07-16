# FACO Parameters: `extend_ls`, `smooth_mmas`, and `fixed_steps`

This note explains how three benchmark parameters are used by the current `cvrptw-faco` implementation and what they specifically do inside the FACO algorithm for CVRP/CVRPTW.

Relevant entry points:

- CLI options are defined in `cvrptw-faco/faco_test.py`.
- Python wrappers pass them through in `cvrptw-faco/faco.py`.
- C++ stores and uses them in `cvrptw-faco/src/mfaco_train.h` and `cvrptw-faco/src/mfaco_train.cpp`.

## Quick Summary

| Parameter | Main Role | Affects |
|---|---|---|
| `extend_ls` | Expands local-search checklist after improving moves | Local search depth/coverage |
| `smooth_mmas` | Switches pheromone update from deposit-based MMAS to smooth interpolation | Pheromone dynamics/convergence behavior |
| `fixed_steps` | Forces each ant to run a fixed number of relocation steps | Sampling/perturbation length |

## `extend_ls`

### What It Controls

`extend_ls` controls whether local search only processes the original set of changed/touched nodes, or whether it dynamically expands that set after it finds improvements.

In this implementation, FACO does not run a full exhaustive local search over every customer by default. Instead, it uses a focused `checklist` of nodes affected by the ant's route perturbation. This makes local search cheaper and tied to the part of the solution the ant changed.

### Where It Enters The Algorithm

The flag is passed from CLI/Python into the C++ solver constructor and stored as:

```cpp
bool extend_ls;
```

Local search is called after an ant has built/modified a route:

```text
sample_ant_direct / sample_ant_direct_traced
  -> build checklist of changed nodes
  -> inter_route_ls_optimized(...)
  -> intra_route_ls(...)
```

The important part is inside the local-search routines when a move improves the solution.

### What It Specifically Does

During local search, the solver keeps a queue-like `checklist`:

```text
checklist = nodes touched by new sampled edges or route modifications
```

Local search pops nodes from this list and tries improvement moves around them, such as:

- relocate moves
- swap moves
- 2-opt-star/inter-route moves
- intra-route 2-opt style moves

When `extend_ls = false`:

```text
Only nodes already in the original checklist are processed.
```

When `extend_ls = true`:

```text
After a successful improving move, nodes near the modified segment are appended to the checklist.
```

So a successful local-search move can trigger additional local-search exploration around the newly changed route area.

### Algorithmic Meaning

Without `extend_ls`, local search is more limited:

```text
ant changes route -> local search improves around directly touched nodes -> stop when original checklist is exhausted
```

With `extend_ls`, local search can propagate:

```text
ant changes route -> local search improves around touched nodes
                  -> newly affected neighbors are added
                  -> local search continues around those neighbors
```

This is useful because one improvement can create new improvement opportunities nearby. For example, relocating customer `u` after customer `v` changes the edges around both customers. The neighbors of that changed segment may now have better swap or 2-opt opportunities.

### Tradeoff

`extend_ls = false`:

- faster
- more focused
- may miss chained improvements

`extend_ls = true`:

- slower
- searches more thoroughly around changed segments
- can improve cost more, especially when the initial ant perturbation creates several downstream opportunities

### Relation To FACO

FACO is source-solution based: each ant modifies the current `source_route`, and then pheromone is updated based on the best modified route. `extend_ls` affects how aggressively each ant's modified route is polished before it competes for pheromone update.

So `extend_ls` does not change the pheromone formula or candidate probabilities directly. It changes the quality and runtime of the local-search phase after sampling.

## `smooth_mmas`

### What It Controls

`smooth_mmas` changes the pheromone update rule.

The implementation has two pheromone-update modes:

1. classic/deposit-style MMAS-like update
2. smooth interpolation-style update

The selected mode affects both:

- how `tau_min` and `tau_max` are computed
- how each pheromone value is updated after an iteration-best route is chosen

### Where It Enters The Algorithm

The flag is stored in the C++ solver as:

```cpp
bool smooth_mmas;
```

It is used during initialization and pheromone update:

```cpp
auto [tmin, tmax] = smooth_mmas
    ? calc_trail_limits_smooth(solution_cost)
    : calc_trail_limits_cl(solution_cost);
```

### Classic Mode: `smooth_mmas = false`

When `smooth_mmas` is disabled, trail limits depend on the current/best solution cost.

The classic trail bounds are computed approximately as:

```cpp
tau_max = 1 / (solution_cost * (1 - rho) + EPS)
tau_min = function(tau_max, p_best, k)
```

Then pheromone update does:

```cpp
tau *= (1 - rho)
if edge is in selected route:
    tau += 1 / selected_route_cost
tau = clamp(tau, tau_min, tau_max)
```

In algorithm terms:

```text
all candidate-edge pheromones evaporate
edges in the selected best route receive extra deposit
pheromone is clipped to [tau_min, tau_max]
```

This is closer to a traditional Max-Min Ant System behavior.

### Smooth Mode: `smooth_mmas = true`

When `smooth_mmas` is enabled, trail limits are fixed/simple:

```cpp
tau_max = 1.0
tau_min = 1.0 / k
```

The update no longer adds a deposit amount based on route cost. Instead, each pheromone value moves smoothly toward a target:

```cpp
if edge is in selected route:
    target = tau_max
else:
    target = tau_min

tau = (1 - rho) * tau + rho * target
```

In algorithm terms:

```text
route edges are pulled toward high pheromone
non-route edges are pulled toward low pheromone
updates are gradual interpolation rather than additive deposit
```

### Algorithmic Meaning

Classic MMAS mode makes route quality directly affect deposit magnitude:

```text
better route cost -> larger deposit -> stronger reinforcement
```

Smooth MMAS mode makes reinforcement more normalized:

```text
selected route edges always move toward 1.0
non-selected candidate edges always move toward 1/k
```

This can make pheromone behavior more stable because the update scale does not depend directly on route cost magnitude. But it can also reduce the distinction between very good and merely acceptable routes, since both are reinforced toward the same `tau_max` target.

### Tradeoff

`smooth_mmas = false`:

- stronger cost-sensitive reinforcement
- can converge faster
- can over-concentrate pheromone if good-looking routes are premature

`smooth_mmas = true`:

- smoother, normalized pheromone dynamics
- can preserve more stable probability ranges
- route cost affects whether a route becomes global best, but not the per-edge deposit amount directly

### Relation To FACO

FACO repeatedly samples route modifications from pheromone-weighted candidate lists. `smooth_mmas` changes how quickly the sparse pheromone matrix concentrates around edges from selected routes.

It therefore affects exploration/exploitation:

```text
classic MMAS: stronger exploitation of low-cost routes
smooth MMAS: gentler movement toward selected-route edge preferences
```

## `fixed_steps`

### What It Controls

`fixed_steps` controls how many internal relocation/sampling decisions each ant performs.

FACO sampling is not a simple constructive process that always visits all customers from scratch. Instead, each ant starts from the current `source_route` and performs route perturbations/relocations until a stopping rule is met.

`fixed_steps` overrides the default adaptive stopping rule.

### Where It Enters The Algorithm

The flag is stored in C++ as:

```cpp
int32_t fixed_steps; // if > 0, fixed number of steps
```

Inside both non-traced and traced ant sampling loops, the stopping logic is:

```cpp
if (fixed_steps > 0) {
    if (steps >= fixed_steps)
        break;
} else {
    if (new_edges_cross >= min_new_edges || visited_count >= m)
        break;
}
```

There is also a safety cap:

```cpp
max_steps = fixed_steps if fixed_steps > 0 else m * 4
```

### Default Behavior: `fixed_steps = 0`

When `fixed_steps` is zero, FACO uses adaptive stopping:

```text
stop when enough new/cross-route edges have been introduced
OR when enough customers have been visited/touched
```

The main condition is:

```text
new_edges_cross >= min_new_edges || visited_count >= m
```

Here:

- `new_edges_cross` counts route-boundary/cross-route changes relative to the source route.
- `min_new_edges` says how much novelty/perturbation the ant should introduce before stopping.
- `visited_count >= m` prevents the ant from trying to touch more customers than exist.

So with `fixed_steps = 0`, the ant stops based on how much meaningful route change it has produced.

### Fixed Behavior: `fixed_steps > 0`

When `fixed_steps` is positive, the ant ignores the `min_new_edges` stopping condition and runs for exactly that many sampling iterations, unless it hits a safety/failure break.

For example:

```bash
--fixed_steps 20
```

means each ant tries to perform up to 20 internal route modification decisions.

These decisions can include:

- selecting a customer to relocate after the current node
- selecting depot `0` to split/end a route
- traversing to an existing route depot
- updating route linked-list structure
- updating route loads and checklist entries

### Algorithmic Meaning

`fixed_steps` controls perturbation length.

With adaptive stopping:

```text
ants stop after enough new route structure has been created
```

With fixed stopping:

```text
ants are forced to keep modifying/traversing for a fixed number of decisions
```

This matters because FACO is source-solution based. Each ant starts from `source_route`, modifies it, and returns a candidate route. The number of internal steps controls how far the ant can move away from the current source solution.

### Tradeoff

`fixed_steps = 0`:

- adaptive to actual new-edge creation
- usually cheaper
- may stop early if `min_new_edges` is reached quickly
- perturbation amount depends on instance and random choices

`fixed_steps > 0`:

- more controlled and comparable across ants
- can increase exploration if the value is larger than the default adaptive stopping length
- can waste time on extra moves if value is too high
- can degrade routes if forced moves disrupt good source structure

### Relation To `min_new_edges`

`fixed_steps` and `min_new_edges` are alternative stopping controls.

If `fixed_steps = 0`:

```text
min_new_edges is active
```

If `fixed_steps > 0`:

```text
min_new_edges is effectively bypassed for ant-loop stopping
```

So these two parameters should not be interpreted as additive. `fixed_steps` takes precedence.

## How They Interact

These parameters affect different parts of one FACO iteration:

```text
1. sample ant route modification
   controlled by fixed_steps / min_new_edges

2. repair/check CVRPTW feasibility
   independent of these parameters

3. local search
   controlled partly by extend_ls

4. select iteration-best route in Python
   based on returned cost

5. pheromone update
   controlled by smooth_mmas
```

A simplified flow is:

```text
source_route
  -> ant relocation loop
       fixed_steps controls how long this loop runs
  -> route flattening and CVRPTW feasibility handling
  -> local search
       extend_ls controls whether local-search checklist expands
  -> Python chooses best ant route
  -> pheromone update
       smooth_mmas controls update formula
  -> source_route for next iteration
```

## Practical Tuning Notes

### If Cost Is Poor And Diversity Is Low

Try increasing exploration:

```text
fixed_steps > 0, e.g. 16, 24, 32
```

This can force ants farther away from the current source route.

### If Cost Improves But Runtime Is High

Avoid expensive chained local search:

```text
extend_ls = false
```

or keep `fixed_steps` modest.

### If Pheromone Converges Too Aggressively

Try:

```text
smooth_mmas = true
```

This makes pheromone updates smoother and less directly cost-scaled.

### If Pheromone Does Not Exploit Good Routes Enough

Use:

```text
smooth_mmas = false
```

This keeps classic deposit behavior where lower-cost selected routes reinforce their edges more strongly.

## Bottom Line

- `fixed_steps` controls how much each ant perturbs the current source route.
- `extend_ls` controls how deeply local search follows up on those perturbations.
- `smooth_mmas` controls how the selected route changes future sampling probabilities.

Together, they shape the FACO balance between:

```text
exploration through route perturbation
improvement through local search
exploitation through pheromone reinforcement
```
