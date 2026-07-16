# FACO C++ Implementation Overview

This document explains how the C++/pybind11 FACO implementation in `cvrptw-faco/src` is organized and how the main files relate to each other.

> Note: the user prompt mentioned `crptw-faco/src`, but the matching implementation in this repository is under `cvrptw-faco/src`.

## High-Level Architecture

The C++ backend is built as a Python extension module named `faco_opt`.

```text
Python code
   |
   | imports faco_opt
   v
binding.cpp  -- pybind11 wrapper layer
   |
   | wraps classes/functions declared in
   v
mfaco_train.h  -- public C++ declarations and solver data structures
   |
   | implemented by
   v
mfaco_train.cpp -- FACO/ACO algorithms, sampling, pheromone updates, local search
   |
   | uses nearest-neighbor acceleration from
   v
kd_tree.h -- KD-tree for candidate/nearest-neighbor lists

setup.py -- compiles binding.cpp + mfaco_train.cpp into faco_opt
```

The backend supports multiple solver variants exposed to Python:

- `MFACO_TSP`: model-guided/focused ACO for TSP.
- `MFACO_CVRP`: model-guided/focused ACO for CVRP.
- `MFACO_CVRPTW`: CVRP with time-window handling, exposed as a Python subclass-style binding of the CVRP wrapper.
- `ACO_TSP`: standard constructive ACO/MMAS for TSP.
- `ACO_CVRP`: standard constructive ACO/MMAS for CVRP.
- `MFACOTrace`: compact trace container used to send ant decision traces back to Python for replay/log-probability training.

## `setup.py`

`setup.py` is the build script for the Python extension.

### Main Responsibilities

- Defines a setuptools `Extension` named `faco_opt`.
- Compiles:
  - `binding.cpp`
  - `mfaco_train.cpp`
- Adds include directories for:
  - `pybind11.get_include()`
  - the local source directory `.`
- Enables C++17.
- Enables OpenMP depending on platform:
  - Windows: `/openmp`
  - Linux: `-fopenmp`
  - macOS: `-Xpreprocessor -fopenmp` and links `-lomp`

### Relationship to Other Files

`setup.py` is the entry point for building the native module. It does not implement FACO logic itself. Its job is to compile the wrapper layer and algorithm layer together into a shared library importable from Python as `faco_opt`.

## `binding.cpp`

`binding.cpp` is the Python interface layer. It uses pybind11 to expose the C++ solvers and their data to Python.

### Main Responsibilities

- Includes `mfaco_train.h`, so it can wrap the solver classes declared there.
- Includes pybind11 headers for Python classes, NumPy arrays, and STL conversion.
- Exposes the module `faco_opt` via `PYBIND11_MODULE(faco_opt, m)`.
- Wraps C++ solver classes in Python-facing wrapper classes:
  - `PyMFACOTrace`
  - `PyMFACO_TSP`
  - `PyMFACO_CVRP`
  - `PyMFACO_CVRPTW`
  - `PyACO_TSP`
  - `PyACO_CVRP`
- Exposes OpenMP controls/utilities:
  - `set_num_threads`
  - `get_max_threads`
  - `get_num_procs`
  - `set_dynamic`
  - `get_dynamic`
- Converts NumPy input arrays into raw C++ pointers for constructors and methods.
- Converts internal C++ vectors into NumPy array views for fast Python access.

### Important Helper Functions

- `make_view<T>(T *data, std::vector<py::ssize_t> shape)`
  - Creates a NumPy view over existing C++ memory without copying.
  - Computes row-major strides.
  - Used for exposing vectors such as pheromone matrices, nearest-neighbor lists, routes, and traces.

- `make_1d_view<T>(T *data, py::ssize_t len)`
  - Convenience wrapper for 1D NumPy views.

- `make_2d_view<T>(T *data, py::ssize_t rows, py::ssize_t cols)`
  - Convenience wrapper for 2D NumPy views.

### `PyMFACOTrace`

This is a Python-facing wrapper around `MFACOTraceBatch`.

It exposes trace arrays as NumPy views:

- `starts`
- `curr_nodes`
- `chosen_nodes`
- `is_stochastic`
- `pick_j`
- `valid_mask`
- `is_new_edge`
- `start_nodes`
- `n_decisions`
- `n_ants`

It also provides `to_trace_list()`, which converts the compact batched trace representation into a Python list of dictionaries, one dictionary per ant. This is useful for Python-side replay of choices, for example when computing training log probabilities.

### Solver Wrappers

Each Python wrapper owns a `std::unique_ptr` to the actual C++ solver in `mfaco_train.cpp`.

For example:

- `PyMFACO_TSP` owns `std::unique_ptr<MFACO_TSP>`.
- `PyMFACO_CVRP` owns `std::unique_ptr<MFACO_CVRP>`.
- `PyACO_TSP` owns `std::unique_ptr<ACO_TSP>`.
- `PyACO_CVRP` owns `std::unique_ptr<ACO_CVRP>`.

The wrappers mainly do four things:

1. Validate Python/NumPy input shapes.
2. Pass raw pointers and scalar parameters into C++ constructors.
3. Call solver methods such as `sample`, `seed_rng`, and pheromone update functions.
4. Expose solver state as Python properties, often as zero-copy NumPy views.

### Exposed Python Classes

The module registers these main classes:

- `MFACOTrace`
- `MFACO_TSP`
- `MFACO_CVRP`
- `MFACO_CVRPTW`
- `ACO_TSP`
- `ACO_CVRP`

### Relationship to Other Files

`binding.cpp` depends on `mfaco_train.h` for declarations and links against `mfaco_train.cpp` for actual solver implementations. It does not directly include `kd_tree.h`; KD-tree usage is internal to the algorithm implementation.

## `kd_tree.h`

`kd_tree.h` implements a lightweight 2D KD-tree used to accelerate nearest-neighbor lookup.

### Main Responsibilities

- Defines a 2D point/vector type.
- Builds a KD-tree over coordinate points.
- Supports nearest-neighbor style queries used when constructing candidate lists.
- Helps avoid expensive all-pairs scans where possible.

### Important Types

#### `Vec2d`

A small structure representing a 2D point/vector. It stores coordinate values and supports distance-related operations used by the KD-tree.

#### `KDTree`

The KD-tree class stores points and recursively partitions them by coordinate dimension.

Important methods include:

- `KDTree(const std::vector<Point> &points, bool round_distances = true)`
  - Builds the KD-tree from input points.
  - The `round_distances` flag controls whether distance calculations are rounded.

- `build(uint32_t l, uint32_t u, const Bounds &bounds)`
  - Recursively builds the tree over a point range.

- `find_max_spread_dimension(uint32_t low, uint32_t up)`
  - Chooses the split dimension with the largest coordinate spread.

- `get_coordinate(const Point &p, int8_t dim)`
  - Returns x or y coordinate for sorting/splitting.

- `select(uint32_t lower, uint32_t upper, uint32_t middle, int8_t dim)`
  - Selects the median point along a dimension.

- `nn(uint32_t point_idx)`
  - Finds a nearest neighbor for a point index.

- `print_in_dot_format(...)`
  - Debug/visualization helper for Graphviz DOT output.

### Relationship to Other Files

`kd_tree.h` is included by `mfaco_train.cpp`, not by the binding layer. The solver implementation uses it when building nearest-neighbor candidate lists such as `nn_list` and `backup_list`. Those lists then drive fast ant construction and local-search neighborhoods.

## `mfaco_train.h`

`mfaco_train.h` is the public C++ declaration file for the FACO/ACO backend.

### Main Responsibilities

- Defines constants, enums, utility classes, result structures, and solver class declarations.
- Establishes the shared namespace `mfaco`.
- Declares the public fields and methods used by `binding.cpp`.
- Declares private/internal helper methods implemented in `mfaco_train.cpp`.

### Constants and Utility Types

- `MAX_CAND_LIST_SIZE`
  - Maximum candidate-list size, used because candidate masks are represented compactly.

- `EPS` and `LOG_EPS`
  - Numerical stability constants.

- `DEMAND_SCALE`
  - Scaling constant for converting floating-point CVRP demands into integer-like quantities.

- `DistanceType`
  - Currently declares `EXPLICT_EUC_2D`, meaning explicit Euclidean 2D distance support.

### `Xoshiro128Plus`

A compact pseudo-random number generator used by the solvers.

Important methods:

- `seed(uint64_t s)`
  - Initializes RNG state using SplitMix64.

- `next_u32()`
  - Returns a random 32-bit integer.

- `next_uint(uint32_t max_exclusive)`
  - Returns a uniform integer in `[0, max_exclusive)`.

- `next_float()`
  - Returns a float in `[0, 1)`.

The solver classes use this RNG for ant start selection, probabilistic transition sampling, and other stochastic choices.

### Trace Structures

#### `MFACOTrace`

Stores the decision trace for one ant.

Fields include:

- `start_node`
- `curr_nodes`
- `chosen_nodes`
- `is_stochastic`
- `pick_j`
- `valid_mask`
- `is_new_edge`

Important methods:

- `clear()`
- `reserve(size_t n)`

#### `MFACOTraceBatch`

Stores traces for many ants in compact batched form.

Fields include:

- `starts`: prefix-sum offsets into the decision arrays.
- `curr_nodes`
- `chosen_nodes`
- `is_stochastic`
- `pick_j`
- `valid_mask`
- `is_new_edge`
- `start_nodes`

Important methods:

- `clear()`
- `reserve(int32_t n_ants, int32_t max_decisions)`
- append/merge-style helpers used by the implementation.

This is the structure wrapped by `PyMFACOTrace` in `binding.cpp`.

### Result Structures

The header declares result containers used to return solver output from C++ to Python wrappers. These include sample result structures containing route costs, routes, probability/log-probability information, and traces.

The wrappers in `binding.cpp` convert these structures into Python dictionaries, NumPy arrays, or `MFACOTrace` objects.

### Solver Classes

#### `MFACO_TSP`

Model-guided FACO solver for TSP.

Typical state includes:

- Problem size and parameters: `n`, `n_ants`, `k`, `bl`, `min_new_edges`, `rho`, `alpha`, `p_best`.
- Coordinates and distances.
- Candidate structures: `nn_list`, `backup_list`, nearest-neighbor position maps.
- Pheromone and heuristic arrays.
- Source/best route state.
- Local-search options.
- RNG state.

Important methods include:

- Constructor from coordinates and FACO parameters.
- `seed_rng(...)`.
- `sample(...)` for generating ant solutions, optionally with traces/probabilities.
- Snapshot/pheromone-loading helpers used by the Python training loop.
- Pheromone update helpers.
- Local-search helpers such as 2-opt style route improvement.
- Distance and route-cost helpers.

#### `MFACO_CVRP`

Model-guided FACO solver for capacitated VRP.

It extends the TSP-style ideas with:

- Depot-aware routes.
- Customer demands.
- Vehicle capacity constraints.
- Route splitting/flattening logic.
- Capacity-feasible local search.
- CVRP-specific pheromone update behavior.

Important methods include:

- Constructor from coordinates, demands, capacity, and FACO parameters.
- `seed_rng(...)`.
- `sample(...)`.
- `update_pheromone_from_route(...)`.
- Route-cost and feasibility helpers.
- Split-DP helpers for turning customer permutations into capacity-feasible routes.
- Intra-route and inter-route local search.
- Timing helpers exposed to Python.

#### `MFACO_CVRPTW`

CVRP with time windows.

It builds on the CVRP structure and adds:

- Service durations.
- Earliest/latest time windows.
- Time-feasibility checks.
- Time-window enforcement/repair during route generation and local search.

This class is exposed in `binding.cpp` as `MFACO_CVRPTW` and shares many properties/methods with the CVRP wrapper.

#### `ACO_TSP`

Standard constructive ACO/MMAS solver for TSP.

Important methods include:

- Constructor.
- `seed_rng(...)`.
- Candidate-list construction.
- Heuristic construction.
- `sample(...)`.
- `run(...)`.
- `update_pheromone(...)`.
- Route-cost and trail-limit helpers.

Unlike `MFACO_TSP`, this class is not centered on modifying a source solution; it constructs routes using pheromone and heuristic information in the more standard ACO style.

#### `ACO_CVRP`

Standard constructive ACO/MMAS solver for CVRP.

It adds CVRP-specific capacity handling to the standard ACO approach:

- Depot/customer routing.
- Demand/capacity feasibility.
- Route construction with returns to depot.
- CVRP local search if enabled.
- Pheromone update over flattened CVRP routes.

### Relationship to Other Files

`mfaco_train.h` is the central contract between the implementation and binding layer:

- `binding.cpp` includes it to know class layouts, public fields, and method signatures.
- `mfaco_train.cpp` includes it to implement the declared methods.
- `setup.py` compiles the implementation and binding together.

## `mfaco_train.cpp`

`mfaco_train.cpp` contains the actual algorithm implementation.

### Main Responsibilities

- Implements all methods declared in `mfaco_train.h`.
- Builds candidate lists and heuristics.
- Samples ants and constructs routes.
- Tracks route decisions for Python-side training.
- Applies local search.
- Computes solution costs.
- Updates pheromone trails.
- Maintains best/source solution state.
- Handles TSP, CVRP, CVRPTW, and standard ACO variants.

### Dependency on `kd_tree.h`

`mfaco_train.cpp` includes and uses `KDTree` when building nearest-neighbor structures.

For example, TSP candidate-list construction creates coordinate points, builds a `KDTree`, and queries nearest neighbors to populate arrays such as `nn_list`. Those arrays are later used by ant sampling and local search.

### Common Implementation Pattern

The solvers generally follow this pattern:

1. Constructor receives raw input arrays from `binding.cpp`.
2. Coordinates/demands/time-window data are copied into C++ vectors.
3. Nearest-neighbor/candidate lists are built.
4. Heuristic values are computed from distances.
5. Pheromone values are initialized.
6. `sample(...)` constructs solutions for many ants.
7. Best solutions are tracked.
8. Optional local search improves sampled solutions.
9. Pheromone updates reinforce good routes and apply evaporation/min-max bounds.
10. Results and traces are returned to the wrapper layer.

### TSP Implementation Functions

The TSP implementations contain logic for:

- `build_nn_lists()`
  - Builds candidate lists from coordinates, using the KD-tree for nearest-neighbor search.

- `build_heuristic()`
  - Computes heuristic desirability values, usually based on inverse distance.

- `get_route_cost(...)`
  - Computes total tour length.

- `sample(...)`
  - Runs ant construction for a batch of ants.

- Pheromone update methods
  - Apply evaporation and route reinforcement.
  - Maintain `tau_min`/`tau_max` limits for MMAS-style behavior.

- Local-search methods
  - Improve tours using neighborhood moves such as 2-opt.

### CVRP Implementation Functions

The CVRP implementation extends ant construction and local search with capacity-aware route handling.

Important responsibilities include:

- Representing routes as flattened sequences with depot separators.
- Checking route load/capacity feasibility.
- Computing CVRP route cost.
- Splitting customer permutations into feasible routes.
- Running intra-route and inter-route local search.
- Updating pheromone using route edges.
- Counting or encouraging new edges relative to a source solution in MFACO mode.

### CVRPTW Implementation Functions

The CVRPTW implementation further adds time-window feasibility.

Important responsibilities include:

- Computing arrival/service timing along routes.
- Checking whether a flattened route satisfies time windows.
- Repairing or enforcing route order where possible.
- Guarding local-search changes so infeasible time-window routes can be rejected or rolled back.

### Trace Generation

During FACO sampling, the implementation records decisions into `MFACOTrace` or `MFACOTraceBatch`:

- Current node.
- Chosen next node.
- Whether the choice was stochastic.
- Candidate index in the nearest-neighbor row.
- Valid-candidate bit mask.
- Whether the edge is new relative to the source route.

These traces flow back to Python through `binding.cpp` and are used for learning/replay.

### Relationship to Other Files

`mfaco_train.cpp` is the algorithm core:

- It implements the declarations in `mfaco_train.h`.
- It uses `kd_tree.h` for nearest-neighbor candidate lists.
- It is wrapped by `binding.cpp` for Python access.
- It is compiled by `setup.py` into the `faco_opt` module.

## How the Files Work Together at Runtime

### Build Time

1. `setup.py` asks setuptools to build extension `faco_opt`.
2. The compiler compiles `binding.cpp` and `mfaco_train.cpp`.
3. `binding.cpp` includes `mfaco_train.h`.
4. `mfaco_train.cpp` includes `mfaco_train.h` and `kd_tree.h`.
5. The compiled shared library becomes importable from Python.

### Python Construction Time

1. Python imports `faco_opt`.
2. Python creates a solver, such as `faco_opt.MFACO_CVRP(...)`.
3. The pybind wrapper validates NumPy shapes.
4. The wrapper passes raw array pointers and parameters into the C++ constructor.
5. The C++ solver copies data, builds nearest-neighbor lists, initializes heuristics, and initializes pheromone.

### Sampling/Training Time

1. Python calls `solver.sample(...)`.
2. `binding.cpp` converts Python inputs such as prior/probability arrays into raw pointers.
3. `mfaco_train.cpp` constructs routes with ants using pheromone, heuristic values, candidate lists, and optional learned priors.
4. The solver records costs, routes, and optional traces.
5. `binding.cpp` converts results back into Python dictionaries/arrays.
6. Python can use returned traces for neural training or policy replay.
7. Python can call pheromone update methods to feed improved solutions back into the C++ solver.

## Dependency Summary

| File | Depends On | Used By | Role |
|---|---|---|---|
| `setup.py` | setuptools, pybind11 | build/install command | Builds `faco_opt` extension |
| `binding.cpp` | `mfaco_train.h`, pybind11, OpenMP | Python import/runtime | Python wrapper layer |
| `mfaco_train.h` | C++ standard library | `binding.cpp`, `mfaco_train.cpp` | Shared declarations and data structures |
| `mfaco_train.cpp` | `mfaco_train.h`, `kd_tree.h`, C++ STL/OpenMP | `binding.cpp` through compiled symbols | Algorithm implementation |
| `kd_tree.h` | C++ standard library | `mfaco_train.cpp` | Nearest-neighbor acceleration |

## Key Data Flow

```text
NumPy arrays from Python
   -> binding.cpp validates and passes raw pointers
   -> mfaco_train.cpp constructors copy/setup problem data
   -> kd_tree.h helps build candidate neighbor lists
   -> mfaco_train.cpp samples ants and updates pheromone
   -> result structs/traces are filled
   -> binding.cpp exposes vectors as NumPy views or Python dicts
   -> Python training loop consumes routes, costs, pheromone, and traces
```

## Practical Notes

- The binding layer exposes many internal vectors as zero-copy NumPy views. This is fast, but the view depends on the lifetime of the owning C++ solver object.
- Candidate-list size is bounded by `MAX_CAND_LIST_SIZE`, and masks such as `valid_mask` assume compact candidate rows.
- OpenMP is enabled at build time and exposed at runtime through helper functions in `binding.cpp`.
- `MFACO_*` classes are designed for training workflows where Python may provide learned priors and consume traces.
- `ACO_*` classes are more conventional ACO/MMAS baselines that construct solutions from pheromone and heuristic information.
