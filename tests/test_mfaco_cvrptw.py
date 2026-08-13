import math
from pathlib import Path

import numpy as np
import pytest

from solvers.faco import MFACO_CVRPTW


def route_distance(coords, route):
    return sum(float(np.linalg.norm(coords[int(a)] - coords[int(b)])) for a, b in zip(route, route[1:]))


def assert_cvrptw_solution(coords, demand, windows, capacity, costs, routes):
    n = len(coords)
    for cost, route in zip(costs, routes):
        route = np.asarray(route, dtype=np.int32)
        assert route[0] == 0
        assert route[-1] == 0
        customers = route[route != 0]
        assert sorted(customers.tolist()) == list(range(1, n))

        load = 0.0
        time = 0.0
        prev = 0
        for node in route[1:]:
            node = int(node)
            travel = float(np.linalg.norm(coords[prev] - coords[node]))
            arrival = time + travel
            if node == 0:
                assert arrival <= float(windows[0, 1]) + 1e-5
                load = 0.0
                time = 0.0
            else:
                assert arrival <= float(windows[node, 1]) + 1e-5
                time = max(arrival, float(windows[node, 0]))
                load += float(demand[node])
                assert load <= capacity + 1e-5
                assert time + float(np.linalg.norm(coords[node] - coords[0])) <= float(windows[0, 1]) + 1e-5
            prev = node

        assert math.isclose(float(cost), route_distance(coords, route), rel_tol=1e-5, abs_tol=1e-5)


def gfacs_instance(n=20):
    rng = np.random.default_rng(1234)
    coords = rng.random((n + 1, 2), dtype=np.float32) * (150.0 / 480.0)
    demand = np.zeros(n + 1, dtype=np.float32)
    dist_to_depot = np.linalg.norm(coords - coords[0], axis=1)
    upper_bound = 1.0 - dist_to_depot
    ts_1 = rng.random(n + 1, dtype=np.float32)
    ts_2 = rng.random(n + 1, dtype=np.float32)
    min_ts = dist_to_depot + (upper_bound - dist_to_depot) * ts_1
    max_ts = dist_to_depot + (upper_bound - dist_to_depot) * ts_2
    windows = np.stack([np.minimum(min_ts, max_ts), np.maximum(min_ts, max_ts)], axis=-1).astype(np.float32)
    windows[0, 0] = 0.0
    windows[0, 1] = float(n)
    return coords, demand, windows, 1.0


def test_cvrptw_samples_gfacs_instance_without_prior():
    coords, demand, windows, capacity = gfacs_instance(20)
    solver = MFACO_CVRPTW(coords, demand, windows, capacity, n_ants=4, use_local_search=False, cand_list_size=8)
    solver.seed_rng(7)

    costs, routes, *_ = solver.sample(prior=None)

    assert_cvrptw_solution(coords, demand, windows, capacity, costs, routes)


def test_cvrptw_accepts_valid_prior():
    coords, demand, windows, capacity = gfacs_instance(20)
    solver = MFACO_CVRPTW(coords, demand, windows, capacity, n_ants=2, use_local_search=False, cand_list_size=8)
    prior = np.ones((solver.n, solver.k), dtype=np.float32)

    costs, routes, *_ = solver.sample(prior=prior)

    assert_cvrptw_solution(coords, demand, windows, capacity, costs, routes)


def test_cvrptw_local_search_preserves_time_windows():
    coords, demand, windows, capacity = gfacs_instance(20)
    solver = MFACO_CVRPTW(coords, demand, windows, capacity, n_ants=4, use_local_search=True, cand_list_size=8)
    solver.seed_rng(11)

    costs, routes, *_ = solver.sample()

    assert_cvrptw_solution(coords, demand, windows, capacity, costs, routes)


def test_cvrptw_local_search_preserves_tight_time_windows_across_seeds():
    coords, demand, windows, capacity = gfacs_instance(32)

    for seed in range(10):
        solver = MFACO_CVRPTW(coords, demand, windows, capacity, n_ants=6, use_local_search=True, cand_list_size=12)
        solver.seed_rng(seed)

        costs, routes, *_ = solver.sample()

        assert_cvrptw_solution(coords, demand, windows, capacity, costs, routes)

def test_cvrptw_fts_checks_match_scan_fallback_feasibility():
    coords, demand, windows, capacity = gfacs_instance(32)

    for use_fts_checks in (False, True):
        solver = MFACO_CVRPTW(
            coords,
            demand,
            windows,
            capacity,
            n_ants=8,
            use_local_search=True,
            cand_list_size=12,
            use_fts_checks=use_fts_checks,
        )
        solver.seed_rng(17)

        costs, routes, *_ = solver.sample()

        assert_cvrptw_solution(coords, demand, windows, capacity, costs, routes)


def test_cvrptw_segment_local_search_preserves_tight_capacity_and_windows():
    coords = np.array(
        [
            [0.0, 0.0],
            [0.10, 0.00],
            [0.20, 0.00],
            [0.30, 0.00],
            [0.00, 0.10],
            [0.00, 0.20],
            [0.00, 0.30],
            [0.30, 0.30],
            [0.20, 0.30],
            [0.10, 0.30],
        ],
        dtype=np.float32,
    )
    demand = np.array([0.0, 0.35, 0.25, 0.30, 0.40, 0.20, 0.35, 0.25, 0.35, 0.20], dtype=np.float32)
    windows = np.array(
        [
            [0.0, 3.0],
            [0.0, 0.75],
            [0.0, 0.85],
            [0.0, 1.00],
            [0.0, 0.75],
            [0.0, 0.90],
            [0.0, 1.10],
            [0.0, 1.20],
            [0.0, 1.10],
            [0.0, 0.95],
        ],
        dtype=np.float32,
    )
    capacity = 0.75

    for seed in range(12):
        solver = MFACO_CVRPTW(coords, demand, windows, capacity, n_ants=8, use_local_search=True, cand_list_size=9)
        solver.seed_rng(seed)

        costs, routes, *_ = solver.sample()

        assert_cvrptw_solution(coords, demand, windows, capacity, costs, routes)


def test_cvrptw_initial_routes_are_time_window_feasible():
    coords, demand, windows, capacity = gfacs_instance(20)
    solver = MFACO_CVRPTW(coords, demand, windows, capacity, n_ants=2, use_local_search=False, cand_list_size=8)

    source = np.asarray(solver.source_route, dtype=np.int32)
    best = np.asarray(solver.best_route, dtype=np.int32)

    assert_cvrptw_solution(coords, demand, windows, capacity, [route_distance(coords, source)], [source])
    assert_cvrptw_solution(coords, demand, windows, capacity, [route_distance(coords, best)], [best])

def test_cvrptw_rejects_infeasible_pheromone_update_route():
    coords, demand, windows, capacity = gfacs_instance(20)
    solver = MFACO_CVRPTW(coords, demand, windows, capacity, n_ants=2, use_local_search=False, cand_list_size=8)
    original_source = np.asarray(solver.source_route, dtype=np.int32).copy()
    original_best = np.asarray(solver.best_route, dtype=np.int32).copy()

    infeasible = np.arange(0, len(coords), dtype=np.int32)
    infeasible = np.concatenate([infeasible, np.array([0], dtype=np.int32)])
    solver.update_pheromone(infeasible, 0.001)

    np.testing.assert_array_equal(np.asarray(solver.source_route, dtype=np.int32), original_source)
    np.testing.assert_array_equal(np.asarray(solver.best_route, dtype=np.int32), original_best)

def test_cvrptw_set_source_route_does_not_update_pheromone():
    coords, demand, windows, capacity = gfacs_instance(20)
    solver = MFACO_CVRPTW(coords, demand, windows, capacity, n_ants=2, use_local_search=False, cand_list_size=8)
    original_best = np.asarray(solver.best_route, dtype=np.int32).copy()
    original_pheromone = np.asarray(solver.pheromone_sparse, dtype=np.float32).copy()

    route = original_best.copy()
    cost = route_distance(coords, route)
    solver.set_source_route(route, cost)

    np.testing.assert_array_equal(np.asarray(solver.source_route, dtype=np.int32), route)
    np.testing.assert_allclose(np.asarray(solver.pheromone_sparse, dtype=np.float32), original_pheromone)

def test_cvrptw_validates_windows_shape():
    coords, demand, windows, capacity = gfacs_instance(20)

    with pytest.raises(ValueError, match="windows must have shape"):
        MFACO_CVRPTW(coords, demand, windows[:, :1], capacity, n_ants=1)


def test_cvrptw_rejects_invalid_prior_shape():
    coords, demand, windows, capacity = gfacs_instance(20)
    solver = MFACO_CVRPTW(coords, demand, windows, capacity, n_ants=1, cand_list_size=8)

    with pytest.raises(RuntimeError, match="prior must be shape"):
        solver.sample(prior=np.ones((solver.n, solver.k + 1), dtype=np.float32))
