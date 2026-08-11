import csv
from pathlib import Path

import numpy as np
import torch

from evaluation.faco_test import main
from evaluation import faco_test


def make_dataset(path: Path, n_nodes: int, n_instances: int, vrptw: bool = False):
    torch.manual_seed(123)
    instances = []
    for _ in range(n_instances):
        coords = torch.rand((n_nodes + 1, 2), dtype=torch.double) * (150.0 / 480.0)
        demand = torch.zeros(n_nodes + 1, dtype=torch.double)
        if not vrptw:
            demand[1:] = torch.randint(1, 10, (n_nodes,), dtype=torch.double) / 20.0
        dist = torch.norm(coords[:, None] - coords[None, :], dim=2, p=2)
        dist[torch.arange(n_nodes + 1), torch.arange(n_nodes + 1)] = 1e-10
        windows = torch.zeros((n_nodes + 1, 2), dtype=torch.double)
        windows[:, 1] = float(n_nodes)
        instances.append(torch.vstack([demand, coords.T, dist, windows.T]))
    torch.save(torch.stack(instances), path)


def run_case(tmp_path, vrptw):
    data_dir = tmp_path / "data" / "cvrptw"
    out_dir = tmp_path / "out"
    data_dir.mkdir(parents=True)
    prefix = "vrptw-" if vrptw else ""
    make_dataset(data_dir / f"testDataset-{prefix}20.pt", n_nodes=20, n_instances=2, vrptw=vrptw)

    avg_cost, avg_diversity, duration, result_txt, result_csv = main(
        n_nodes=20,
        k_sparse=4,
        size=2,
        n_ants=3,
        n_iter=2,
        seed=5,
        tam=False,
        vrptw=vrptw,
        data_dir=data_dir,
        output_dir=out_dir,
        use_local_search=False,
    )

    assert len(avg_cost) == 2
    assert len(avg_diversity) == 2
    assert duration >= 0
    assert result_txt.exists()
    assert result_csv.exists()
    text = result_txt.read_text()
    assert f"problem: {'vrptw' if vrptw else 'cvrptw'}" in text
    assert "checkpoint: none" in text
    assert "T=1, avg. cost" in text
    assert "T=2, avg. cost" in text

    with result_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert [row["T"] for row in rows] == ["1", "2"]
    assert set(rows[0]) == {"T", "avg_cost", "avg_diversity"}


def test_faco_test_outputs_gfacs_format_for_cvrptw(tmp_path):
    run_case(tmp_path, vrptw=False)


def test_faco_test_outputs_gfacs_format_for_vrptw(tmp_path):
    run_case(tmp_path, vrptw=True)

def test_faco_test_supports_mini_iterations(tmp_path):
    data_dir = tmp_path / "data" / "cvrptw"
    out_dir = tmp_path / "out"
    data_dir.mkdir(parents=True)
    make_dataset(data_dir / "testDataset-20.pt", n_nodes=20, n_instances=1)

    avg_cost, avg_diversity, duration, result_txt, result_csv = main(
        n_nodes=20,
        k_sparse=4,
        size=1,
        n_ants=3,
        n_iter=2,
        mini_H=2,
        seed=5,
        data_dir=data_dir,
        output_dir=out_dir,
        use_local_search=False,
    )

    assert len(avg_cost) == 2
    assert len(avg_diversity) == 2
    assert duration >= 0
    assert "miniH2" in result_txt.name
    text = result_txt.read_text()
    assert "mini_H: 2" in text
    assert "T=1, avg. cost" in text
    assert "T=2, avg. cost" in text

    with result_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert [row["T"] for row in rows] == ["1", "2"]

def test_faco_test_falls_back_to_global_best_source_for_last_mini_iteration(monkeypatch):
    calls = []

    class FakeSolver:
        def __init__(self, *args, **kwargs):
            self.sample_count = 0

        def seed_rng(self, seed):
            pass

        def sample(self, prior=None):
            calls.append(("sample", self.sample_count))
            routes = np.array(
                [
                    [0, 1, 0, 2, 0],
                    [0, 2, 0, 1, 0],
                ],
                dtype=np.int32,
            )
            if self.sample_count == 0:
                costs = np.array([10.0, 5.0], dtype=np.float32)
            else:
                costs = np.array([7.0, 8.0], dtype=np.float32)
            self.sample_count += 1
            return costs, routes, None, None, None, None, None, None, None

        def update_pheromone(self, route, cost):
            calls.append(("update", tuple(np.asarray(route, dtype=np.int32)), float(cost)))

        def set_source_route(self, route, cost):
            calls.append(("set_source", tuple(np.asarray(route, dtype=np.int32)), float(cost)))

    monkeypatch.setattr(faco_test, "MFACO_CVRPTW", FakeSolver)
    demands = torch.tensor([0.0, 0.1, 0.1])
    positions = torch.zeros((3, 2))
    windows = torch.tensor([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])

    faco_test.infer_instance(
        demands=demands,
        positions=positions,
        windows=windows,
        n_ants=2,
        n_iter=1,
        mini_H=2,
        threads=None,
        seed=1,
        cand_list_size=2,
        backup_list_size=2,
        min_new_edges=1,
        decay=0.9,
        alpha=1.0,
        p_best=0.05,
        use_local_search=False,
        disable_heuristic=False,
        extend_ls=False,
        smooth_mmas=False,
        fixed_steps=0,
        nls=False,
        T_nls=10,
    )

    assert calls == [
        ("sample", 0),
        ("update", (0, 2, 0, 1, 0), 5.0),
        ("set_source", (0, 2, 0, 1, 0), 5.0),
        ("sample", 1),
        ("update", (0, 1, 0, 2, 0), 7.0),
    ]

def test_elite_archive_keeps_top_diverse_routes():
    archive = []
    routes = np.array(
        [
            [0, 1, 2, 0, 3, 0],
            [0, 1, 2, 0, 3, 0],
            [0, 1, 3, 0, 2, 0],
            [0, 3, 2, 0, 1, 0],
        ],
        dtype=np.int32,
    )
    costs = np.array([10.0, 9.0, 11.0, 12.0], dtype=np.float32)

    archive = faco_test.update_elite_archive(
        archive,
        routes,
        costs,
        elite_k=3,
        elite_min_diversity=0.1,
        elite_cost_tolerance=1.5,
    )

    assert [round(item["cost"], 3) for item in archive] == [9.0, 11.0, 12.0]
    assert [tuple(item["route"].tolist()) for item in archive] == [
        (0, 1, 2, 0, 3, 0),
        (0, 1, 3, 0, 2, 0),
        (0, 3, 2, 0, 1, 0),
    ]

def test_elite_archive_accepts_cached_and_uncached_entries():
    cached_route = np.array([0, 1, 2, 0, 3, 0], dtype=np.int32)
    archive = [
        {
            "route": cached_route,
            "cost": 10.0,
            "edges": faco_test.route_edges(cached_route),
        },
        {"route": np.array([0, 1, 3, 0, 2, 0], dtype=np.int32), "cost": 11.0},
    ]
    routes = np.array([[0, 3, 2, 0, 1, 0]], dtype=np.int32)
    costs = np.array([12.0], dtype=np.float32)

    updated = faco_test.update_elite_archive(
        archive,
        routes,
        costs,
        elite_k=3,
        elite_min_diversity=0.1,
        elite_cost_tolerance=1.5,
    )

    assert all("edges" in item for item in updated)
    assert [round(item["cost"], 3) for item in updated] == [10.0, 11.0, 12.0]

def test_elite_archive_can_pin_global_best_even_when_not_diverse():
    archive = []
    routes = np.array(
        [
            [0, 1, 3, 0, 2, 0],
            [0, 3, 2, 0, 1, 0],
        ],
        dtype=np.int32,
    )
    costs = np.array([11.0, 12.0], dtype=np.float32)
    global_best = {"route": np.array([0, 1, 2, 0, 3, 0], dtype=np.int32), "cost": 10.0}

    archive = faco_test.update_elite_archive(
        archive,
        routes,
        costs,
        elite_k=2,
        elite_min_diversity=0.95,
        elite_cost_tolerance=1.5,
        pinned_elite=global_best,
    )

    assert tuple(archive[0]["route"].tolist()) == (0, 1, 2, 0, 3, 0)
    assert archive[0]["cost"] == 10.0
    assert len(archive) >= 1

def test_faco_test_uses_multisource_sampling_for_last_mini_iteration(monkeypatch):
    calls = []

    class FakeSolver:
        def __init__(self, *args, **kwargs):
            self.sample_count = 0

        def seed_rng(self, seed):
            pass

        def sample(self, prior=None):
            calls.append(("sample", self.sample_count))
            if self.sample_count == 0:
                routes = np.array(
                    [
                        [0, 1, 2, 0, 3, 0],
                        [0, 1, 3, 0, 2, 0],
                    ],
                    dtype=np.int32,
                )
                costs = np.array([10.0, 11.0], dtype=np.float32)
            elif self.sample_count == 1:
                routes = np.array(
                    [
                        [0, 3, 2, 0, 1, 0],
                        [0, 2, 1, 0, 3, 0],
                    ],
                    dtype=np.int32,
                )
                costs = np.array([12.0, 13.0], dtype=np.float32)
            else:
                routes = np.array(
                    [
                        [0, 3, 1, 0, 2, 0],
                        [0, 2, 3, 0, 1, 0],
                    ],
                    dtype=np.int32,
                )
                costs = np.array([9.0, 14.0], dtype=np.float32)
            self.sample_count += 1
            return costs, routes, None, None, None, None, None, None, None

        def update_pheromone(self, route, cost):
            calls.append(("update", tuple(np.asarray(route, dtype=np.int32)), float(cost)))

        def set_source_route(self, route, cost):
            calls.append(("set_source", tuple(np.asarray(route, dtype=np.int32)), float(cost)))

    monkeypatch.setattr(faco_test, "MFACO_CVRPTW", FakeSolver)
    demands = torch.tensor([0.0, 0.1, 0.1, 0.1])
    positions = torch.zeros((4, 2))
    windows = torch.tensor([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])

    faco_test.infer_instance(
        demands=demands,
        positions=positions,
        windows=windows,
        n_ants=2,
        n_iter=1,
        mini_H=2,
        threads=None,
        seed=1,
        cand_list_size=2,
        backup_list_size=2,
        min_new_edges=1,
        decay=0.9,
        alpha=1.0,
        p_best=0.05,
        use_local_search=False,
        disable_heuristic=False,
        extend_ls=False,
        smooth_mmas=False,
        fixed_steps=0,
        nls=False,
        T_nls=10,
        elite_k=2,
        elite_min_diversity=0.1,
        elite_cost_tolerance=1.5,
        source_count=2,
    )

    assert calls == [
        ("sample", 0),
        ("update", (0, 1, 2, 0, 3, 0), 10.0),
        ("set_source", (0, 1, 2, 0, 3, 0), 10.0),
        ("sample", 1),
        ("set_source", (0, 1, 3, 0, 2, 0), 11.0),
        ("sample", 2),
        ("update", (0, 3, 1, 0, 2, 0), 9.0),
    ]

def test_faco_test_supports_cpp_thread_count(tmp_path):
    data_dir = tmp_path / "data" / "cvrptw"
    out_dir = tmp_path / "out"
    data_dir.mkdir(parents=True)
    make_dataset(data_dir / "testDataset-20.pt", n_nodes=20, n_instances=1)

    avg_cost, avg_diversity, duration, result_txt, result_csv = main(
        n_nodes=20,
        k_sparse=4,
        size=1,
        n_ants=3,
        n_iter=1,
        threads=1,
        seed=5,
        data_dir=data_dir,
        output_dir=out_dir,
        use_local_search=False,
    )

    assert len(avg_cost) == 1
    assert len(avg_diversity) == 1
    assert duration >= 0
    assert result_csv.exists()
    assert "threads1" in result_txt.name
    assert "threads: 1" in result_txt.read_text()
