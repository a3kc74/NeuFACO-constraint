import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cvrptw-faco"))

from faco_test import main


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
