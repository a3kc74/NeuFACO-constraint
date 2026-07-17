import csv
import importlib.util
from pathlib import Path
from multiprocessing import Queue

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_macs_test_module():
    module_path = ROOT / "cvrptw-macs" / "test.py"
    spec = importlib.util.spec_from_file_location("macs_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_dataset(path: Path, n_nodes: int, n_instances: int, vrptw: bool = False):
    torch.manual_seed(321)
    instances = []
    for _ in range(n_instances):
        coords = torch.rand((n_nodes + 1, 2), dtype=torch.double) * (150.0 / 480.0)
        demand = torch.zeros(n_nodes + 1, dtype=torch.double)
        if not vrptw:
            demand[1:] = torch.randint(1, 4, (n_nodes,), dtype=torch.double) / 20.0
        dist = torch.norm(coords[:, None] - coords[None, :], dim=2, p=2)
        dist[torch.arange(n_nodes + 1), torch.arange(n_nodes + 1)] = 1e-10
        windows = torch.zeros((n_nodes + 1, 2), dtype=torch.double)
        windows[:, 1] = float(n_nodes)
        instances.append(torch.vstack([demand, coords.T, dist, windows.T]))
    torch.save(torch.stack(instances), path)


def run_case(tmp_path, vrptw):
    macs_test = load_macs_test_module()
    data_dir = tmp_path / "data" / "cvrptw"
    result_dir = tmp_path / "results"
    data_dir.mkdir(parents=True)
    prefix = "vrptw-" if vrptw else ""
    make_dataset(data_dir / f"testDataset-{prefix}5.pt", n_nodes=5, n_instances=2, vrptw=vrptw)

    dataset = macs_test.load_gfacs_test_dataset(5, data_dir=data_dir, vrptw=vrptw)
    input_file = macs_test.write_macs_input_file(dataset[0], tmp_path / "instance0.txt")
    graph = macs_test.VrptwGraph(input_file)
    assert graph.node_num == 6
    assert graph.vehicle_capacity == 1
    assert graph.nodes[0].is_depot
    assert graph.nodes[0].due_time == 5.0

    summary, result_txt, result_csv = macs_test.main(
        n_nodes=5,
        size=2,
        ants=2,
        seed=7,
        time_limit=0.0,
        data_dir=data_dir,
        result_dir=result_dir,
        vrptw=vrptw,
        quiet=True,
    )

    assert summary["instances"] == 2
    assert summary["avg_cost"] > 0
    assert summary["avg_vehicles"] >= 1
    assert summary["avg_time"] >= 0
    assert result_txt.exists()
    assert result_csv.exists()
    text = result_txt.read_text()
    assert f"problem: {'vrptw' if vrptw else 'cvrptw'}" in text
    assert "algorithm: macs" in text

    with result_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert set(rows[0]) == {"instance", "cost", "vehicles", "time"}


def test_macs_test_outputs_gfacs_format_for_cvrptw(tmp_path):
    run_case(tmp_path, vrptw=False)


def test_macs_test_outputs_gfacs_format_for_vrptw(tmp_path):
    run_case(tmp_path, vrptw=True)

def test_extracts_vehicle_improvement_announcements(tmp_path):
    macs_test = load_macs_test_module()
    log_file = tmp_path / "macs.log"
    log_file.write_text(
        "[macs]: vehicle num of found path (13) better than best path's (14), found path distance is 9.501108\n"
        "it takes 56.538 second multiple_ant_colony_system running\n"
        "other line\n"
    )

    assert macs_test.extract_quiet_announcements(log_file) == [
        "[macs]: vehicle num of found path (13) better than best path's (14), found path distance is 9.501108",
        "it takes 56.538 second multiple_ant_colony_system running",
    ]

def test_live_quiet_stream_captures_vehicle_improvements():
    macs_test = load_macs_test_module()
    queue = Queue()
    stream = macs_test.LiveMacsAnnouncementStream(queue)

    stream.write("[macs]: vehicle num of found path (13) better than best path's (14), found path distance is 9.501108\n")
    stream.write("it takes 56.538 second multiple_ant_colony_system running\n")
    stream.write("[acs_time]: new iteration\n")

    assert queue.get(timeout=1) == "[macs]: vehicle num of found path (13) better than best path's (14), found path distance is 9.501108"
    assert queue.get(timeout=1) == "it takes 56.538 second multiple_ant_colony_system running"
    assert queue.empty()
