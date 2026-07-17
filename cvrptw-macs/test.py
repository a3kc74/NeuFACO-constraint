import argparse
import contextlib
import csv
import io
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from multiple_ant_colony_system import MultipleAntColonySystem
from vrptw_base import Node, VrptwGraph


def dataset_mode_prefix(tam=False, vrptw=False):
    if tam and vrptw:
        return "tam-vrptw-"
    if tam:
        return "tam-"
    if vrptw:
        return "vrptw-"
    return ""


def default_data_dir():
    return Path(__file__).resolve().parents[1] / "data" / "cvrptw"


def default_result_dir(n_nodes, tam=False, vrptw=False):
    problem = f"{'tam-' if tam else ''}{'vrptw' if vrptw else 'cvrptw'}"
    return Path(__file__).resolve().parents[1] / "pretrained" / "macs" / problem / str(n_nodes)


def load_gfacs_test_dataset(n_nodes, data_dir=None, tam=False, vrptw=False, map_location="cpu"):
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    filename = data_dir / f"testDataset-{dataset_mode_prefix(tam, vrptw)}{n_nodes}.pt"
    if not filename.is_file():
        raise FileNotFoundError(f"File {filename} not found. Generate it with cvrptw-gfacs/utils.py first.")
    return torch.load(filename, map_location=map_location)


def graph_from_gfacs_instance(instance, capacity=1.0, service_time=0.0, rho=0.1):
    instance = torch.as_tensor(instance, dtype=torch.double).cpu()
    node_num = int(instance.shape[1])
    demands = instance[0].numpy()
    positions = instance[1:3].T.numpy()
    distances = instance[3:-2].numpy()
    windows = instance[-2:].T.numpy()

    graph = object.__new__(VrptwGraph)
    graph.node_num = node_num
    graph.nodes = [
        Node(
            id=index,
            x=float(positions[index, 0]),
            y=float(positions[index, 1]),
            demand=float(demands[index]),
            ready_time=float(windows[index, 0]),
            due_time=float(windows[index, 1]),
            service_time=float(service_time),
        )
        for index in range(node_num)
    ]
    graph.node_dist_mat = np.asarray(distances, dtype=float).copy()
    np.fill_diagonal(graph.node_dist_mat, 1e-8)
    graph.vehicle_num = node_num
    graph.vehicle_capacity = float(capacity)
    graph.rho = float(rho)

    graph.nnh_travel_path, graph.init_pheromone_val, _ = graph.nearest_neighbor_heuristic()
    if graph.init_pheromone_val <= 0:
        graph.init_pheromone_val = 1e-8
    graph.init_pheromone_val = 1 / (graph.init_pheromone_val * graph.node_num)
    graph.pheromone_mat = np.ones((graph.node_num, graph.node_num)) * graph.init_pheromone_val
    graph.heuristic_info_mat = 1 / graph.node_dist_mat
    return graph


def run_instance(instance, ants=10, beta=1, q0=0.1, time_limit=600.0, seed=0, quiet=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    graph = graph_from_gfacs_instance(instance)
    macs = MultipleAntColonySystem(
        graph,
        ants_num=ants,
        beta=beta,
        q0=q0,
        whether_or_not_to_show_figure=False,
    )

    started = time.time()
    stream = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(stream):
        path, cost, vehicles = macs.run_multiple_ant_colony_system(
            time_limit_seconds=time_limit,
            use_process=False,
        )
    elapsed = time.time() - started
    return {
        "path": path,
        "cost": float(cost),
        "vehicles": int(vehicles),
        "time": elapsed,
    }


def write_results(summary, rows, result_dir, result_name):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_txt = result_dir / f"{result_name}.txt"
    result_csv = result_dir / f"{result_name}.csv"

    with result_txt.open("w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    with result_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance", "cost", "vehicles", "time"])
        writer.writeheader()
        writer.writerows(rows)

    return result_txt, result_csv


def main(
    n_nodes,
    size=None,
    ants=10,
    seed=0,
    time_limit=600.0,
    tam=False,
    vrptw=False,
    data_dir=None,
    result_dir=None,
    beta=1,
    q0=0.1,
    quiet=False,
):
    dataset = load_gfacs_test_dataset(n_nodes, data_dir=data_dir, tam=tam, vrptw=vrptw)
    dataset = dataset[: size or len(dataset)]
    problem = f"{'tam-' if tam else ''}{'vrptw' if vrptw else 'cvrptw'}"

    rows = []
    for index, instance in enumerate(dataset):
        print(f"Running instance {index}...")
        result = run_instance(
            instance,
            ants=ants,
            beta=beta,
            q0=q0,
            time_limit=time_limit,
            seed=seed + index,
            quiet=quiet,
        )
        rows.append({
            "instance": index,
            "cost": result["cost"],
            "vehicles": result["vehicles"],
            "time": result["time"],
        })

    avg_cost = float(np.mean([row["cost"] for row in rows])) if rows else float("nan")
    avg_vehicles = float(np.mean([row["vehicles"] for row in rows])) if rows else float("nan")
    avg_time = float(np.mean([row["time"] for row in rows])) if rows else float("nan")
    summary = {
        "algorithm": "macs",
        "problem": problem,
        "problem scale": n_nodes,
        "number of instances": len(rows),
        "instances": len(rows),
        "ants": ants,
        "beta": beta,
        "q0": q0,
        "seed": seed,
        "time_limit_seconds": time_limit,
        "average inference time": avg_time,
        "avg_time": avg_time,
        "avg_cost": avg_cost,
        "avg_vehicles": avg_vehicles,
    }

    if not quiet:
        for key, value in summary.items():
            print(f"{key}: {value}")

    if result_dir is None:
        result_dir = default_result_dir(n_nodes, tam=tam, vrptw=vrptw)
    result_name = f"test_result_macs-{problem}{n_nodes}-ninst{len(rows)}-nants{ants}-timelimit{time_limit}-seed{seed}"
    result_txt, result_csv = write_results(summary, rows, result_dir, result_name)
    return summary, result_txt, result_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-s", "--size", type=int, default=None, help="Number of instances to test")
    parser.add_argument("-n", "--ants", type=int, default=10, help="Number of ants")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--time-limit", type=float, default=600.0, help="Per-instance MACS time limit in seconds")
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")
    parser.add_argument("--vrptw", action="store_true", help="Use VRPTW data with all customer demands set to zero")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing GFACS testDataset-*.pt files")
    parser.add_argument("--result-dir", type=Path, default=None, help="Directory for MACS benchmark outputs")
    parser.add_argument("--beta", type=float, default=1, help="MACS beta parameter")
    parser.add_argument("--q0", type=float, default=0.1, help="MACS q0 parameter")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-instance MACS logs")
    args = parser.parse_args()

    main(
        n_nodes=args.nodes,
        size=args.size,
        ants=args.ants,
        seed=args.seed,
        time_limit=args.time_limit,
        tam=args.tam,
        vrptw=args.vrptw,
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        beta=args.beta,
        q0=args.q0,
        quiet=args.quiet,
    )
