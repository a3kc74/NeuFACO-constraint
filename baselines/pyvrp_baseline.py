from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def preload_pyvrp_root() -> None:
    explicit_root = os.environ.get("PYVRP_ROOT")
    for idx, arg in enumerate(sys.argv):
        if arg == "--pyvrp_root" and idx + 1 < len(sys.argv):
            explicit_root = sys.argv[idx + 1]
            break
        if arg.startswith("--pyvrp_root="):
            explicit_root = arg.split("=", 1)[1]
            break
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


preload_pyvrp_root()

from envs.gfacs_data import load_test_dataset, load_val_dataset

try:
    from pyvrp import Client, Depot, Location, Model, ProblemData, VehicleType
except ImportError:
    from pyvrp import Client, Depot, Model, ProblemData, VehicleType

    Location = None

from pyvrp.stop import MaxIterations

try:
    from pyvrp.stop import MaxRuntime
except ImportError:  # pragma: no cover - depends on PyVRP version
    MaxRuntime = None

SCALE = 10**4
CAPACITY = 600


def load_gfacs_split(args: argparse.Namespace):
    if args.type == "val":
        dataset = load_val_dataset(args.nodes, args.k_sparse, "cpu", tam=args.tam, vrptw=args.vrptw)
    else:
        dataset = load_test_dataset(
            args.nodes,
            args.k_sparse,
            "cpu",
            tam=args.tam,
            vrptw=args.vrptw,
            data_dir=args.data_dir if args.data_dir is not None else ROOT / "data" / "cvrptw",
        )
    return dataset[: args.size]

def convert_instance(item):
    _pyg_data, demands, distances, positions, windows = item
    positions_int = np.rint(positions.numpy() * SCALE).astype(np.int64)
    distances_int = np.rint(distances.numpy() * SCALE).astype(np.int64)
    windows_int = np.rint(windows.numpy() * SCALE).astype(np.int64)
    demands_int = np.rint(demands[1:].numpy() * CAPACITY).astype(np.int64)
    np.fill_diagonal(distances_int, 0)
    return positions_int, demands_int, distances_int, windows_int


def make_problem_data(positions, demands, distances, windows) -> ProblemData:
    duration_matrix = distances.copy()
    if Location is not None:
        locations = [Location(int(x), int(y)) for x, y in positions]
        clients = [
            Client(
                location=idx,
                delivery=[int(demand)],
                tw_early=int(window[0]),
                tw_late=int(window[1]),
            )
            for idx, (demand, window) in enumerate(zip(demands, windows[1:]), start=1)
        ]
        depots = [
            Depot(
                location=0,
                tw_early=int(windows[0][0]),
                tw_late=int(windows[0][1]),
            )
        ]
        vehicle_types = [
            VehicleType(
                num_available=len(positions) - 1,
                capacity=[CAPACITY],
                start_depot=0,
                end_depot=0,
            )
        ]
        return ProblemData(
            locations=locations,
            clients=clients,
            depots=depots,
            vehicle_types=vehicle_types,
            distance_matrices=[distances],
            duration_matrices=[duration_matrix],
        )

    clients = [
        Client(
            x=int(pos[0]),
            y=int(pos[1]),
            delivery=int(demand),
            tw_early=int(window[0]),
            tw_late=int(window[1]),
        )
        for pos, demand, window in zip(positions[1:], demands, windows[1:])
    ]
    depots = [
        Depot(
            x=int(positions[0][0]),
            y=int(positions[0][1]),
            tw_early=int(windows[0][0]),
            tw_late=int(windows[0][1]),
        )
    ]
    vehicle_types = [VehicleType(len(positions) - 1, CAPACITY, 0)]
    return ProblemData(
        clients=clients,
        depots=depots,
        vehicle_types=vehicle_types,
        distance_matrix=distances,
        duration_matrix=duration_matrix,
    )


def build_stop(args: argparse.Namespace):
    if args.max_runtime is not None:
        if MaxRuntime is None:
            raise RuntimeError("Installed PyVRP does not provide MaxRuntime")
        return MaxRuntime(args.max_runtime)
    return MaxIterations(args.maxiter)


def route_count(result: Any) -> int | None:
    best = getattr(result, "best", None)
    if best is None:
        return None
    num_routes = getattr(best, "num_routes", None)
    return int(num_routes()) if callable(num_routes) else None


def is_feasible(result: Any) -> bool:
    feasible = getattr(result, "is_feasible", None)
    if callable(feasible):
        return bool(feasible())
    best = getattr(result, "best", None)
    best_feasible = getattr(best, "is_feasible", None)
    return bool(best_feasible()) if callable(best_feasible) else math.isfinite(float(result.cost()))


def solve_one(payload):
    idx, item, args = payload
    positions, demands, distances, windows = convert_instance(item)
    data = make_problem_data(positions, demands, distances, windows)
    model = Model.from_data(data)
    start = time.time()
    result = model.solve(stop=build_stop(args), seed=args.seed + idx, display=False)
    wall_time = time.time() - start
    return {
        "instance": idx,
        "cost": float(result.cost()) / SCALE,
        "runtime": float(getattr(result, "runtime", wall_time)),
        "wall_time": wall_time,
        "feasible": is_feasible(result),
        "num_routes": route_count(result),
        "seed": args.seed + idx,
        "maxiter": args.maxiter,
        "max_runtime": args.max_runtime if args.max_runtime is not None else "",
    }


def result_paths(args: argparse.Namespace, size: int) -> tuple[Path, Path]:
    problem = f"{'tam-' if args.tam else ''}{'vrptw' if args.vrptw else 'cvrptw'}"
    out_dir = Path(args.result_dir) / problem / str(args.nodes)
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = f"maxruntime{args.max_runtime}" if args.max_runtime is not None else f"maxiter{args.maxiter}"
    name = f"test_result_pyvrp-{args.type}-{problem}{args.nodes}-ninst{size}-{budget}-seed{args.seed}"
    return out_dir / f"{name}.txt", out_dir / f"{name}.csv"


def write_results(args: argparse.Namespace, rows: list[dict[str, Any]], duration: float) -> tuple[Path, Path]:
    costs = np.asarray([row["cost"] for row in rows], dtype=np.float64)
    runtimes = np.asarray([row["runtime"] for row in rows], dtype=np.float64)
    feasible = np.asarray([row["feasible"] for row in rows], dtype=bool)
    result_txt, result_csv = result_paths(args, len(rows))

    with result_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "instance",
                "cost",
                "runtime",
                "wall_time",
                "feasible",
                "num_routes",
                "seed",
                "maxiter",
                "max_runtime",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["instance"]))

    with result_txt.open("w") as file:
        file.write("[PyVRP]\n")
        file.write(f"problem scale: {args.nodes}\n")
        file.write(f"dataset type: {args.type}\n")
        file.write(f"number of instances: {len(rows)}\n")
        file.write(f"k_sparse loader value: {args.k_sparse}\n")
        file.write(f"seed: {args.seed}\n")
        file.write(f"maxiter: {args.maxiter}\n")
        file.write(f"max_runtime: {args.max_runtime}\n")
        file.write(f"n_cpus: {args.n_cpus}\n")
        file.write(f"average cost: {float(costs.mean())}\n")
        file.write(f"std cost: {float(costs.std(ddof=0))}\n")
        file.write(f"average runtime: {float(runtimes.mean())}\n")
        file.write(f"std runtime: {float(runtimes.std(ddof=0))}\n")
        file.write(f"feasible rate: {float(feasible.mean())}\n")
        file.write(f"total wall time: {duration}\n")
        file.write(f"total solver runtime: {timedelta(seconds=int(runtimes.sum()))}\n")

    return result_txt, result_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PyVRP on GFACS-format CVRPTW datasets.")
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="Sparse graph size used by GFACS dataset loader")
    parser.add_argument("-s", "--size", type=int, default=None, help="Number of instances to solve")
    parser.add_argument("-i", "--maxiter", type=int, default=20000, help="PyVRP MaxIterations budget")
    parser.add_argument("--max_runtime", type=float, default=None, help="Optional PyVRP MaxRuntime budget in seconds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--type", choices=["test", "val"], default="test", help="Dataset split")
    parser.add_argument("--data_dir", type=Path, default=None, help="Directory containing GFACS-format datasets")
    parser.add_argument("--pyvrp_root", type=Path, default=None, help="Optional local PyVRP checkout inserted before importing pyvrp")
    parser.add_argument("--result_dir", type=Path, default=ROOT / "pretrained" / "pyvrp", help="Result directory")
    parser.add_argument("--n_cpus", type=int, default=1, help="Number of parallel worker threads")
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")
    parser.add_argument("--vrptw", action="store_true", help="Use VRPTW data with all customer demands set to zero")
    args = parser.parse_args()
    if args.k_sparse is None:
        args.k_sparse = args.nodes // 5
    if args.size is not None and args.size < 1:
        raise ValueError("--size must be positive")
    if args.n_cpus < 1:
        raise ValueError("--n_cpus must be positive")
    return args


def main() -> None:
    args = parse_args()
    dataset = load_gfacs_split(args)
    start = time.time()
    payloads = [(idx, item, args) for idx, item in enumerate(dataset)]
    if args.n_cpus == 1:
        rows = [solve_one(payload) for payload in tqdm(payloads, dynamic_ncols=True)]
    else:
        rows = []
        with tqdm(total=len(payloads), dynamic_ncols=True) as progress, ThreadPoolExecutor(max_workers=args.n_cpus) as executor:
            futures = [executor.submit(solve_one, payload) for payload in payloads]
            for future in as_completed(futures):
                rows.append(future.result())
                progress.update(1)
    duration = time.time() - start
    result_txt, result_csv = write_results(args, rows, duration)

    costs = np.asarray([row["cost"] for row in rows], dtype=np.float64)
    runtimes = np.asarray([row["runtime"] for row in rows], dtype=np.float64)
    feasible = np.asarray([row["feasible"] for row in rows], dtype=bool)
    print(f"Average cost: {float(costs.mean())} +- {float(costs.std(ddof=0))}")
    print(f"Average runtime: {float(runtimes.mean())} +- {float(runtimes.std(ddof=0))}")
    print(f"Feasible rate: {float(feasible.mean())}")
    print(f"Total wall time: {timedelta(seconds=int(duration))}")
    print(f"Wrote: {result_txt}")
    print(f"Wrote: {result_csv}")


if __name__ == "__main__":
    main()





