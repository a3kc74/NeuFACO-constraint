from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import numpy as np

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.cvrptw_env import generate_cvrptw_instance
from solvers.faco import MFACO_CVRPTW, set_faco_cpp_threads


def route_distance(coords: np.ndarray, route: np.ndarray) -> float:
    route = np.asarray(route, dtype=np.int64)
    return float(np.linalg.norm(coords[route[:-1]] - coords[route[1:]], axis=1).sum())


def is_feasible(coords: np.ndarray, demand: np.ndarray, windows: np.ndarray, capacity: float, route: np.ndarray) -> bool:
    load = 0.0
    current_time = 0.0
    prev = 0
    seen: list[int] = []
    for raw_node in np.asarray(route, dtype=np.int64)[1:]:
        node = int(raw_node)
        arrival = current_time + float(np.linalg.norm(coords[prev] - coords[node]))
        if node == 0:
            if arrival > float(windows[0, 1]) + 1e-5:
                return False
            load = 0.0
            current_time = 0.0
        else:
            if arrival > float(windows[node, 1]) + 1e-5:
                return False
            current_time = max(arrival, float(windows[node, 0]))
            load += float(demand[node])
            if load > capacity + 1e-5:
                return False
            if current_time + float(np.linalg.norm(coords[node] - coords[0])) > float(windows[0, 1]) + 1e-5:
                return False
            seen.append(node)
        prev = node
    return sorted(seen) == list(range(1, len(coords)))


def run_case(coords: np.ndarray, demand: np.ndarray, windows: np.ndarray, capacity: float, use_fts: bool, args: argparse.Namespace) -> dict[str, float]:
    solver = MFACO_CVRPTW(
        coords,
        demand,
        windows,
        capacity,
        n_ants=args.ants,
        cand_list_size=args.cand_list_size,
        backup_list_size=args.backup_list_size,
        min_new_edges=args.min_new_edges,
        use_local_search=True,
        use_fts_checks=use_fts,
    )
    solver.seed_rng(args.solver_seed)
    solver.reset_timings()

    best_cost = math.inf
    mean_costs: list[float] = []
    infeasible = 0
    start = time.perf_counter()
    for _ in range(args.samples):
        costs, routes, *_ = solver.sample()
        best_cost = min(best_cost, float(np.min(costs)))
        mean_costs.append(float(np.mean(costs)))
        for route in routes:
            if not is_feasible(coords, demand, windows, capacity, route):
                infeasible += 1
    wall_time = time.perf_counter() - start
    timings = solver.get_timings()
    return {
        "wall_time": wall_time,
        "time_ant": float(timings.get("time_ant", 0.0)),
        "time_ls": float(timings.get("time_ls", 0.0)),
        "time_split": float(timings.get("time_split", 0.0)),
        "fts_checks": float(timings.get("fts_checks", 0.0)),
        "fts_fallback_scans": float(timings.get("fts_fallback_scans", 0.0)),
        "best_cost": best_cost,
        "mean_cost": statistics.mean(mean_costs),
        "infeasible": float(infeasible),
    }


def speedup(base: float, fts: float) -> float:
    return base / fts if fts > 0 else math.inf


def fmt(value: float) -> str:
    if math.isinf(value) or math.isnan(value):
        return "n/a"
    return f"{value:.6f}"


def write_report(path: Path, args: argparse.Namespace, baseline: dict[str, float], fts: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in ("wall_time", "time_ant", "time_ls", "time_split", "fts_checks", "fts_fallback_scans", "best_cost", "mean_cost", "infeasible"):
        rows.append((key, baseline[key], fts[key], speedup(baseline[key], fts[key]) if key.startswith("time") or key == "wall_time" else fts[key] - baseline[key]))

    lines = [
        "# FTS CVRPTW Benchmark",
        "",
        f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Nodes: `{args.nodes}`",
        f"- Ants: `{args.ants}`",
        f"- Samples: `{args.samples}`",
        f"- Instance seed: `{args.instance_seed}`",
        f"- Solver seed: `{args.solver_seed}`",
        f"- Threads: `{args.threads}`",
        "",
        "## Results",
        "",
        "| Metric | Baseline scan | FTS checks | Speedup / Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, base_value, fts_value, delta in rows:
        lines.append(f"| {key} | {fmt(base_value)} | {fmt(fts_value)} | {fmt(delta)} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Baseline uses `use_fts_checks=False`; FTS uses `use_fts_checks=True` with the same binary.",
        "- FTS uses metadata fast-paths for ant relocation insert/remove checks plus optimized in-place fallback simulation; remaining LS move cases keep scan fallback because all-move FTS scans were slower on this workload.",
        "- `time_ls` and `time_split` are currently not instrumented separately in the backend, so this report uses `time_ant` and wall time for efficiency.",
        "- `infeasible` counts independently verified sampled routes; expected value is `0`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CVRPTW FTS checks against scan fallback.")
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--ants", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--instance-seed", type=int, default=100)
    parser.add_argument("--solver-seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cand-list-size", type=int, default=32)
    parser.add_argument("--backup-list-size", type=int, default=64)
    parser.add_argument("--min-new-edges", type=int, default=8)
    parser.add_argument("--report-path", type=Path, default=Path("reports/fts_cvrptw_n100_a100.md"))
    args = parser.parse_args()

    set_faco_cpp_threads(args.threads)
    instance = generate_cvrptw_instance(args.nodes, seed=args.instance_seed)
    coords = np.asarray(instance["coords"].cpu().numpy(), dtype=np.float32)
    demand = np.asarray(instance["demand"].cpu().numpy(), dtype=np.float32)
    windows = np.asarray(instance["windows"].cpu().numpy(), dtype=np.float32)
    capacity = float(instance["capacity"])

    baseline = run_case(coords, demand, windows, capacity, False, args)
    fts = run_case(coords, demand, windows, capacity, True, args)
    write_report(args.report_path, args, baseline, fts)
    print(f"Wrote {args.report_path}")
    print(f"wall speedup: {speedup(baseline['wall_time'], fts['wall_time']):.3f}x")
    print(f"LS speedup: {speedup(baseline['time_ls'], fts['time_ls']):.3f}x")


if __name__ == "__main__":
    main()
