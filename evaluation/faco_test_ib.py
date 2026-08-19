from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

from envs.cvrptw_env import gen_pyg_data, normalize_instance
from models.faco_net import Net
from solvers.faco import MFACO_CVRPTW, set_faco_cpp_threads


def dataset_mode_prefix(tam: bool = False, vrptw: bool = False) -> str:
    if tam and vrptw:
        return "tam-vrptw-"
    if tam:
        return "tam-"
    if vrptw:
        return "vrptw-"
    return ""


def default_data_dir() -> Path:
    return ROOT_DIR / "data" / "cvrptw"


def load_dataset(n_nodes: int, device: str, tam: bool = False, vrptw: bool = False,
                 data_dir: str | Path | None = None):
    base_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    filename = base_dir / f"testDataset-{dataset_mode_prefix(tam, vrptw)}{n_nodes}.pt"
    if not filename.is_file():
        raise FileNotFoundError(
            f"File {filename} not found. Generate it with generate_data.py first."
        )

    dataset = torch.load(filename, map_location=device)
    test_list = []
    for i in range(len(dataset)):
        demands = dataset[i, 0, :]
        positions = dataset[i, 1:3, :].T
        distances = dataset[i, 3:-2, :]
        windows = dataset[i, -2:, :].T
        test_list.append((demands, distances, positions, windows))
    return test_list


def route_edges(route: Iterable[int]) -> set[frozenset[int]]:
    nodes = [int(node) for node in route]
    return set(frozenset((u, v)) for u, v in zip(nodes[:-1], nodes[1:]))


def route_diversity(routes) -> float:
    if len(routes) < 2:
        return 0.0
    edge_sets = [route_edges(route) for route in routes]
    total = 0.0
    pairs = 0
    for left, right in itertools.combinations(edge_sets, 2):
        union = left | right
        similarity = len(left & right) / len(union) if union else 1.0
        total += similarity
        pairs += 1
    return 1.0 - total / pairs if pairs else 0.0

def route_edge_distance(route_a, route_b) -> float:
    edges_a = route_edges(route_a)
    edges_b = route_edges(route_b)
    return edge_set_distance(edges_a, edges_b)


def edge_set_distance(edges_a, edges_b) -> float:
    union = edges_a | edges_b
    if not union:
        return 0.0
    return 1.0 - len(edges_a & edges_b) / len(union)


def route_entry(route, cost, edges=None):
    route = np.asarray(route, dtype=np.int32).copy()
    return {"route": route, "cost": float(cost), "edges": route_edges(route) if edges is None else edges}


def normalize_archive_entry(item):
    return route_entry(item["route"], item["cost"], item.get("edges"))

def update_elite_archive(
    archive,
    routes,
    costs,
    elite_k: int,
    elite_min_diversity: float,
    elite_cost_tolerance: float,
    pinned_elite=None,
):
    if elite_k <= 0:
        return []

    candidates = [normalize_archive_entry(item) for item in archive]
    if pinned_elite is not None:
        pinned_elite = normalize_archive_entry(pinned_elite)
        candidates.append(pinned_elite)

    best_candidate_cost = min([float(item["cost"]) for item in candidates] + [float(np.min(costs))])
    max_allowed_cost = best_candidate_cost * elite_cost_tolerance

    for route, cost in zip(routes, costs):
        cost = float(cost)
        if cost <= max_allowed_cost:
            candidates.append(route_entry(route, cost))

    candidates.sort(key=lambda item: item["cost"])
    diverse = []
    if pinned_elite is not None:
        diverse.append(pinned_elite)

    for candidate in candidates:
        if any(np.array_equal(candidate["route"], item["route"]) for item in diverse):
            continue
        if all(edge_set_distance(candidate["edges"], item["edges"]) >= elite_min_diversity for item in diverse):
            diverse.append(candidate)
        if len(diverse) >= elite_k:
            break
    return diverse

def select_elite_source(archive, step: int):
    if not archive:
        return None
    return archive[(step + 1) % len(archive)]

def sample_multisource(solver, archive, source_count: int, prior=None):
    source_count = min(source_count, len(archive))
    all_costs = []
    all_routes = []
    for source in archive[:source_count]:
        solver.set_source_route(source["route"], source["cost"])
        costs, routes, *_ = solver.sample(prior=prior)
        all_costs.append(np.asarray(costs, dtype=np.float32))
        all_routes.extend(np.asarray(route, dtype=np.int32) for route in routes)
    return np.concatenate(all_costs), np.asarray(all_routes, dtype=object)


def infer_instance(
    demands,
    positions,
    windows,
    n_ants: int,
    n_iter: int,
    threads: int | None,
    seed: int,
    cand_list_size: int,
    backup_list_size: int,
    min_new_edges: int,
    decay: float,
    alpha: float,
    p_best: float,
    use_local_search: bool,
    disable_heuristic: bool,
    extend_ls: bool,
    smooth_mmas: bool,
    fixed_steps: int,
    nls: bool,
    T_nls: int,
    elite_k: int = 8,
    elite_min_diversity: float = 0.15,
    elite_cost_tolerance: float = 1.05,
    source_count: int = 2,
    pin_global_best_elite: bool = False,
    prior=None,
):
    if threads is not None and threads < 1:
        raise ValueError("threads must be >= 1")

    solver = MFACO_CVRPTW(
        positions,
        demands,
        windows,
        capacity=1.0,
        n_ants=n_ants,
        cand_list_size=cand_list_size,
        backup_list_size=backup_list_size,
        min_new_edges=min_new_edges,
        decay=decay,
        alpha=alpha,
        p_best=p_best,
        use_local_search=use_local_search,
        disable_heuristic=disable_heuristic,
        extend_ls=extend_ls,
        smooth_mmas=smooth_mmas,
        device="cpu",
        fixed_steps=fixed_steps,
        nls=nls,
        T_nls=T_nls,
    )
    solver.seed_rng(seed)

    results = torch.zeros(size=(n_iter,), dtype=torch.float32)
    diversities = torch.zeros(size=(n_iter,), dtype=torch.float32)
    best_so_far = float("inf")
    global_best_route = None

   
    start = time.time()
    for t in range(n_iter):
        costs, routes, *_ = solver.sample(prior=prior)
        costs_np = np.asarray(costs, dtype=np.float32)

        best_idx = int(np.argmin(costs_np))
        best_cost = float(costs_np[best_idx])
        best_route = np.asarray(routes[best_idx], dtype=np.int32)
        if best_cost < best_so_far:
            best_so_far = best_cost
            global_best_route = best_route.copy()

        solver.update_pheromone(best_route, best_cost)

        results[t] = best_so_far
        diversities[t] = route_diversity(routes)
    elapsed = time.time() - start
    return results, diversities, elapsed


def load_neural_guide(checkpoint_path: str | Path, device: str):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Neural guide checkpoint {checkpoint_path} not found")

    state_dict = torch.load(checkpoint_path, map_location=device)
    for value_head in (True, False):
        model = Net(value_head=value_head).to(device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            continue
        model.eval()
        return model
    raise RuntimeError(f"Checkpoint {checkpoint_path} is not compatible with FACO Net")

@torch.no_grad()
def neural_prior(model, demands, positions, windows, cand_list_size: int, device: str):
    if model is None:
        return None
    instance = normalize_instance(
        {
            "coords": positions,
            "demand": demands,
            "windows": windows,
            "capacity": 1.0,
        },
        device=device,
    )
    pyg_data = gen_pyg_data(instance, cand_list_size=cand_list_size, device=device)
    heu_vec = model(pyg_data)
    return model.reshape(pyg_data, heu_vec).detach().cpu().numpy()

def test(dataset, n_ants: int, n_iter: int, threads: int | None, seed: int, model=None, device: str = "cpu", cand_list_size: int = 32, **solver_kwargs):
    sum_results = torch.zeros(size=(n_iter,), dtype=torch.float32)
    sum_diversities = torch.zeros(size=(n_iter,), dtype=torch.float32)
    sum_times = 0.0

    for idx, (demands, _distances, positions, windows) in enumerate(tqdm(dataset, dynamic_ncols=True)):
        prior = neural_prior(model, demands, positions, windows, cand_list_size=cand_list_size, device=device)
        results, diversities, elapsed_time = infer_instance(
            demands=demands.cpu(),
            positions=positions.cpu(),
            windows=windows.cpu(),
            n_ants=n_ants,
            n_iter=n_iter,
            threads=threads,
            seed=seed + idx,
            cand_list_size=cand_list_size,
            prior=prior,
            **solver_kwargs,
        )
        sum_results += results
        sum_diversities += diversities
        sum_times += elapsed_time

    return sum_results / len(dataset), sum_diversities / len(dataset), sum_times / len(dataset)


def write_results(
    result_txt: Path,
    result_csv: Path,
    problem_name: str,
    n_nodes: int,
    n_instances: int,
    n_ants: int,
    n_iter: int,
    threads: int | None,
    seed: int,
    duration: float,
    avg_cost,
    avg_diversity,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
):
    result_txt.parent.mkdir(parents=True, exist_ok=True)
    with result_txt.open("w") as f:
        f.write(f"problem: {problem_name}\n")
        f.write(f"problem scale: {n_nodes}\n")
        f.write(f"checkpoint: {checkpoint if checkpoint is not None else 'none'}\n")
        f.write(f"number of instances: {n_instances}\n")
        f.write(f"device: {device}\n")
        f.write(f"n_ants: {n_ants}\n")
        f.write(f"threads: {threads if threads is not None else 'default'}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"average inference time: {duration}\n")
        for i in range(n_iter):
            f.write(f"T={i + 1}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}\n")

    with result_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["T", "avg_cost", "avg_diversity"])
        writer.writeheader()
        for i in range(n_iter):
            writer.writerow({"T": i + 1, "avg_cost": float(avg_cost[i]), "avg_diversity": float(avg_diversity[i])})


def main(
    n_nodes: int,
    k_sparse: int | None = None,
    size: int | None = None,
    n_ants: int = 100,
    n_iter: int = 10,
    threads: int | None = None,
    seed: int = 0,
    tam: bool = False,
    vrptw: bool = False,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    cand_list_size: int | None = None,
    backup_list_size: int = 64,
    min_new_edges: int = 8,
    decay: float = 0.9,
    alpha: float = 1.0,
    p_best: float = 0.05,
    use_local_search: bool = True,
    disable_heuristic: bool = False,
    extend_ls: bool = False,
    smooth_mmas: bool = False,
    fixed_steps: int = 0,
    nls: bool = False,
    T_nls: int = 10,
    elite_k: int = 8,
    elite_min_diversity: float = 0.15,
    elite_cost_tolerance: float = 1.05,
    source_count: int = 2,
    pin_global_best_elite: bool = False,
    neural_guide: str | Path | None = None,
    device: str = "cuda:0",
):
    if threads is not None and threads < 1:
        raise ValueError("threads must be >= 1")
    if threads is not None:
        set_faco_cpp_threads(threads)

    if k_sparse is None:
        k_sparse = n_nodes // 5
    if cand_list_size is None:
        cand_list_size = min(k_sparse, 32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model_device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    model = load_neural_guide(neural_guide, model_device) if neural_guide is not None else None

    dataset = load_dataset(n_nodes, "cpu", tam=tam, vrptw=vrptw, data_dir=data_dir)
    dataset = dataset[: (size or len(dataset))]

    problem_name = f"{'tam-' if tam else ''}{'vrptw' if vrptw else 'cvrptw'}"
    print("problem:", problem_name)
    print("problem scale:", n_nodes)
    print("checkpoint:", neural_guide if neural_guide is not None else None)
    print("number of instances:", len(dataset))
    print("device:", f"{model_device}+cpu" if model is not None and model_device != "cpu" else "cpu")
    print("n_ants:", n_ants)
    print("threads:", threads if threads is not None else "default")
    print("seed:", seed)

    avg_cost, avg_diversity, duration = test(
        dataset,
        n_ants=n_ants,
        n_iter=n_iter,
        threads=threads,
        seed=seed,
        model=model,
        device=model_device,
        cand_list_size=cand_list_size,
        backup_list_size=backup_list_size,
        min_new_edges=min_new_edges,
        decay=decay,
        alpha=alpha,
        p_best=p_best,
        use_local_search=use_local_search,
        disable_heuristic=disable_heuristic,
        extend_ls=extend_ls,
        smooth_mmas=smooth_mmas,
        fixed_steps=fixed_steps,
        nls=nls,
        T_nls=T_nls,
        elite_k=elite_k,
        elite_min_diversity=elite_min_diversity,
        elite_cost_tolerance=elite_cost_tolerance,
        source_count=source_count,
        pin_global_best_elite=pin_global_best_elite,
    )

    print("average inference time: ", duration)
    for i in range(n_iter):
        print(f"T={i + 1}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}")

    out_dir = Path(output_dir) if output_dir is not None else ROOT_DIR / "pretrained" / "cvrptw" / str(n_nodes) / "faco"
    checkpoint_name = Path(neural_guide).stem if neural_guide is not None else "none"
    result_filename = (
        f"test_result_ckpt{checkpoint_name}-{problem_name}{n_nodes}-ninst{size}-"
        f"nants{n_ants}-niter{n_iter}-threads{threads if threads is not None else 'default'}-seed{seed}-faco"
    )
    result_txt = out_dir / f"{result_filename}.txt"
    result_csv = out_dir / f"{result_filename}.csv"
    write_results(
        result_txt=result_txt,
        result_csv=result_csv,
        problem_name=problem_name,
        n_nodes=n_nodes,
        n_instances=len(dataset),
        n_ants=n_ants,
        n_iter=n_iter,
        threads=threads,
        seed=seed,
        duration=duration,
        avg_cost=avg_cost,
        avg_diversity=avg_diversity,
        checkpoint=neural_guide,
        device=f"{model_device}+cpu" if model is not None and model_device != "cpu" else "cpu",
    )
    return avg_cost, avg_diversity, duration, result_txt, result_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Test MFACO_CVRPTW on GFACS-format datasets.")
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse / default FACO candidate list size")
    parser.add_argument("-i", "--n_iter", type=int, default=10, help="Number of FACO iterations")
    parser.add_argument("--threads", type=int, default=None, help="OpenMP thread count for parallel C++ FACO sampling/update")
    parser.add_argument("-n", "--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("-s", "--size", type=int, default=None, help="Number of instances to test")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")
    parser.add_argument("--vrptw", action="store_true", help="Use VRPTW dataset with zero customer demands")
    parser.add_argument("--data_dir", type=Path, default=None, help="Directory containing GFACS-format testDataset-*.pt files")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for result txt/csv files")
    parser.add_argument("--neural_guide", type=Path, default=None, help="Path to learned FACO PPO .pt checkpoint used as neural sampling prior")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for neural guide prior computation")

    parser.add_argument("--cand_list_size", type=int, default=None, help="FACO candidate list size; defaults to k_sparse")
    parser.add_argument("--backup_list_size", type=int, default=64, help="FACO backup list size")
    parser.add_argument("--min_new_edges", type=int, default=8, help="FACO min_new_edges")
    parser.add_argument("--decay", type=float, default=0.9, help="FACO pheromone decay")
    parser.add_argument("--alpha", type=float, default=1.0, help="FACO pheromone exponent")
    parser.add_argument("--p_best", type=float, default=0.05, help="FACO p_best")
    parser.add_argument("--disable_local_search", action="store_true", help="Disable FACO local search")
    parser.add_argument("--disable_heuristic", action="store_true", help="Disable FACO distance heuristic")
    parser.add_argument("--extend_ls", action="store_true", help="Enable extended local search checklist")
    parser.add_argument("--smooth_mmas", action="store_true", help="Enable smooth MMAS trail limits")
    parser.add_argument("--fixed_steps", type=int, default=0, help="Fixed sampler steps; 0 means default")
    parser.add_argument("--nls", action="store_true", help="Enable NLS mode")
    parser.add_argument("--T_nls", type=int, default=10, help="Number of NLS iterations")
    parser.add_argument("--elite_k", type=int, default=8, help="Number of quality-diverse elite source routes")
    parser.add_argument("--elite_min_diversity", type=float, default=0.15, help="Minimum edge distance between elite source routes")
    parser.add_argument("--elite_cost_tolerance", type=float, default=1.05, help="Maximum elite source cost ratio versus current best")
    parser.add_argument("--source_count", type=int, default=2, help="Number of elite source routes used by multisource sampling")
    parser.add_argument("--pin_global_best_elite", action="store_true", help="Always keep global best as one elite archive route")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        n_nodes=args.nodes,
        k_sparse=args.k_sparse,
        size=args.size,
        n_ants=args.n_ants,
        n_iter=args.n_iter,
        threads=args.threads,
        seed=args.seed,
        tam=args.tam,
        vrptw=args.vrptw,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cand_list_size=args.cand_list_size,
        backup_list_size=args.backup_list_size,
        min_new_edges=args.min_new_edges,
        decay=args.decay,
        alpha=args.alpha,
        p_best=args.p_best,
        use_local_search=not args.disable_local_search,
        disable_heuristic=args.disable_heuristic,
        extend_ls=args.extend_ls,
        smooth_mmas=args.smooth_mmas,
        fixed_steps=args.fixed_steps,
        nls=args.nls,
        T_nls=args.T_nls,
        elite_k=args.elite_k,
        elite_min_diversity=args.elite_min_diversity,
        elite_cost_tolerance=args.elite_cost_tolerance,
        source_count=args.source_count,
        pin_global_best_elite=args.pin_global_best_elite,
        neural_guide=args.neural_guide,
        device=args.device,
    )
