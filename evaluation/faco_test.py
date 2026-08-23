from __future__ import annotations

__test__ = False

import argparse
import csv
import itertools
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from utils import (
    dataset_mode_prefix,
    default_data_dir,
    load_dataset,
    load_rl4co_dataset,
    normalize_dataset_item,
)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

from envs import cvrptw_env as ppo_utils
from models.faco_net import Net as PPONet
from models.gfacs_net import Net as OriginalGFACSNet
from solvers.faco import MFACO_CVRPTW, set_faco_cpp_threads


def _edge_key(u: int, v: int) -> int:
    u = int(u)
    v = int(v)
    if u > v:
        u, v = v, u
    return (u << 32) | v


def route_edges(route: Iterable[int]) -> set[int]:
    nodes = [int(node) for node in route]
    return {_edge_key(u, v) for u, v in zip(nodes[:-1], nodes[1:])}


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


def load_deepaco_prior_model(checkpoint_path: str | Path, device: str):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DeepACO checkpoint {checkpoint_path} not found")

    state_dict = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"DeepACO checkpoint {checkpoint_path} must contain a state_dict")

    model = OriginalGFACSNet(gfn=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_gfacs_prior_model(checkpoint_path: str | Path, device: str, guided_exploration: bool | None = None):
    del guided_exploration
    return load_deepaco_prior_model(checkpoint_path, device)


def gen_deepaco_pyg_data(demands, distances, windows, device: str, k_sparse: int):
    demands = torch.as_tensor(demands, dtype=torch.float32, device=device)
    distances = torch.as_tensor(distances, dtype=torch.float32, device=device)
    windows = torch.as_tensor(windows, dtype=torch.float32, device=device)
    n_nodes = int(demands.size(0))
    k_sparse = min(int(k_sparse), max(1, n_nodes - 1))

    temp_dists = distances.clone()
    if n_nodes > 1:
        eye = torch.eye(n_nodes - 1, dtype=torch.bool, device=device)
        temp_dists[1:, 1:][eye] = 1e9
        topk_values, topk_indices = torch.topk(temp_dists[1:, 1:], k=k_sparse, dim=1, largest=False)
        edge_index_1 = torch.stack([
            torch.repeat_interleave(torch.arange(n_nodes - 1, device=device), repeats=k_sparse),
            torch.flatten(topk_indices),
        ]) + 1
        edge_attr_1 = topk_values.reshape(-1, 1)
        edge_index_2 = torch.stack([
            torch.zeros(n_nodes - 1, device=device, dtype=torch.long),
            torch.arange(1, n_nodes, device=device, dtype=torch.long),
        ])
        edge_attr_2 = temp_dists[1:, 0].reshape(-1, 1)
        edge_index_3 = torch.stack([
            torch.arange(1, n_nodes, device=device, dtype=torch.long),
            torch.zeros(n_nodes - 1, device=device, dtype=torch.long),
        ])
        edge_index = torch.concat([edge_index_1, edge_index_2, edge_index_3], dim=1)
        edge_attr = torch.concat([edge_attr_1, edge_attr_2, edge_attr_2])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)

    x = torch.cat([demands.unsqueeze(1), windows], dim=-1)
    return Data(x=x.float(), edge_attr=edge_attr.float(), edge_index=edge_index)

def gen_granular_pyg_data(demands, distances, windows, solver, device: str):
    demands = torch.as_tensor(demands, dtype=torch.float32, device=device)
    distances = torch.as_tensor(distances, dtype=torch.float32, device=device)
    windows = torch.as_tensor(windows, dtype=torch.float32, device=device)
    nn_list = torch.as_tensor(np.asarray(solver.nn_list), dtype=torch.long, device=device)
    if nn_list.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
    else:
        rows = torch.arange(nn_list.shape[0], device=device).unsqueeze(1).expand_as(nn_list)
        edge_index = torch.stack([rows.reshape(-1), nn_list.reshape(-1)], dim=0)
        edge_attr = distances[edge_index[0], edge_index[1]].reshape(-1, 1)
    x = torch.cat([demands.unsqueeze(1), windows], dim=-1)
    return Data(x=x.float(), edge_attr=edge_attr.float(), edge_index=edge_index)


def build_instance_for_prior(demands, distances, windows):
    return {
        "demand": demands,
        "distances": distances,
        "windows": windows,
    }


@torch.no_grad()
def deepaco_prior(
    model,
    instance,
    solver,
    k_sparse: int,
    device: str,
    prior_scale: float = 1.0,
    prior_center: bool = False,
    graph_mode: str = "distance",
):
    if model is None:
        return None
    if graph_mode == "granular":
        pyg_data = gen_granular_pyg_data(
            instance["demand"],
            instance["distances"],
            instance["windows"],
            solver,
            device=device,
        )
    else:
        pyg_data = gen_deepaco_pyg_data(
            instance["demand"],
            instance["distances"],
            instance["windows"],
            device=device,
            k_sparse=k_sparse,
        )
    heuristic_mat = model.reshape(pyg_data, model(pyg_data))
    nn_list = torch.as_tensor(np.asarray(solver.nn_list), dtype=torch.long, device=heuristic_mat.device)
    rows = torch.arange(nn_list.shape[0], device=heuristic_mat.device).unsqueeze(1)
    sparse_heuristic = heuristic_mat[rows, nn_list]
    prior = torch.zeros_like(sparse_heuristic, dtype=torch.float32)
    positive = sparse_heuristic > 0
    prior[positive] = torch.log(sparse_heuristic[positive].to(torch.float32).clamp_min(1e-10))
    if prior_center:
        prior = prior - prior.mean(dim=1, keepdim=True)
    if prior_scale != 1.0:
        prior = prior * float(prior_scale)
    return prior.detach().cpu().numpy().astype(np.float32, copy=False)


# Deprecated compatibility aliases: faco_test now treats these as DeepACO prior helpers.
gen_original_gfacs_pyg_data = gen_deepaco_pyg_data
gfacs_prior = deepaco_prior


def load_ppo_prior_model(checkpoint_path: str | Path, device: str, norm_type: str = "batch"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PPO checkpoint {checkpoint_path} not found")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"PPO checkpoint {checkpoint_path} must contain a state_dict")
    model = PPONet(value_head=False, norm_type=norm_type).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_ppo_instance_for_prior(demands, positions, windows):
    return {
        "coords": positions,
        "demand": demands,
        "windows": windows,
        "capacity": 1.0,
    }


@torch.no_grad()
def ppo_prior(
    model,
    instance,
    solver,
    cand_list_size: int,
    backup_list_size: int,
    n_ants: int,
    min_new_edges: int,
    device: str,
    edge_feature_mode: str = "full",
    prior_scale: float = 1.0,
    prior_center: bool = False,
):
    if model is None:
        return None
    pyg_data = ppo_utils.gen_pyg_data(
        instance,
        cand_list_size=cand_list_size,
        backup_list_size=backup_list_size,
        n_ants=n_ants,
        min_new_edges=min_new_edges,
        solver=solver,
        device=device,
        edge_feature_mode=edge_feature_mode,
    )
    prior = model.reshape(pyg_data, model(pyg_data))
    if prior_center:
        prior = prior - prior.mean(dim=1, keepdim=True)
    if prior_scale != 1.0:
        prior = prior * float(prior_scale)
    return prior.detach().cpu().numpy().astype(np.float32, copy=False)


def sample_multisource(solver, archive, source_count: int, prior=None, prior_fn=None):
    source_count = min(source_count, len(archive))
    all_costs = []
    all_routes = []
    for source in archive[:source_count]:
        solver.set_source_route(source["route"], source["cost"])
        sample_prior = prior_fn() if prior_fn is not None else prior
        costs, routes, *_ = solver.sample(prior=sample_prior)
        all_costs.append(np.asarray(costs, dtype=np.float32))
        all_routes.extend(np.asarray(route, dtype=np.int32) for route in routes)
    return np.concatenate(all_costs), np.asarray(all_routes, dtype=object)


def resolve_prior(prior=None, prior_fn=None):
    return prior_fn() if prior_fn is not None else prior


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
    deep_nls: bool = False,
    log_period: int = 1,
    granular_mode: int = 0,
    granular_wait_weight: float = 0.2,
    granular_time_warp_weight: float = 1.0,
    hgs_soft_deep_ls: bool = False,
    hgs_soft_cheap_ls: bool = False,
    hgs_soft_intra_ls: bool = False,
    hgs_deep_ls: bool = False,
    hgs_deep_top_k: int = 0,
    hgs_tw_penalty: float = 10.0,
    hgs_capacity_penalty: float = 10.0,
    hgs_adaptive_penalty: bool = False,
    hgs_target_feasible: float = 0.8,
    hgs_deep_rounds: int = 1,
    hgs_deep_route_pair_prune: bool = False,
    hgs_deep_route_pair_top_k: int = 3,
    elite_k: int = 8,
    elite_min_diversity: float = 0.15,
    elite_cost_tolerance: float = 1.05,
    source_count: int = 2,
    elite_source_period: int = 1,
    pin_global_best_elite: bool = False,
    deepaco_model=None,
    deepaco_k_sparse: int | None = None,
    deepaco_device: str = "cpu",
    deepaco_prior_scale: float = 1.0,
    deepaco_prior_center: bool = False,
    deepaco_graph_mode: str = "distance",
    ppo_model=None,
    ppo_device: str = "cpu",
    ppo_edge_feature_mode: str = "full",
    ppo_prior_refresh: str = "outer",
    ppo_prior_scale: float = 1.0,
    ppo_prior_center: bool = False,
    distances=None,
    cost_evaluator=None,
):
    if elite_source_period < 0:
        raise ValueError("elite_source_period must be >= 0")
    if log_period < 1:
        raise ValueError("log_period must be >= 1")
    if source_count < 1:
        raise ValueError("source_count must be >= 1")
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
        deep_nls=deep_nls,
        granular_mode=granular_mode,
        granular_wait_weight=granular_wait_weight,
        granular_time_warp_weight=granular_time_warp_weight,
        hgs_soft_deep_ls=hgs_soft_deep_ls,
        hgs_soft_cheap_ls=hgs_soft_cheap_ls,
        hgs_soft_intra_ls=hgs_soft_intra_ls,
        hgs_deep_ls=hgs_deep_ls,
        hgs_deep_top_k=hgs_deep_top_k,
        hgs_tw_penalty=hgs_tw_penalty,
        hgs_capacity_penalty=hgs_capacity_penalty,
        hgs_adaptive_penalty=hgs_adaptive_penalty,
        hgs_target_feasible=hgs_target_feasible,
        hgs_deep_rounds=hgs_deep_rounds,
        hgs_deep_route_pair_prune=hgs_deep_route_pair_prune,
        hgs_deep_route_pair_top_k=hgs_deep_route_pair_top_k,
    )
    solver.seed_rng(seed)

    log_steps = list(range(log_period, n_iter + 1, log_period))
    if not log_steps or log_steps[-1] != n_iter:
        log_steps.append(n_iter)
    log_step_set = set(log_steps)
    results = torch.zeros(size=(len(log_steps),), dtype=torch.float32)
    diversities = torch.zeros(size=(len(log_steps),), dtype=torch.float32)
    log_idx = 0
    best_so_far = float("inf")
    global_best_route = None
    elite_archive = []
    if distances is None:
        positions_tensor = torch.as_tensor(positions, dtype=torch.float32)
        distances = torch.cdist(positions_tensor, positions_tensor)
    gfacs_prior_instance = build_instance_for_prior(demands, distances, windows)
    ppo_prior_instance = build_ppo_instance_for_prior(demands, positions, windows)
    static_prior = deepaco_prior(
        deepaco_model,
        gfacs_prior_instance,
        solver,
        k_sparse=deepaco_k_sparse if deepaco_k_sparse is not None else cand_list_size,
        device=deepaco_device,
        prior_scale=deepaco_prior_scale,
        prior_center=deepaco_prior_center,
        graph_mode=deepaco_graph_mode,
    )

    def current_prior():
        if ppo_model is not None:
            return ppo_prior(
                ppo_model,
                ppo_prior_instance,
                solver,
                cand_list_size=cand_list_size,
                backup_list_size=backup_list_size,
                n_ants=n_ants,
                min_new_edges=min_new_edges,
                device=ppo_device,
                edge_feature_mode=ppo_edge_feature_mode,
                prior_scale=ppo_prior_scale,
                prior_center=ppo_prior_center,
            )
        return static_prior

    start = time.time()
    log_group_idx = 0
    iter_prior = current_prior() if ppo_model is not None and ppo_prior_refresh == "outer" else static_prior
    for step in range(1, n_iter + 1):
        is_log_step = step in log_step_set
        if (step - 1) % log_period == 0:
            iter_prior = current_prior() if ppo_model is not None and ppo_prior_refresh == "outer" else static_prior
        iter_prior_fn = current_prior if ppo_model is not None and ppo_prior_refresh == "sample" else None

        if is_log_step and elite_source_period > 0 and step % elite_source_period == 0 and len(elite_archive) > 1:
            costs_np, routes = sample_multisource(
                solver,
                elite_archive,
                source_count,
                prior_fn=iter_prior_fn,
                prior=iter_prior,
            )
        else:
            if is_log_step and elite_source_period > 0 and step % elite_source_period == 0:
                elite_source = select_elite_source(elite_archive, log_group_idx)
                if elite_source is not None:
                    solver.set_source_route(elite_source["route"], elite_source["cost"])
                elif global_best_route is not None:
                    solver.set_source_route(global_best_route, best_so_far)
            costs, routes, *_ = solver.sample(prior=resolve_prior(iter_prior, iter_prior_fn))
            costs_np = np.asarray(costs, dtype=np.float32)

        best_idx = int(np.argmin(costs_np))
        best_cost = float(costs_np[best_idx])
        best_route = np.asarray(routes[best_idx], dtype=np.int32)
        if best_cost < best_so_far:
            best_so_far = best_cost
            global_best_route = best_route.copy()
        pinned_elite = {"route": global_best_route, "cost": best_so_far} if pin_global_best_elite and global_best_route is not None else None
        elite_archive = update_elite_archive(
            elite_archive,
            routes,
            costs_np,
            elite_k=elite_k,
            elite_min_diversity=elite_min_diversity,
            elite_cost_tolerance=elite_cost_tolerance,
            pinned_elite=pinned_elite,
        )

        solver.update_pheromone(best_route, best_cost)
        if is_log_step:
            if cost_evaluator is not None and global_best_route is not None:
                results[log_idx] = float(cost_evaluator([global_best_route])[0])
            else:
                results[log_idx] = best_so_far
            diversities[log_idx] = route_diversity(routes)
            log_idx += 1
            log_group_idx += 1
    elapsed = time.time() - start
    timings = solver.get_timings() if hasattr(solver, "get_timings") else {}
    return results, diversities, elapsed, log_steps, timings


def test(dataset, n_ants: int, n_iter: int, log_period: int, threads: int | None, seed: int, **solver_kwargs):
    log_steps = list(range(log_period, n_iter + 1, log_period))
    if not log_steps or log_steps[-1] != n_iter:
        log_steps.append(n_iter)
    sum_results = torch.zeros(size=(len(log_steps),), dtype=torch.float32)
    sum_diversities = torch.zeros(size=(len(log_steps),), dtype=torch.float32)
    sum_times = 0.0
    timing_sums: dict[str, float] = {}

    for idx, item in enumerate(tqdm(dataset, dynamic_ncols=True)):
        demands, distances, positions, windows, cost_evaluator = normalize_dataset_item(item)
        results, diversities, elapsed_time, instance_log_steps, timings = infer_instance(
            demands=demands.cpu(),
            distances=distances.cpu(),
            positions=positions.cpu(),
            windows=windows.cpu(),
            cost_evaluator=cost_evaluator,
            n_ants=n_ants,
            n_iter=n_iter,
            log_period=log_period,
            threads=threads,
            seed=seed + idx,
            **solver_kwargs,
        )
        sum_results += results
        sum_diversities += diversities
        sum_times += elapsed_time
        for key, value in timings.items():
            timing_sums[key] = timing_sums.get(key, 0.0) + float(value)

    assert instance_log_steps == log_steps
    avg_timings = {key: value / len(dataset) for key, value in timing_sums.items()}
    return sum_results / len(dataset), sum_diversities / len(dataset), sum_times / len(dataset), log_steps, avg_timings


def write_results(
    result_txt: Path,
    result_csv: Path,
    problem_name: str,
    n_nodes: int,
    n_instances: int,
    n_ants: int,
    n_iter: int,
    log_period: int,
    elite_source_period: int,
    threads: int | None,
    seed: int,
    duration: float,
    avg_cost,
    avg_diversity,
    log_steps: list[int],
    data_source: str = "gfacs",
    rl4co_phase: str = "test",
    rl4co_file: str | Path | None = None,
    rl4co_seed: int = 1234,
    rl4co_scale: bool = False,
    checkpoint: str | None = None,
    checkpoint_type: str = "none",
    granular_mode: int = 0,
    use_local_search: bool = True,
    extend_ls: bool = False,
    smooth_mmas: bool = False,
    hgs_soft_deep_ls: bool = False,
    hgs_soft_cheap_ls: bool = False,
    hgs_soft_intra_ls: bool = False,
    hgs_deep_ls: bool = False,
    hgs_deep_top_k: int = 0,
    hgs_tw_penalty: float = 10.0,
    hgs_capacity_penalty: float = 10.0,
    hgs_adaptive_penalty: bool = False,
    hgs_target_feasible: float = 0.8,
    hgs_deep_rounds: int = 1,
    hgs_deep_route_pair_prune: bool = False,
    hgs_deep_route_pair_top_k: int = 3,
    deepaco_device: str = "cpu",
    deepaco_prior_scale: float = 1.0,
    deepaco_prior_center: bool = False,
    deepaco_graph_mode: str = "distance",
    ppo_device: str = "cpu",
    ppo_norm_type: str = "batch",
    ppo_edge_feature_mode: str = "full",
    ppo_prior_refresh: str = "outer",
    ppo_prior_scale: float = 1.0,
    ppo_prior_center: bool = False,
    timings: dict | None = None,
):
    result_txt.parent.mkdir(parents=True, exist_ok=True)
    with result_txt.open("w") as f:
        f.write("[Data]\n")
        f.write(f"data_source: {data_source}\n")
        f.write(f"number of instances: {n_instances}\n")
        if data_source == "rl4co":
            f.write(f"rl4co_phase: {rl4co_phase}\n")
            f.write(f"rl4co_file: {rl4co_file if rl4co_file is not None else 'default/generated'}\n")
            f.write(f"rl4co_seed: {rl4co_seed}\n")
            f.write(f"rl4co_scale: {rl4co_scale}\n")

        f.write("[FACO]\n")
        f.write(f"problem: {problem_name}\n")
        f.write(f"problem scale: {n_nodes}\n")
        f.write("device: cpu\n")
        f.write(f"n_ants: {n_ants}\n")
        f.write(f"n_iter: {n_iter}\n")
        f.write(f"log_period: {log_period}\n")
        f.write(f"elite_source_period: {elite_source_period}\n")
        f.write(f"threads: {threads if threads is not None else 'default'}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"granular_mode: {granular_mode}\n")
        f.write(f"use_local_search: {use_local_search}\n")
        f.write(f"extend_ls: {extend_ls}\n")
        f.write(f"smooth_mmas: {smooth_mmas}\n")

        f.write("[HGS / DeepLS]\n")
        f.write(f"hgs_soft_deep_ls: {hgs_soft_deep_ls}\n")
        f.write(f"hgs_soft_cheap_ls: {hgs_soft_cheap_ls}\n")
        f.write(f"hgs_soft_intra_ls: {hgs_soft_intra_ls}\n")
        f.write(f"hgs_deep_ls: {hgs_deep_ls}\n")
        f.write(f"hgs_deep_top_k: {hgs_deep_top_k}\n")
        f.write(f"hgs_tw_penalty: {hgs_tw_penalty}\n")
        f.write(f"hgs_capacity_penalty: {hgs_capacity_penalty}\n")
        f.write(f"hgs_adaptive_penalty: {hgs_adaptive_penalty}\n")
        f.write(f"hgs_target_feasible: {hgs_target_feasible}\n")
        f.write(f"hgs_deep_rounds: {hgs_deep_rounds}\n")
        f.write(f"hgs_deep_route_pair_prune: {hgs_deep_route_pair_prune}\n")
        f.write(f"hgs_deep_route_pair_top_k: {hgs_deep_route_pair_top_k}\n")

        if checkpoint_type == "deepaco":
            f.write("[DeepACO Prior]\n")
            f.write(f"checkpoint: {checkpoint}\n")
            f.write(f"deepaco_device: {deepaco_device}\n")
            f.write(f"deepaco_prior_scale: {deepaco_prior_scale}\n")
            f.write(f"deepaco_prior_center: {deepaco_prior_center}\n")
            f.write(f"deepaco_graph_mode: {deepaco_graph_mode}\n")
        elif checkpoint_type == "ppo":
            f.write("[PPO Prior]\n")
            f.write(f"checkpoint: {checkpoint}\n")
            f.write(f"ppo_device: {ppo_device}\n")
            f.write(f"ppo_norm_type: {ppo_norm_type}\n")
            f.write(f"ppo_edge_feature_mode: {ppo_edge_feature_mode}\n")
            f.write(f"ppo_prior_refresh: {ppo_prior_refresh}\n")
            f.write(f"ppo_prior_scale: {ppo_prior_scale}\n")
            f.write(f"ppo_prior_center: {ppo_prior_center}\n")
        else:
            f.write("[Prior]\n")
            f.write("checkpoint: none\n")

        f.write(f"average inference time: {duration}\n")
        if timings:
            for key in sorted(timings):
                f.write(f"{key}: {timings[key]}\n")
        for i, _step in enumerate(log_steps):
            f.write(f"T={i + 1}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}\n")

    with result_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["T", "avg_cost", "avg_diversity"])
        writer.writeheader()
        for i, _step in enumerate(log_steps):
            writer.writerow({"T": i + 1, "avg_cost": float(avg_cost[i]), "avg_diversity": float(avg_diversity[i])})


def main(
    n_nodes: int,
    k_sparse: int | None = None,
    size: int | None = None,
    n_ants: int = 100,
    n_iter: int = 10,
    log_period: int = 1,
    elite_source_period: int = 1,
    threads: int | None = None,
    seed: int = 0,
    tam: bool = False,
    vrptw: bool = False,
    data_source: str = "gfacs",
    data_dir: str | Path | None = None,
    rl4co_root: str | Path | None = None,
    rl4co_phase: str = "test",
    rl4co_file: str | Path | None = None,
    rl4co_seed: int = 1234,
    rl4co_scale: bool = False,
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
    deep_nls: bool = False,
    granular_mode: int = 0,
    granular_wait_weight: float = 0.2,
    granular_time_warp_weight: float = 1.0,
    hgs_soft_deep_ls: bool = False,
    hgs_soft_cheap_ls: bool = False,
    hgs_soft_intra_ls: bool = False,
    hgs_deep_ls: bool = False,
    hgs_deep_top_k: int = 0,
    hgs_tw_penalty: float = 10.0,
    hgs_capacity_penalty: float = 10.0,
    hgs_adaptive_penalty: bool = False,
    hgs_target_feasible: float = 0.8,
    hgs_deep_rounds: int = 1,
    hgs_deep_route_pair_prune: bool = False,
    hgs_deep_route_pair_top_k: int = 3,
    elite_k: int = 8,
    elite_min_diversity: float = 0.15,
    elite_cost_tolerance: float = 1.05,
    source_count: int = 2,
    pin_global_best_elite: bool = False,
    deepaco_pretrained: str | Path | None = None,
    deepaco_device: str | None = None,
    deepaco_prior_scale: float = 1.0,
    deepaco_prior_center: bool = False,
    deepaco_graph_mode: str = "distance",
    gfacs_pretrained: str | Path | None = None,
    gfacs_device: str | None = None,
    gfacs_prior_scale: float = 1.0,
    gfacs_prior_center: bool = False,
    ppo_pretrained: str | Path | None = None,
    ppo_device: str | None = None,
    ppo_norm_type: str = "batch",
    ppo_edge_feature_mode: str = "full",
    ppo_prior_refresh: str = "outer",
    ppo_prior_scale: float = 1.0,
    ppo_prior_center: bool = False,
):
    if elite_source_period < 0:
        raise ValueError("elite_source_period must be >= 0")
    if log_period < 1:
        raise ValueError("log_period must be >= 1")
    if source_count < 1:
        raise ValueError("source_count must be >= 1")
    if threads is not None and threads < 1:
        raise ValueError("threads must be >= 1")
    if threads is not None:
        set_faco_cpp_threads(threads)
    prior_flags = [
        name
        for name, value in (
            ("deepaco_pretrained", deepaco_pretrained),
            ("gfacs_pretrained", gfacs_pretrained),
            ("ppo_pretrained", ppo_pretrained),
        )
        if value is not None
    ]
    if len(prior_flags) > 1:
        raise ValueError("Only one prior checkpoint can be used at a time: " + ", ".join(prior_flags))
    if gfacs_pretrained is not None:
        deepaco_pretrained = gfacs_pretrained
        if deepaco_device is None:
            deepaco_device = gfacs_device
        deepaco_prior_scale = gfacs_prior_scale
        deepaco_prior_center = gfacs_prior_center

    if k_sparse is None:
        k_sparse = n_nodes // 5
    if cand_list_size is None:
        cand_list_size = min(n_nodes // 5, 32)
    if deepaco_device is None:
        deepaco_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if ppo_device is None:
        ppo_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if data_source == "rl4co":
        if vrptw:
            raise ValueError("RL4CO data source is available for CVRPTW only")
        if tam:
            raise ValueError("RL4CO data source does not support TAM datasets")
        dataset = load_rl4co_dataset(
            n_nodes=n_nodes,
            size=size,
            phase=rl4co_phase,
            filename=rl4co_file,
            data_dir=data_dir,
            rl4co_root=rl4co_root,
            seed=rl4co_seed,
            scale=rl4co_scale,
        )
    else:
        dataset = load_dataset(n_nodes, "cpu", tam=tam, vrptw=vrptw, data_dir=data_dir)
        dataset = dataset[: (size or len(dataset))]
    deepaco_model = load_deepaco_prior_model(deepaco_pretrained, deepaco_device) if deepaco_pretrained is not None else None
    ppo_model = load_ppo_prior_model(ppo_pretrained, ppo_device, ppo_norm_type) if ppo_pretrained is not None else None
    checkpoint_label = str(deepaco_pretrained) if deepaco_pretrained is not None else (str(ppo_pretrained) if ppo_pretrained is not None else None)
    checkpoint_type = "deepaco" if deepaco_pretrained is not None else ("ppo" if ppo_pretrained is not None else "none")

    problem_name = f"{'tam-' if tam else ''}{'vrptw' if vrptw else 'cvrptw'}"
    def print_block(title: str, items: dict):
        print(f"[{title}]")
        for key, value in items.items():
            print(f"{key}:", value)

    data_log = {"data_source": data_source, "number of instances": len(dataset)}
    if data_source == "rl4co":
        data_log.update({
            "rl4co_phase": rl4co_phase,
            "rl4co_file": rl4co_file if rl4co_file is not None else "default/generated",
            "rl4co_seed": rl4co_seed,
            "rl4co_scale": rl4co_scale,
        })
    print_block("Data", data_log)
    print_block("FACO", {
        "problem": problem_name,
        "problem scale": n_nodes,
        "device": "cpu",
        "n_ants": n_ants,
        "n_iter": n_iter,
        "log_period": log_period,
        "elite_source_period": elite_source_period,
        "threads": threads if threads is not None else "default",
        "seed": seed,
        "granular_mode": granular_mode,
        "use_local_search": use_local_search,
        "extend_ls": extend_ls,
        "smooth_mmas": smooth_mmas,
    })
    print_block("HGS / DeepLS", {
        "hgs_soft_deep_ls": hgs_soft_deep_ls,
        "hgs_soft_cheap_ls": hgs_soft_cheap_ls,
        "hgs_soft_intra_ls": hgs_soft_intra_ls,
        "hgs_deep_ls": hgs_deep_ls,
        "hgs_deep_top_k": hgs_deep_top_k,
        "hgs_tw_penalty": hgs_tw_penalty,
        "hgs_capacity_penalty": hgs_capacity_penalty,
        "hgs_adaptive_penalty": hgs_adaptive_penalty,
        "hgs_target_feasible": hgs_target_feasible,
        "hgs_deep_rounds": hgs_deep_rounds,
        "hgs_deep_route_pair_prune": hgs_deep_route_pair_prune,
        "hgs_deep_route_pair_top_k": hgs_deep_route_pair_top_k,
    })
    if deepaco_pretrained is not None:
        print_block("DeepACO Prior", {
            "checkpoint": checkpoint_label,
            "deepaco_device": deepaco_device,
            "deepaco_prior_scale": deepaco_prior_scale,
            "deepaco_prior_center": deepaco_prior_center,
            "deepaco_graph_mode": deepaco_graph_mode,
        })
    elif ppo_pretrained is not None:
        print_block("PPO Prior", {
            "checkpoint": checkpoint_label,
            "ppo_device": ppo_device,
            "ppo_norm_type": ppo_norm_type,
            "ppo_edge_feature_mode": ppo_edge_feature_mode,
            "ppo_prior_refresh": ppo_prior_refresh,
            "ppo_prior_scale": ppo_prior_scale,
            "ppo_prior_center": ppo_prior_center,
        })
    else:
        print_block("Prior", {"checkpoint": "none"})

    avg_cost, avg_diversity, duration, log_steps, timings = test(
        dataset,
        n_ants=n_ants,
        n_iter=n_iter,
        log_period=log_period,
        threads=threads,
        seed=seed,
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
        deep_nls=deep_nls,
        granular_mode=granular_mode,
        granular_wait_weight=granular_wait_weight,
        granular_time_warp_weight=granular_time_warp_weight,
        hgs_soft_deep_ls=hgs_soft_deep_ls,
        hgs_soft_cheap_ls=hgs_soft_cheap_ls,
        hgs_soft_intra_ls=hgs_soft_intra_ls,
        hgs_deep_ls=hgs_deep_ls,
        hgs_deep_top_k=hgs_deep_top_k,
        hgs_tw_penalty=hgs_tw_penalty,
        hgs_capacity_penalty=hgs_capacity_penalty,
        hgs_adaptive_penalty=hgs_adaptive_penalty,
        hgs_target_feasible=hgs_target_feasible,
        hgs_deep_rounds=hgs_deep_rounds,
        hgs_deep_route_pair_prune=hgs_deep_route_pair_prune,
        hgs_deep_route_pair_top_k=hgs_deep_route_pair_top_k,
        elite_k=elite_k,
        elite_min_diversity=elite_min_diversity,
        elite_cost_tolerance=elite_cost_tolerance,
        source_count=source_count,
        elite_source_period=elite_source_period,
        pin_global_best_elite=pin_global_best_elite,
        deepaco_model=deepaco_model,
        deepaco_k_sparse=k_sparse,
        deepaco_device=deepaco_device,
        deepaco_prior_scale=deepaco_prior_scale,
        deepaco_prior_center=deepaco_prior_center,
        deepaco_graph_mode=deepaco_graph_mode,
        ppo_model=ppo_model,
        ppo_device=ppo_device,
        ppo_edge_feature_mode=ppo_edge_feature_mode,
        ppo_prior_refresh=ppo_prior_refresh,
        ppo_prior_scale=ppo_prior_scale,
        ppo_prior_center=ppo_prior_center,
    )

    print("average inference time: ", duration)
    if timings:
        for key in sorted(timings):
            print(f"{key}: {timings[key]}")
    for i, _step in enumerate(log_steps):
        print(f"T={i + 1}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}")

    out_dir = Path(output_dir) if output_dir is not None else ROOT_DIR / "pretrained" / "cvrptw" / str(n_nodes) / "faco"
    result_filename = (
        f"test_result_ckpt{checkpoint_type}-{data_source}-{problem_name}{n_nodes}-ninst{size}-"
        f"nants{n_ants}-niter{n_iter}-logperiod{log_period}-eliteperiod{elite_source_period}-gmode{granular_mode}-threads{threads if threads is not None else 'default'}-seed{seed}-faco"
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
        log_period=log_period,
        elite_source_period=elite_source_period,
        threads=threads,
        seed=seed,
        duration=duration,
        avg_cost=avg_cost,
        avg_diversity=avg_diversity,
        log_steps=log_steps,
        data_source=data_source,
        rl4co_phase=rl4co_phase,
        rl4co_file=rl4co_file,
        rl4co_seed=rl4co_seed,
        rl4co_scale=rl4co_scale,
        checkpoint=checkpoint_label,
        checkpoint_type=checkpoint_type,
        granular_mode=granular_mode,
        use_local_search=use_local_search,
        extend_ls=extend_ls,
        smooth_mmas=smooth_mmas,
        hgs_soft_deep_ls=hgs_soft_deep_ls,
        hgs_soft_cheap_ls=hgs_soft_cheap_ls,
        hgs_soft_intra_ls=hgs_soft_intra_ls,
        hgs_deep_ls=hgs_deep_ls,
        hgs_deep_top_k=hgs_deep_top_k,
        hgs_tw_penalty=hgs_tw_penalty,
        hgs_capacity_penalty=hgs_capacity_penalty,
        hgs_adaptive_penalty=hgs_adaptive_penalty,
        hgs_target_feasible=hgs_target_feasible,
        hgs_deep_rounds=hgs_deep_rounds,
        hgs_deep_route_pair_prune=hgs_deep_route_pair_prune,
        hgs_deep_route_pair_top_k=hgs_deep_route_pair_top_k,
        deepaco_device=deepaco_device,
        deepaco_prior_scale=deepaco_prior_scale,
        deepaco_prior_center=deepaco_prior_center,
        deepaco_graph_mode=deepaco_graph_mode,
        ppo_device=ppo_device,
        ppo_norm_type=ppo_norm_type,
        ppo_edge_feature_mode=ppo_edge_feature_mode,
        ppo_prior_refresh=ppo_prior_refresh,
        ppo_prior_scale=ppo_prior_scale,
        ppo_prior_center=ppo_prior_center,
        timings=timings,
    )
    return avg_cost, avg_diversity, duration, result_txt, result_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Test MFACO_CVRPTW on GFACS-format datasets.")
    # Data / output
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-s", "--size", type=int, default=None, help="Number of instances to test")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")
    parser.add_argument("--vrptw", action="store_true", help="Use VRPTW dataset with zero customer demands")
    parser.add_argument("--data_source", type=str, default="gfacs", choices=["gfacs", "rl4co"], help="Dataset and metric source")
    parser.add_argument("--data_dir", type=Path, default=None, help="Directory containing GFACS-format testDataset-*.pt files")
    parser.add_argument("--rl4co_root", type=Path, default=None, help="Path to the RL4CO repository if it is not installed")
    parser.add_argument("--rl4co_phase", type=str, default="test", choices=["train", "val", "test"], help="RL4CO dataset phase")
    parser.add_argument("--rl4co_file", type=Path, default=None, help="Optional RL4CO .npz dataset file")
    parser.add_argument("--rl4co_seed", type=int, default=1234, help="Seed used when RL4CO generates data")
    parser.add_argument("--rl4co_scale", action="store_true", help="Use RL4CO CVRPTW scaled generator")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for result txt/csv files")

    # FACO core
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse / default FACO candidate list size")
    parser.add_argument("-i", "--n_iter", type=int, default=10, help="Number of FACO iterations")
    parser.add_argument("--threads", type=int, default=None, help="OpenMP thread count for parallel C++ FACO sampling/update")
    parser.add_argument("-n", "--n_ants", type=int, default=100, help="Number of ants")
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
    parser.add_argument("--deep_nls", action="store_true", help="Enable deep LS inside NLS phase")
    parser.add_argument("--granular_mode", type=int, default=0, choices=[0, 1], help="FACO KNN mode: 0 euclidean, 1 spatio-temporal")
    parser.add_argument("--granular_wait_weight", type=float, default=0.2, help="Weight for minimum wait time in granular KNN")
    parser.add_argument("--granular_time_warp_weight", type=float, default=1.0, help="Weight for minimum time warp in granular KNN")

    # HGS / DeepLS
    parser.add_argument("--hgs_soft_deep_ls", action="store_true", help="Enable HGS-style penalized TW/capacity objective in deep LS")
    parser.add_argument("--hgs_soft_cheap_ls", action="store_true", help="Enable HGS-style penalized TW/capacity objective in cheap inter-route LS")
    parser.add_argument("--hgs_soft_intra_ls", action="store_true", help="Enable HGS-style penalized TW/capacity objective in intra-route LS")
    parser.add_argument("--hgs_deep_ls", action="store_true", help="Enable selective RELOCATE*/SWAP* deep local search")
    parser.add_argument("--hgs_deep_top_k", type=int, default=0, help="Run HGS deep LS post-sample only on top-k ants; 0 runs inline on every ant")
    parser.add_argument("--hgs_tw_penalty", type=float, default=10.0, help="Penalty weight for time-warp violation in HGS-style LS")
    parser.add_argument("--hgs_capacity_penalty", type=float, default=10.0, help="Penalty weight for capacity violation in HGS-style LS")
    parser.add_argument("--hgs_adaptive_penalty", action="store_true", help="Adapt HGS penalty weights from deep LS feasibility feedback")
    parser.add_argument("--hgs_target_feasible", type=float, default=0.8, help="Target feasible ratio for adaptive HGS penalty updates")
    parser.add_argument("--hgs_deep_rounds", type=int, default=1, help="Max accepted deep LS moves per LS call")
    parser.add_argument("--hgs_deep_route_pair_prune", action="store_true", help="Prune DeepLS route pairs by granular route overlap")
    parser.add_argument("--hgs_deep_route_pair_top_k", type=int, default=3, help="Max target routes kept per source route for DeepLS route-pair pruning")

    # Elite source management
    parser.add_argument("--elite_k", type=int, default=8, help="Number of quality-diverse elite source routes")
    parser.add_argument("--elite_min_diversity", type=float, default=0.15, help="Minimum edge distance between elite source routes")
    parser.add_argument("--elite_cost_tolerance", type=float, default=1.05, help="Maximum elite source cost ratio versus current best")
    parser.add_argument("--source_count", type=int, default=2, help="Number of elite routes retained for periodic source selection")
    parser.add_argument("--elite_source_period", type=int, default=1, help="Use an elite source every N iterations; 0 disables periodic elite source")
    parser.add_argument("--log_period", type=int, default=1, help="Record cost/diversity every N iterations")
    parser.add_argument("--pin_global_best_elite", action="store_true", help="Always keep global best as one elite archive route")

    # DeepACO prior
    parser.add_argument("--deepaco_pretrained", type=Path, default=None, help="Path to deepaco_trainer.py checkpoint used to generate sampling prior")
    parser.add_argument("--deepaco_device", type=str, default=None, help="Device for DeepACO prior computation; defaults to cuda:0 if available else cpu")
    parser.add_argument("--deepaco_prior_scale", type=float, default=1.0, help="Multiplier applied to DeepACO prior logits before sampling")
    parser.add_argument("--deepaco_prior_center", action="store_true", help="Row-center DeepACO prior logits before scaling")
    parser.add_argument("--deepaco_graph_mode", type=str, choices=["distance", "granular"], default="distance", help="Graph mode for DeepACO GNN input")
    parser.add_argument("--gfacs_pretrained", type=Path, default=None, help="Deprecated alias for --deepaco_pretrained")
    parser.add_argument("--gfacs_device", type=str, default=None, help="Deprecated alias for --deepaco_device")
    parser.add_argument("--gfacs_prior_scale", type=float, default=1.0, help="Deprecated alias for --deepaco_prior_scale")
    parser.add_argument("--gfacs_prior_center", action="store_true", help="Deprecated alias for --deepaco_prior_center")

    # PPO prior
    parser.add_argument("--ppo_pretrained", type=Path, default=None, help="Path to train_dynaco_ppo.py checkpoint used to generate dynamic sampling prior")
    parser.add_argument("--ppo_device", type=str, default=None, help="Device for PPO prior computation; defaults to cuda:0 if available else cpu")
    parser.add_argument("--ppo_norm_type", type=str, choices=["batch", "layer"], default="batch", help="PPO Net normalization type used by checkpoint")
    parser.add_argument("--ppo_edge_feature_mode", type=str, choices=["full", "static"], default="full", help="Edge features for PPO prior graph")
    parser.add_argument("--ppo_prior_refresh", type=str, choices=["outer", "sample"], default="outer", help="When to recompute PPO prior; outer matches train_dynaco_ppo.py validation")
    parser.add_argument("--ppo_prior_scale", type=float, default=1.0, help="Multiplier applied to PPO prior logits before sampling")
    parser.add_argument("--ppo_prior_center", action="store_true", help="Row-center PPO prior logits before scaling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        n_nodes=args.nodes,
        k_sparse=args.k_sparse,
        size=args.size,
        n_ants=args.n_ants,
        n_iter=args.n_iter,
        log_period=args.log_period,
        threads=args.threads,
        seed=args.seed,
        tam=args.tam,
        vrptw=args.vrptw,
        data_source=args.data_source,
        data_dir=args.data_dir,
        rl4co_root=args.rl4co_root,
        rl4co_phase=args.rl4co_phase,
        rl4co_file=args.rl4co_file,
        rl4co_seed=args.rl4co_seed,
        rl4co_scale=args.rl4co_scale,
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
        deep_nls=args.deep_nls,
        granular_mode=args.granular_mode,
        granular_wait_weight=args.granular_wait_weight,
        granular_time_warp_weight=args.granular_time_warp_weight,
        hgs_soft_deep_ls=args.hgs_soft_deep_ls,
        hgs_soft_cheap_ls=args.hgs_soft_cheap_ls,
        hgs_soft_intra_ls=args.hgs_soft_intra_ls,
        hgs_deep_ls=args.hgs_deep_ls,
        hgs_deep_top_k=args.hgs_deep_top_k,
        hgs_tw_penalty=args.hgs_tw_penalty,
        hgs_capacity_penalty=args.hgs_capacity_penalty,
        hgs_adaptive_penalty=args.hgs_adaptive_penalty,
        hgs_target_feasible=args.hgs_target_feasible,
        hgs_deep_rounds=args.hgs_deep_rounds,
        hgs_deep_route_pair_prune=args.hgs_deep_route_pair_prune,
        hgs_deep_route_pair_top_k=args.hgs_deep_route_pair_top_k,
        elite_k=args.elite_k,
        elite_min_diversity=args.elite_min_diversity,
        elite_cost_tolerance=args.elite_cost_tolerance,
        source_count=args.source_count,
        elite_source_period=args.elite_source_period,
        pin_global_best_elite=args.pin_global_best_elite,
        deepaco_pretrained=args.deepaco_pretrained,
        deepaco_device=args.deepaco_device,
        deepaco_prior_scale=args.deepaco_prior_scale,
        deepaco_prior_center=args.deepaco_prior_center,
        deepaco_graph_mode=args.deepaco_graph_mode,
        gfacs_pretrained=args.gfacs_pretrained,
        gfacs_device=args.gfacs_device,
        gfacs_prior_scale=args.gfacs_prior_scale,
        gfacs_prior_center=args.gfacs_prior_center,
        ppo_pretrained=args.ppo_pretrained,
        ppo_device=args.ppo_device,
        ppo_norm_type=args.ppo_norm_type,
        ppo_edge_feature_mode=args.ppo_edge_feature_mode,
        ppo_prior_refresh=args.ppo_prior_refresh,
        ppo_prior_scale=args.ppo_prior_scale,
        ppo_prior_center=args.ppo_prior_center,
    )
