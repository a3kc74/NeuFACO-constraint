from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from faco import MFACO_CVRPTW


def _as_tensor(x, device=None):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def route_distance(coords, route) -> float:
    coords_np = np.asarray(coords, dtype=np.float32)
    route_np = np.asarray(route, dtype=np.int64)
    if len(route_np) < 2:
        return 0.0
    diffs = coords_np[route_np[:-1]] - coords_np[route_np[1:]]
    return float(np.linalg.norm(diffs, axis=1).sum())


def generate_cvrptw_instance(n_nodes: int, seed: int | None = None, device=None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    coords = rng.random((n_nodes, 2), dtype=np.float32)
    coords[0] = np.array([0.5, 0.5], dtype=np.float32)
    demand = rng.uniform(0.02, 0.12, size=n_nodes).astype(np.float32)
    demand[0] = 0.0
    capacity = 1.0

    depot_dist = np.linalg.norm(coords - coords[0], axis=1).astype(np.float32)
    ready = rng.uniform(0.0, 0.45, size=n_nodes).astype(np.float32)
    width = rng.uniform(0.45, 0.9, size=n_nodes).astype(np.float32)
    due = np.maximum(ready + width, depot_dist * 2.0 + 0.2).astype(np.float32)
    ready[0] = 0.0
    due[0] = max(3.0, float(due.max() + depot_dist.max() + 1.0))
    windows = np.stack([ready, due], axis=1).astype(np.float32)

    return {
        'coords': _as_tensor(coords, device),
        'demand': _as_tensor(demand, device),
        'windows': _as_tensor(windows, device),
        'capacity': float(capacity),
    }



def normalize_instance(instance: Any, n_nodes: int | None = None, device=None) -> dict[str, Any]:
    if isinstance(instance, dict):
        return {
            'coords': _as_tensor(instance['coords'], device),
            'demand': _as_tensor(instance['demand'], device),
            'windows': _as_tensor(instance['windows'], device),
            'capacity': float(instance.get('capacity', 1.0)),
        }

    tensor = torch.as_tensor(instance, dtype=torch.float32, device=device)
    if tensor.ndim != 2:
        raise ValueError(f'CVRPTW instance tensor must be 2D, got shape {tuple(tensor.shape)}')

    inferred_nodes = tensor.shape[1]
    if n_nodes is not None and inferred_nodes != n_nodes + 1:
        inferred_nodes = tensor.shape[1]
    expected_rows = inferred_nodes + 5
    if tensor.shape[0] < expected_rows:
        raise ValueError(
            f'GFACS-format tensor must have at least n_nodes + 5 rows; got shape {tuple(tensor.shape)}'
        )

    return {
        'coords': tensor[1:3, :].T.contiguous(),
        'demand': tensor[0, :].contiguous(),
        'windows': tensor[-2:, :].T.contiguous(),
        'capacity': 1.0,
    }


def build_solver(
    instance: dict[str, Any],
    n_ants: int = 1,
    cand_list_size: int = 32,
    backup_list_size: int = 64,
    min_new_edges: int = 8,
    decay: float = 0.9,
    alpha: float = 0.0,
    p_best: float = 0.05,
    use_local_search: bool = False,
    disable_heuristic: bool = True,
    extend_ls: bool = False,
    smooth_mmas: bool = True,
    fixed_steps: int = 0,
    nls: bool = False,
    T_nls: int = 10,
    device: str = 'cpu',
):
    instance = normalize_instance(instance, device=device)
    return MFACO_CVRPTW(
        instance['coords'],
        instance['demand'],
        instance['windows'],
        instance['capacity'],
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
        fixed_steps=fixed_steps,
        nls=nls,
        T_nls=T_nls,
        device=device,
    )


def build_training_solver(instance: dict[str, Any], **kwargs):
    kwargs.setdefault('alpha', 0.0)
    kwargs.setdefault('disable_heuristic', True)
    kwargs.setdefault('use_local_search', False)
    return build_solver(instance, **kwargs)


def gen_pyg_data(
    instance: dict[str, Any],
    cand_list_size: int = 32,
    backup_list_size: int = 64,
    n_ants: int = 1,
    min_new_edges: int = 8,
    device: str | torch.device | None = None,
    return_solver: bool = False,
):
    instance = normalize_instance(instance, device=device)
    solver = build_training_solver(
        instance,
        n_ants=n_ants,
        cand_list_size=cand_list_size,
        backup_list_size=backup_list_size,
        min_new_edges=min_new_edges,
        device=str(device or 'cpu'),
    )
    coords = torch.as_tensor(instance['coords'], dtype=torch.float32, device=device)
    demand = torch.as_tensor(instance['demand'], dtype=torch.float32, device=device)
    windows = torch.as_tensor(instance['windows'], dtype=torch.float32, device=device)
    capacity = float(instance['capacity'])
    n_nodes = coords.shape[0]
    nn_list = torch.as_tensor(solver.nn_list, dtype=torch.long, device=device)
    k_sparse = nn_list.shape[1]

    time_scale = torch.clamp(windows[:, 1].max(), min=torch.tensor(1.0, device=windows.device))
    node_features = torch.stack([
        coords[:, 0],
        coords[:, 1],
        demand / max(capacity, 1e-8),
        windows[:, 0] / time_scale,
        windows[:, 1] / time_scale,
        (windows[:, 1] - windows[:, 0]) / time_scale,
        (torch.arange(n_nodes, device=coords.device) == 0).float(),
    ], dim=1)

    src = torch.arange(n_nodes, device=coords.device).repeat_interleave(k_sparse)
    dst = nn_list.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)

    edge_vec = coords[src] - coords[dst]
    dist = torch.linalg.norm(edge_vec, dim=1)
    dmax = torch.clamp(torch.cdist(coords, coords).max(), min=1e-8)
    depot = torch.zeros((), dtype=torch.long, device=coords.device)
    dist_i0 = torch.linalg.norm(coords[src] - coords[depot], dim=1)
    dist_0j = torch.linalg.norm(coords[depot] - coords[dst], dim=1)
    saving = dist_i0 + dist_0j - dist
    earliest_arrival = windows[src, 0] + dist
    tw_compat = (earliest_arrival <= windows[dst, 1]).float()
    slack = (windows[dst, 1] - earliest_arrival) / time_scale
    edge_attr = torch.stack([
        dist,
        dist / dmax,
        saving,
        tw_compat,
        slack,
        (dst == 0).float(),
    ], dim=1)

    data = Data(x=node_features.float(), edge_index=edge_index.long(), edge_attr=edge_attr.float())
    data.n_nodes = int(n_nodes)
    data.k_sparse = int(k_sparse)
    data.nn_list = nn_list
    if return_solver:
        return data, solver
    return data


def generate_traindata(batch_size, n_nodes, k_sparse, **kwargs):
    seed = kwargs.pop('seed', None)
    data = []
    for i in range(batch_size):
        inst_seed = None if seed is None else seed + i
        instance = generate_cvrptw_instance(n_nodes, seed=inst_seed)
        pyg_data = gen_pyg_data(instance, cand_list_size=k_sparse, **kwargs)
        data.append((pyg_data, instance))
    return data


def load_val_dataset(n_nodes, k_sparse, device='cpu', val_size=20, data_dir='../data/cvrptw'):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f'testDataset-{n_nodes}.pt')
    if os.path.exists(path):
        print(f'loading gfacs test set')
        dataset = torch.load(path, map_location=device, weights_only=False)
    else:
        dataset = [generate_cvrptw_instance(n_nodes, seed=10_000 + i, device=device) for i in range(val_size)]
        torch.save(dataset, path)
    val_list = []
    for instance in dataset[:val_size]:
        normalized = normalize_instance(instance, n_nodes=n_nodes, device=device)
        val_list.append((gen_pyg_data(normalized, cand_list_size=k_sparse, device=device), normalized))
    return val_list
