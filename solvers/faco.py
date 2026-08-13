"""
MFACO CVRP/CVRPTW interface backed by the C++ `faco_opt` extension.

Usage:
    from solvers.faco import MFACO_CVRP, MFACO_CVRPTW

    solver = MFACO_CVRPTW(coords, demand, windows, capacity, n_ants=20, ...)
    costs, routes, *_ = solver.sample()
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch
from typing import Optional, List, Tuple, Any

# Ensure src is in path to find C++ extension if not installed globally
src_dir = Path(__file__).resolve().parents[1] / "cpp" / "faco" / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

try:
    import faco_opt
except ImportError:
    # Try importing from current directory if compiled in-place at root
    try:
        import faco_opt
    except ImportError:
        # Check if it is in src but import failed
        raise ImportError(
            "C++ backend 'faco_opt' not found. Please build the C++ extension: "
            "uv run python cpp/faco/setup.py build_ext --inplace. "
            "On Windows, install Microsoft C++ Build Tools first if compilation fails."
        )

def set_faco_cpp_threads(n_threads: int) -> None:
    """Set OpenMP thread count for the C++ backend."""
    faco_opt.set_num_threads(int(n_threads))

def _as_numpy_f32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(x)

def _as_numpy_i32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.int32)
    return np.ascontiguousarray(x)

def _normalize_prior(prior, n: int, k: int):
    prior = _as_numpy_f32(prior)
    if prior.ndim != 2:
        raise ValueError(f"prior must be 2D, got shape {prior.shape}")
    if prior.shape == (n, k):
        return prior
    if prior.shape == (n - 1, k):
        depot_prior = np.zeros((1, k), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate([depot_prior, prior], axis=0))
    raise RuntimeError(
        f"prior must be shape ({n}, {k}) or customer-only shape ({n - 1}, {k}), got {prior.shape}"
    )

class MFACO_CVRP:
    """
    Unified MFACO CVRP solver wrapping C++ backend.
    
    API is designed for FACO CVRP/CVRPTW benchmark usage.
    """
    
    def __init__(
        self,
        coords,          
        demand,          
        capacity: float,
        n_ants: int,
        cand_list_size: int = 32,
        backup_list_size: int = 64,
        min_new_edges: int = 8,
        decay: float = 0.9,
        alpha: float = 1.0,
        p_best: float = 0.05,
        use_local_search: bool = True,
        disable_heuristic: bool = False,
        extend_ls: bool = False,
        smooth_mmas: bool = False,
        device: str = "cpu",
        enable_torch_sync: bool = True,
        normalized_heuristic: bool = False,
        fixed_steps: int = 0,
        nls: bool = False,
        T_nls: int = 10,
        use_fts_checks: bool = True,
        **kwargs
    ):
        coords_np = _as_numpy_f32(coords)
        demand_np = _as_numpy_f32(demand)
        if demand_np.ndim != 1:
            raise ValueError("demand must be 1D")
        demand_np[0] = 0.0
        
        # Use _cpp as the backing native solver.
        self._cpp = faco_opt.MFACO_CVRP(
            coords_np,
            demand_np,
            float(capacity),
            int(n_ants),
            int(cand_list_size),
            int(backup_list_size),
            int(min_new_edges),
            float(decay),
            float(alpha),
            float(p_best),
            bool(use_local_search),
            bool(disable_heuristic),
            bool(extend_ls),
            bool(smooth_mmas),
            int(fixed_steps),
            bool(nls),
            int(T_nls),
        )
        self.device = device
        self._enable_torch_sync = enable_torch_sync
        self.alpha = alpha
        self.disable_heuristic = disable_heuristic
        self._cpp.use_fts_checks = bool(use_fts_checks)
        
        if normalized_heuristic and not disable_heuristic:
            h = np.asarray(self._cpp.heuristic_sparse_np)
            row_sums = h.sum(axis=1, keepdims=True)
            h_norm = h / (row_sums + 1e-12)
            np.copyto(h, h_norm)
        
        # Torch buffers mirrored from C++ sparse arrays.
        self._pheromone_sparse = torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(device)
        self._h_sparse_torch = torch.from_numpy(self._cpp.heuristic_sparse_np.copy()).to(device)
        self._nn_torch = torch.from_numpy(self._cpp.nn_list.copy()).to(device).long()

    # Properties
    @property
    def n(self) -> int: return self._cpp.n
    @property
    def m(self) -> int: return self._cpp.m
    @property
    def k(self) -> int: return self._cpp.k
    @property
    def n_ants(self) -> int: return self._cpp.n_ants

    @property
    def heuristic_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.heuristic_sparse_np)
    @property
    def nn_list(self) -> np.ndarray: return np.asarray(self._cpp.nn_list)
    @property
    def backup_list(self) -> np.ndarray: return np.asarray(self._cpp.backup_list)
    
    @property
    def pheromone_sparse(self) -> torch.Tensor:
        return self._pheromone_sparse
    
    @property
    def h_sparse_torch(self) -> torch.Tensor:
        """Alias for heuristic tensor."""
        return self._h_sparse_torch
    
    @property
    def nn_torch(self) -> torch.Tensor:
        return self._nn_torch

    @property
    def source_perm(self) -> np.ndarray:
        return np.asarray(self._cpp.source_route)
    
    @property
    def source_route(self) -> np.ndarray:
        return np.asarray(self._cpp.source_route)

    @property
    def best_route(self) -> np.ndarray:
        return np.asarray(self._cpp.best_route)
    
    @property
    def enable_torch_sync(self) -> bool:
        return self._enable_torch_sync

    def seed_rng(self, seed: int) -> None:
        self._cpp.seed_rng(int(seed))

    def sample(
        self,
        invtemp: float = 1.0,  # Kept for API compatibility (unused for CVRP)
        require_prob: bool = False,
        prior: Optional[Any] = None,
        parallel_traced: bool = False,
        return_decoded: bool = False,
    ):
        """
        Sample from C++ backend.
        
        Returns the benchmark tuple:
            (costs, flats, touched, logps, traces, costs_raw, flats_raw, new_edges_count, survival)
        
        Note: For CVRP, 'touched' contains decoded routes if return_decoded=True.
        """
        if prior is not None:
            prior = _normalize_prior(prior, self.n, self.k)
        
        costs, routes, decoded, logps, traces, costs_raw, routes_raw, new_edges_count, survival = self._cpp.sample(
            require_prob, prior, parallel_traced, return_decoded
        )

        if isinstance(survival, np.ndarray):
            survival = torch.from_numpy(survival).to(self.device)

        # Return format: (costs, routes, decoded, logps, traces, costs_raw, routes_raw, new_edges, survival)
        return costs, routes, decoded, logps, traces, costs_raw, routes_raw, new_edges_count, survival

    def update_pheromone(self, best_route, best_cost: float) -> None:
        """Update pheromone with the best depot-separated route."""
        p = _as_numpy_i32(best_route)
        self._cpp.update_pheromone_from_route(p, float(best_cost))
        if self._enable_torch_sync:
            self.sync_pheromone_to_torch()

    def set_source_route(self, route, cost: float) -> bool:
        """Set the construction source route without changing pheromone."""
        p = _as_numpy_i32(route)
        return bool(self._cpp.set_source_route(p, float(cost)))
    
    def _update_pheromone_from_flat(self, best_flat, best_cost: float) -> None:
        """Alias for update_pheromone."""
        self.update_pheromone(best_flat, best_cost)

    def reset_timings(self) -> None:
        self._cpp.reset_timings()

    def get_timings(self) -> dict:
        return self._cpp.get_timings()


class MFACO_CVRPTW(MFACO_CVRP):
    """
    MFACO CVRPTW solver wrapping the C++ backend.

    Matches GFACS in-memory semantics: coords/positions, normalized
    demand, capacity, and windows[:, 0:2] = [ready_time, due_time]. Service
    time is treated as zero and the objective is pure Euclidean travel cost.
    """

    def __init__(
        self,
        coords,
        demand,
        windows,
        capacity: float,
        n_ants: int,
        cand_list_size: int = 32,
        backup_list_size: int = 64,
        min_new_edges: int = 8,
        decay: float = 0.9,
        alpha: float = 1.0,
        p_best: float = 0.05,
        use_local_search: bool = True,
        disable_heuristic: bool = False,
        extend_ls: bool = False,
        smooth_mmas: bool = False,
        device: str = "cpu",
        enable_torch_sync: bool = True,
        normalized_heuristic: bool = False,
        fixed_steps: int = 0,
        nls: bool = False,
        T_nls: int = 10,
        use_fts_checks: bool = True,
        **kwargs
    ):
        coords_np = _as_numpy_f32(coords)
        demand_np = _as_numpy_f32(demand)
        windows_np = _as_numpy_f32(windows)
        if coords_np.ndim != 2 or coords_np.shape[1] != 2:
            raise ValueError(f"coords must have shape (n, 2), got {coords_np.shape}")
        if demand_np.ndim != 1 or demand_np.shape[0] != coords_np.shape[0]:
            raise ValueError("demand must be shape (n,) matching coords")
        if windows_np.ndim != 2 or windows_np.shape != (coords_np.shape[0], 2):
            raise ValueError(f"windows must have shape (n, 2), got {windows_np.shape}")
        demand_np = demand_np.copy()
        demand_np[0] = 0.0

        self._cpp = faco_opt.MFACO_CVRPTW(
            coords_np,
            demand_np,
            windows_np,
            float(capacity),
            int(n_ants),
            int(cand_list_size),
            int(backup_list_size),
            int(min_new_edges),
            float(decay),
            float(alpha),
            float(p_best),
            bool(use_local_search),
            bool(disable_heuristic),
            bool(extend_ls),
            bool(smooth_mmas),
            int(fixed_steps),
            bool(nls),
            int(T_nls),
        )
        self.device = device
        self._enable_torch_sync = enable_torch_sync
        self.alpha = alpha
        self.disable_heuristic = disable_heuristic
        self._cpp.use_fts_checks = bool(use_fts_checks)

        if normalized_heuristic and not disable_heuristic:
            h = np.asarray(self._cpp.heuristic_sparse_np)
            row_sums = h.sum(axis=1, keepdims=True)
            h_norm = h / (row_sums + 1e-12)
            np.copyto(h, h_norm)

        self._pheromone_sparse = torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(device)
        self._h_sparse_torch = torch.from_numpy(self._cpp.heuristic_sparse_np.copy()).to(device)
        self._nn_torch = torch.from_numpy(self._cpp.nn_list.copy()).to(device).long()

    def sync_pheromone_to_torch(self) -> None:
        phe_np = np.asarray(self._cpp.pheromone_sparse_np)
        self._pheromone_sparse.copy_(torch.from_numpy(phe_np).to(self.device))

    def prob_sparse_torch(self, invtemp: float = 1.0, prior: torch.Tensor = None) -> torch.Tensor:
        """Compute sparse transition probability tensor."""
        EPS = 1e-10
        tau = self._pheromone_sparse.clamp_min(EPS)
        logit = self.alpha * torch.log(tau)
        
        if not self.disable_heuristic:
            h = self._h_sparse_torch.clamp_min(EPS)
            if invtemp != 1.0:
                logit = logit + float(invtemp) * torch.log(h)
            else:
                logit = logit + torch.log(h)
        if prior is not None:
            logit = logit + prior
        return torch.exp(logit)

    def tau_nk_torch(self) -> torch.Tensor:
        return self._pheromone_sparse.clone()
        


