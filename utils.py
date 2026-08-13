from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent


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


def load_dataset(
    n_nodes: int,
    device: str,
    tam: bool = False,
    vrptw: bool = False,
    data_dir: str | Path | None = None,
):
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


def default_rl4co_root() -> Path:
    return ROOT_DIR.parent / "rl4co"


def import_rl4co(rl4co_root: str | Path | None = None):
    root = Path(rl4co_root) if rl4co_root is not None else default_rl4co_root()
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import importlib.metadata as importlib_metadata

        importlib_metadata.version("rl4co")
    except Exception:
        import importlib.metadata as importlib_metadata

        original_version = importlib_metadata.version

        def local_version(package_name):
            if package_name == "rl4co":
                return "0.0.local"
            return original_version(package_name)

        importlib_metadata.version = local_version
    from rl4co.envs import CVRPTWEnv

    return CVRPTWEnv


def make_rl4co_eval_td(env, td):
    batch_size = td.batch_size
    return env.reset(td, batch_size=batch_size)


def rl4co_actions_from_route(route) -> torch.Tensor:
    actions = torch.as_tensor(np.asarray(route, dtype=np.int64), dtype=torch.long)
    return actions.unsqueeze(0)


def rl4co_route_costs(env, eval_td, routes) -> np.ndarray:
    costs = []
    for route in routes:
        actions = rl4co_actions_from_route(route).to(eval_td.device)
        reward = env.get_reward(eval_td, actions)
        costs.append(float((-reward).detach().cpu().item()))
    return np.asarray(costs, dtype=np.float32)


def load_rl4co_dataset(
    n_nodes: int,
    size: int | None = None,
    phase: str = "test",
    filename: str | Path | None = None,
    data_dir: str | Path | None = None,
    rl4co_root: str | Path | None = None,
    seed: int = 1234,
    scale: bool = False,
):
    CVRPTWEnv = import_rl4co(rl4co_root)
    root = Path(rl4co_root) if rl4co_root is not None else default_rl4co_root()
    base_dir = Path(data_dir) if data_dir is not None else root / "data" / "cvrptw"
    env = CVRPTWEnv(
        generator_params={"num_loc": n_nodes, "loc_distribution": "uniform", "scale": scale},
        data_dir=str(base_dir),
        test_file=f"cvrptw{n_nodes}_test_seed1234.npz",
        val_file=f"cvrptw{n_nodes}_val_seed4321.npz",
        check_solution=True,
    )
    torch.manual_seed(seed)
    batch_size = size or 100
    if filename is not None:
        data_file = Path(filename)
    elif phase == "test":
        data_file = base_dir / f"cvrptw{n_nodes}_test_seed1234.npz"
    elif phase == "val":
        data_file = base_dir / f"cvrptw{n_nodes}_val_seed4321.npz"
    else:
        data_file = None

    if data_file is not None and data_file.is_file():
        from rl4co.data.utils import load_npz_to_tensordict

        td = load_npz_to_tensordict(str(data_file))
    else:
        td = env.generator(batch_size=batch_size)
    if size is not None:
        td = td[:size]

    test_list = []
    for idx in range(td.batch_size[0]):
        instance_td = td[idx : idx + 1]
        eval_td = make_rl4co_eval_td(env, instance_td.clone())
        positions = eval_td["locs"][0].detach().cpu()
        demands = torch.cat(
            [torch.zeros(1, dtype=eval_td["demand"].dtype), eval_td["demand"][0].detach().cpu()]
        )
        distances = torch.cdist(positions.float(), positions.float()).to(torch.double)
        distances[torch.arange(n_nodes + 1), torch.arange(n_nodes + 1)] = 1e-10
        windows = eval_td["time_windows"][0].detach().cpu()
        test_list.append(
            {
                "demands": demands,
                "distances": distances,
                "positions": positions,
                "windows": windows,
                "rl4co_env": env,
                "rl4co_eval_td": eval_td,
            }
        )
    return test_list


def normalize_dataset_item(item):
    if isinstance(item, dict):
        evaluator = None
        if "rl4co_env" in item and "rl4co_eval_td" in item:
            evaluator = lambda routes, env=item["rl4co_env"], eval_td=item[
                "rl4co_eval_td"
            ]: rl4co_route_costs(env, eval_td, routes)
        return (
            item["demands"],
            item["distances"],
            item["positions"],
            item["windows"],
            evaluator,
        )
    demands, distances, positions, windows = item
    return demands, distances, positions, windows, None
