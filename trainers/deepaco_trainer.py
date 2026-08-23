import os
import random
import time
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch

from envs.gfacs_data import gen_instance, gen_pyg_data, load_test_dataset
from evaluation import faco_test
from models.gfacs_net import Net
from solvers.faco import MFACO_CVRPTW, set_faco_cpp_threads
from trainers.dynaco_ppo_trainer import replay_logp_from_trace

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional for local smoke tests
    wandb = None

EPS = 1e-10
T = 5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_WANDB = False
TAM = False
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FACO_PARAMS = {
    "log_period": 1,
    "threads": 1,
    "backup_list_size": 64,
    "min_new_edges": 8,
    "decay": 0.9,
    "alpha": 1.0,
    "p_best": 0.05,
    "use_local_search": True,
    "disable_heuristic": False,
    "extend_ls": False,
    "smooth_mmas": False,
    "fixed_steps": 0,
    "nls": False,
    "T_nls": 10,
    "deep_nls": False,
    "granular_mode": 0,
    "granular_wait_weight": 0.2,
    "granular_time_warp_weight": 1.0,
    "hgs_soft_deep_ls": False,
    "hgs_soft_cheap_ls": False,
    "hgs_soft_intra_ls": False,
    "hgs_deep_ls": False,
    "hgs_deep_top_k": 0,
    "hgs_tw_penalty": 10.0,
    "hgs_capacity_penalty": 10.0,
    "hgs_adaptive_penalty": False,
    "hgs_target_feasible": 0.8,
    "hgs_deep_rounds": 1,
    "hgs_deep_route_pair_prune": False,
    "hgs_deep_route_pair_top_k": 3,
    "parallel_traced": False,
    "deepaco_prior_scale": 1.0,
    "deepaco_prior_center": False,
    "deepaco_graph_mode": "distance",
}


def deepaco_reinforce_loss(
    costs: torch.Tensor,
    log_probs: torch.Tensor,
    reward_costs: torch.Tensor | None = None,
) -> torch.Tensor:
    """DeepACO REINFORCE loss with a per-instance mean-cost baseline."""
    costs_for_reward = costs if reward_costs is None else reward_costs
    advantages = costs_for_reward - costs_for_reward.mean()
    path_log_probs = log_probs if log_probs.dim() == 1 else log_probs.sum(dim=0)
    return torch.sum(advantages.detach() * path_log_probs) / costs.numel()


def merged_faco_params(faco_params: dict | None = None) -> dict:
    return {**DEFAULT_FACO_PARAMS, **(faco_params or {})}


def faco_solver_kwargs(params: dict, n_ants: int, cand_list_size: int) -> dict:
    return {
        "n_ants": n_ants,
        "cand_list_size": cand_list_size,
        "backup_list_size": params["backup_list_size"],
        "min_new_edges": params["min_new_edges"],
        "decay": params["decay"],
        "alpha": params["alpha"],
        "p_best": params["p_best"],
        "use_local_search": params["use_local_search"],
        "disable_heuristic": params["disable_heuristic"],
        "extend_ls": params["extend_ls"],
        "smooth_mmas": params["smooth_mmas"],
        "device": "cpu",
        "fixed_steps": params["fixed_steps"],
        "nls": params["nls"],
        "T_nls": params["T_nls"],
        "deep_nls": params["deep_nls"],
        "granular_mode": params["granular_mode"],
        "granular_wait_weight": params["granular_wait_weight"],
        "granular_time_warp_weight": params["granular_time_warp_weight"],
        "hgs_soft_deep_ls": params["hgs_soft_deep_ls"],
        "hgs_soft_cheap_ls": params["hgs_soft_cheap_ls"],
        "hgs_soft_intra_ls": params["hgs_soft_intra_ls"],
        "hgs_deep_ls": params["hgs_deep_ls"],
        "hgs_deep_top_k": params["hgs_deep_top_k"],
        "hgs_tw_penalty": params["hgs_tw_penalty"],
        "hgs_capacity_penalty": params["hgs_capacity_penalty"],
        "hgs_adaptive_penalty": params["hgs_adaptive_penalty"],
        "hgs_target_feasible": params["hgs_target_feasible"],
        "hgs_deep_rounds": params["hgs_deep_rounds"],
        "hgs_deep_route_pair_prune": params["hgs_deep_route_pair_prune"],
        "hgs_deep_route_pair_top_k": params["hgs_deep_route_pair_top_k"],
    }


def build_faco_solver(demands, positions, windows, n_ants: int, cand_list_size: int, params: dict):
    return MFACO_CVRPTW(
        positions,
        demands,
        windows,
        capacity=1.0,
        **faco_solver_kwargs(params, n_ants, cand_list_size),
    )


def sparse_deepaco_prior_tensor(
    model,
    pyg_data,
    solver,
    prior_scale: float = 1.0,
    prior_center: bool = False,
) -> torch.Tensor:
    heuristic_mat = model.reshape(pyg_data, model(pyg_data))
    nn_list = torch.as_tensor(np.asarray(solver.nn_list), dtype=torch.long, device=heuristic_mat.device)
    rows = torch.arange(nn_list.shape[0], device=heuristic_mat.device).unsqueeze(1)
    sparse_heuristic = heuristic_mat[rows, nn_list]
    prior = torch.log(sparse_heuristic.to(torch.float32).clamp_min(EPS))
    if prior_center:
        prior = prior - prior.mean(dim=1, keepdim=True)
    if prior_scale != 1.0:
        prior = prior * float(prior_scale)
    return prior


def train_instance(
    model,
    optimizer,
    data,
    n_ants,
    invtemp=1.0,
    use_ls_reward=False,
    it=0,
    local_search_params=None,
    faco_params=None,
    k_sparse=None,
):
    model.train()
    params = merged_faco_params(faco_params)
    cand_list_size = k_sparse or params.get("cand_list_size") or n_ants

    train_mean_cost = 0.0
    train_min_cost = 0.0
    train_mean_reward_cost = 0.0
    train_min_reward_cost = 0.0
    train_entropy = 0.0

    sum_loss = torch.tensor(0.0, device=DEVICE)
    count = 0

    for pyg_data, demands, distances, positions, windows in data:
        solver = build_faco_solver(demands, positions, windows, n_ants, cand_list_size, params)
        if params["deepaco_graph_mode"] == "granular":
            pyg_data = faco_test.gen_granular_pyg_data(demands, distances, windows, solver, DEVICE)
        tau = solver.pheromone_sparse.detach().clone().to(DEVICE)
        eta = solver.h_sparse_torch.detach().clone().to(DEVICE)
        prior_logits = sparse_deepaco_prior_tensor(
            model,
            pyg_data,
            solver,
            prior_scale=params["deepaco_prior_scale"],
            prior_center=params["deepaco_prior_center"],
        )
        costs, _, _, _, traces, _, _, _, _ = solver.sample(
            require_prob=True,
            prior=prior_logits.detach().cpu().numpy(),
            parallel_traced=params["parallel_traced"],
        )
        costs = torch.as_tensor(costs, dtype=torch.float32, device=DEVICE)
        replay_logp, _, _ = replay_logp_from_trace(
            traces,
            tau,
            eta,
            prior_logits,
            alpha=params["alpha"],
            disable_heuristic=params["disable_heuristic"],
            return_decision_counts=True,
        )
        reward_costs = costs
        loss = deepaco_reinforce_loss(costs, replay_logp, reward_costs=reward_costs)
        sum_loss += loss
        count += 1

        if USE_WANDB:
            train_mean_cost += costs.mean().item()
            train_min_cost += costs.min().item()
            train_mean_reward_cost += reward_costs.mean().item()
            train_min_reward_cost += reward_costs.min().item()
            prob_prior = torch.softmax(prior_logits, dim=1)
            train_entropy += (-(prob_prior * torch.log(prob_prior.clamp_min(EPS))).sum(dim=1).mean()).item()

    sum_loss = sum_loss / count

    optimizer.zero_grad()
    sum_loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3.0, norm_type=2)
    optimizer.step()

    if USE_WANDB and wandb is not None:
        wandb.log(
            {
                "train_mean_cost": train_mean_cost / count,
                "train_min_cost": train_min_cost / count,
                "train_mean_reward_cost": train_mean_reward_cost / count,
                "train_min_reward_cost": train_min_reward_cost / count,
                "train_entropy": train_entropy / count,
                "train_loss": sum_loss.item(),
                "invtemp": invtemp,
                "use_ls_reward": False,
            },
            step=it,
        )


@torch.no_grad()
def infer_instance(
    model,
    pyg_data,
    demands,
    distances,
    positions,
    windows,
    n_ants,
    local_search_params=None,
    faco_params=None,
    k_sparse=None,
):
    model.eval()
    params = merged_faco_params(faco_params)
    cand_list_size = k_sparse or params.get("cand_list_size") or n_ants
    results, diversities, *_ = faco_test.infer_instance(
        demands,
        positions,
        windows,
        n_ants=n_ants,
        n_iter=params.get("val_n_iter", T),
        threads=params["threads"],
        seed=params.get("seed", 0),
        cand_list_size=cand_list_size,
        backup_list_size=params["backup_list_size"],
        min_new_edges=params["min_new_edges"],
        decay=params["decay"],
        alpha=params["alpha"],
        p_best=params["p_best"],
        use_local_search=params["use_local_search"],
        disable_heuristic=params["disable_heuristic"],
        extend_ls=params["extend_ls"],
        smooth_mmas=params["smooth_mmas"],
        fixed_steps=params["fixed_steps"],
        nls=params["nls"],
        T_nls=params["T_nls"],
        deep_nls=params["deep_nls"],
        log_period=params["log_period"],
        granular_mode=params["granular_mode"],
        granular_wait_weight=params["granular_wait_weight"],
        granular_time_warp_weight=params["granular_time_warp_weight"],
        hgs_soft_deep_ls=params["hgs_soft_deep_ls"],
        hgs_soft_cheap_ls=params["hgs_soft_cheap_ls"],
        hgs_soft_intra_ls=params["hgs_soft_intra_ls"],
        hgs_deep_ls=params["hgs_deep_ls"],
        hgs_deep_top_k=params["hgs_deep_top_k"],
        hgs_tw_penalty=params["hgs_tw_penalty"],
        hgs_capacity_penalty=params["hgs_capacity_penalty"],
        hgs_adaptive_penalty=params["hgs_adaptive_penalty"],
        hgs_target_feasible=params["hgs_target_feasible"],
        hgs_deep_rounds=params["hgs_deep_rounds"],
        hgs_deep_route_pair_prune=params["hgs_deep_route_pair_prune"],
        hgs_deep_route_pair_top_k=params["hgs_deep_route_pair_top_k"],
        deepaco_model=model,
        deepaco_k_sparse=cand_list_size,
        deepaco_device=DEVICE,
        deepaco_prior_scale=params["deepaco_prior_scale"],
        deepaco_prior_center=params["deepaco_prior_center"],
        deepaco_graph_mode=params["deepaco_graph_mode"],
        distances=distances,
    )
    return np.array([float(results[-1]), float(diversities[-1])])


def generate_traindata(count, n_node, k_sparse, vrptw=False):
    for _ in range(count):
        instance = demands, distance, _, windows = gen_instance(n_node, DEVICE, tam=TAM, vrptw=vrptw)
        yield gen_pyg_data(demands, distance, windows, DEVICE, k_sparse), *instance


def train_epoch(
    n_node,
    k_sparse,
    n_ants,
    epoch,
    steps_per_epoch,
    net,
    optimizer,
    batch_size=1,
    invtemp=1.0,
    use_ls_reward=False,
    local_search_params=None,
    vrptw=False,
    faco_params=None,
):
    for i in tqdm(range(steps_per_epoch), desc="Train", dynamic_ncols=True):
        it = (epoch - 1) * steps_per_epoch + i
        data = generate_traindata(batch_size, n_node, k_sparse, vrptw=vrptw)
        train_instance(net, optimizer, data, n_ants, invtemp, use_ls_reward, it, local_search_params, faco_params, k_sparse)


@torch.no_grad()
def validation(val_list, n_ants, net, epoch, steps_per_epoch, local_search_params=None, faco_params=None, k_sparse=None):
    stats = []
    for data, demands, distances, positions, windows in tqdm(val_list, desc="Val", dynamic_ncols=True):
        stats.append(infer_instance(net, data, demands, distances, positions, windows, n_ants, local_search_params, faco_params, k_sparse))
    avg_stats = [i.item() for i in np.stack(stats).mean(0)]

    print(f"epoch {epoch}:", avg_stats)
    if USE_WANDB and wandb is not None:
        wandb.log(
            {
                "val_faco_final_cost": avg_stats[0],
                "val_faco_final_diversity": avg_stats[1],
                "epoch": epoch,
            },
            step=epoch * steps_per_epoch,
        )

    return avg_stats[0]


def train(
    n_nodes,
    k_sparse,
    n_ants,
    n_val_ants,
    steps_per_epoch,
    epochs,
    lr=1e-4,
    batch_size=3,
    val_size=None,
    val_interval=5,
    pretrained=None,
    savepath="pretrained/cvrptw-deepaco",
    run_name="",
    invtemp_schedule_params=(1.0, 1.0, 5),
    use_ls_reward=False,
    local_search_params=None,
    vrptw=False,
    faco_params=None,
):
    savepath = os.path.join(savepath, str(n_nodes), run_name)
    os.makedirs(savepath, exist_ok=True)

    net = Net(gfn=False).to(DEVICE)
    if pretrained:
        net.load_state_dict(torch.load(pretrained, map_location=DEVICE))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr * 0.1)

    val_list = load_test_dataset(n_nodes, k_sparse, DEVICE, TAM, vrptw=vrptw, data_dir=ROOT / "data" / "cvrptw")
    val_list = val_list[:(val_size or len(val_list))]

    best_result = validation(val_list, n_val_ants, net, 0, steps_per_epoch, local_search_params, faco_params, k_sparse)

    sum_time = 0
    for epoch in range(1, epochs + 1):
        invtemp_min, invtemp_max, invtemp_flat_epochs = invtemp_schedule_params
        denom = max(epochs - invtemp_flat_epochs, 1)
        invtemp = invtemp_min + (invtemp_max - invtemp_min) * min((epoch - 1) / denom, 1.0)

        start = time.time()
        train_epoch(
            n_nodes,
            k_sparse,
            n_ants,
            epoch,
            steps_per_epoch,
            net,
            optimizer,
            batch_size,
            invtemp,
            use_ls_reward,
            local_search_params,
            vrptw,
            faco_params,
        )
        sum_time += time.time() - start

        if epoch % val_interval == 0:
            curr_result = validation(val_list, n_val_ants, net, epoch, steps_per_epoch, local_search_params, faco_params, k_sparse)
            if curr_result < best_result:
                torch.save(net.state_dict(), os.path.join(savepath, "best.pt"))
                best_result = curr_result

            torch.save(net.state_dict(), os.path.join(savepath, f"{epoch}.pt"))

        scheduler.step()

    print("\ntotal training duration:", sum_time)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DeepACO for CVRPTW using FACO sampling and inference.")
    parser.add_argument("nodes", metavar="N", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="Sparse candidate-list size")
    parser.add_argument("-l", "--lr", metavar="eta", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-d", "--device", type=str, default=("cuda:0" if torch.cuda.is_available() else "cpu"), help="Device for training NNs")
    parser.add_argument("-p", "--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("-a", "--ants", type=int, default=20, help="Number of ants for training")
    parser.add_argument("-va", "--val_ants", type=int, default=50, help="Number of ants for validation")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Steps per epoch")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Epochs to run")
    parser.add_argument("-v", "--val_size", type=int, default=10, help="Number of instances for validation")
    parser.add_argument("-o", "--output", type=str, default="pretrained/cvrptw-deepaco", help="Directory to store checkpoints")
    parser.add_argument("--val_interval", type=int, default=5, help="Interval to validate model")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default="", help="Run name")
    parser.add_argument("--invtemp_min", type=float, default=1.0, help="Kept for schedule compatibility")
    parser.add_argument("--invtemp_max", type=float, default=1.0, help="Kept for schedule compatibility")
    parser.add_argument("--invtemp_flat_epochs", type=int, default=5, help="Kept for schedule compatibility")
    parser.add_argument("--use_ls_reward", action="store_true", help="Deprecated: FACO sample costs are used for reward")
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")
    parser.add_argument("--vrptw", action="store_true", help="Use VRPTW mode by setting all customer demands to zero")
    parser.add_argument("--n_cpus", type=int, default=1, help="Deprecated PyVRP LS compatibility parameter")
    parser.add_argument("--max_trials", type=int, default=10, help="Deprecated PyVRP LS compatibility parameter")
    parser.add_argument("--load_penalty", type=int, default=20, help="Deprecated PyVRP LS compatibility parameter")
    parser.add_argument("--tw_penalty", type=int, default=20, help="Deprecated PyVRP LS compatibility parameter")
    parser.add_argument("--nb_granular", type=int, default=None, help="Deprecated PyVRP LS compatibility parameter")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--val_n_iter", type=int, default=T, help="FACO validation iterations")
    parser.add_argument("--log_period", type=int, default=1, help="FACO validation log period")
    parser.add_argument("--threads", type=int, default=1, help="FACO C++ threads")
    parser.add_argument("--backup_list_size", type=int, default=64)
    parser.add_argument("--min_new_edges", type=int, default=8)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--p_best", type=float, default=0.05)
    parser.add_argument("--disable_local_search", action="store_true")
    parser.add_argument("--disable_heuristic", action="store_true")
    parser.add_argument("--extend_ls", action="store_true")
    parser.add_argument("--smooth_mmas", action="store_true")
    parser.add_argument("--fixed_steps", type=int, default=0)
    parser.add_argument("--nls", action="store_true")
    parser.add_argument("--T_nls", type=int, default=10)
    parser.add_argument("--deep_nls", action="store_true")
    parser.add_argument("--granular_mode", type=int, default=0)
    parser.add_argument("--granular_wait_weight", type=float, default=0.2)
    parser.add_argument("--granular_time_warp_weight", type=float, default=1.0)
    parser.add_argument("--hgs_soft_deep_ls", action="store_true")
    parser.add_argument("--hgs_soft_cheap_ls", action="store_true")
    parser.add_argument("--hgs_soft_intra_ls", action="store_true")
    parser.add_argument("--hgs_deep_ls", action="store_true")
    parser.add_argument("--hgs_deep_top_k", type=int, default=0)
    parser.add_argument("--hgs_tw_penalty", type=float, default=10.0)
    parser.add_argument("--hgs_capacity_penalty", type=float, default=10.0)
    parser.add_argument("--hgs_adaptive_penalty", action="store_true")
    parser.add_argument("--hgs_target_feasible", type=float, default=0.8)
    parser.add_argument("--hgs_deep_rounds", type=int, default=1)
    parser.add_argument("--hgs_deep_route_pair_prune", action="store_true")
    parser.add_argument("--hgs_deep_route_pair_top_k", type=int, default=3)
    parser.add_argument("--parallel_traced", action="store_true")
    parser.add_argument("--deepaco_prior_scale", type=float, default=1.0)
    parser.add_argument("--deepaco_prior_center", action="store_true")
    parser.add_argument("--deepaco_graph_mode", type=str, choices=["distance", "granular"], default="distance")

    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 5
    if args.nb_granular is None:
        args.nb_granular = args.nodes // 2

    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    USE_WANDB = not args.disable_wandb
    TAM = args.tam
    set_faco_cpp_threads(args.threads)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_name = f"[{args.run_name}]" if args.run_name else ""
    mode_name = f"{'tam-' if args.tam else ''}{'vrptw' if args.vrptw else 'cvrptw'}"
    run_name += f"deepaco-{mode_name}{args.nodes}_sd{args.seed}"
    pretrained_name = (
        args.pretrained.replace("../pretrained/cvrptw/", "").replace("/", "_").replace(".pt", "")
        if args.pretrained is not None else None
    )
    run_name += f"{'' if pretrained_name is None else '_fromckpt-' + pretrained_name}"

    if USE_WANDB:
        if wandb is None:
            raise ImportError("wandb is required unless --disable_wandb is set")
        wandb.init(project="deepaco-cvrptw", name=run_name)
        wandb.config.update(args)
        wandb.config.update({"T": T, "model": "DeepACO-FACO", "tam": TAM, "vrptw": args.vrptw})

    local_search_params = {
        "n_cpus": args.n_cpus,
        "max_trials": args.max_trials,
        "neighbourhood_params": {"nb_granular": args.nb_granular},
        "cost_evaluator_params": {"load_penalty": args.load_penalty, "tw_penalty": args.tw_penalty},
    }
    faco_params = {
        "val_n_iter": args.val_n_iter,
        "log_period": args.log_period,
        "threads": args.threads,
        "backup_list_size": args.backup_list_size,
        "min_new_edges": args.min_new_edges,
        "decay": args.decay,
        "alpha": args.alpha,
        "p_best": args.p_best,
        "use_local_search": not args.disable_local_search,
        "disable_heuristic": args.disable_heuristic,
        "extend_ls": args.extend_ls,
        "smooth_mmas": args.smooth_mmas,
        "fixed_steps": args.fixed_steps,
        "nls": args.nls,
        "T_nls": args.T_nls,
        "deep_nls": args.deep_nls,
        "granular_mode": args.granular_mode,
        "granular_wait_weight": args.granular_wait_weight,
        "granular_time_warp_weight": args.granular_time_warp_weight,
        "hgs_soft_deep_ls": args.hgs_soft_deep_ls,
        "hgs_soft_cheap_ls": args.hgs_soft_cheap_ls,
        "hgs_soft_intra_ls": args.hgs_soft_intra_ls,
        "hgs_deep_ls": args.hgs_deep_ls,
        "hgs_deep_top_k": args.hgs_deep_top_k,
        "hgs_tw_penalty": args.hgs_tw_penalty,
        "hgs_capacity_penalty": args.hgs_capacity_penalty,
        "hgs_adaptive_penalty": args.hgs_adaptive_penalty,
        "hgs_target_feasible": args.hgs_target_feasible,
        "hgs_deep_rounds": args.hgs_deep_rounds,
        "hgs_deep_route_pair_prune": args.hgs_deep_route_pair_prune,
        "hgs_deep_route_pair_top_k": args.hgs_deep_route_pair_top_k,
        "parallel_traced": args.parallel_traced,
        "deepaco_prior_scale": args.deepaco_prior_scale,
        "deepaco_prior_center": args.deepaco_prior_center,
        "deepaco_graph_mode": args.deepaco_graph_mode,
        "seed": args.seed,
    }

    train(
        args.nodes,
        args.k_sparse,
        args.ants,
        args.val_ants,
        args.steps,
        args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_size=args.val_size,
        val_interval=args.val_interval,
        pretrained=args.pretrained,
        savepath=args.output,
        run_name=run_name,
        invtemp_schedule_params=(args.invtemp_min, args.invtemp_max, args.invtemp_flat_epochs),
        use_ls_reward=args.use_ls_reward,
        local_search_params=local_search_params,
        vrptw=args.vrptw,
        faco_params=faco_params,
    )
