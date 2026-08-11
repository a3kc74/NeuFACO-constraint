import math
import os
import random
import time

from tqdm import tqdm
import numpy as np
import torch

from models.gfacs_net import Net
from solvers.gfacs_aco import ACO
from envs.gfacs_data import gen_pyg_data, load_val_dataset, gen_instance

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional for local smoke tests
    wandb = None


EPS = 1e-10
T = 5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_WANDB = False
TAM = False


def deepaco_reinforce_loss(costs: torch.Tensor, log_probs: torch.Tensor, reward_costs: torch.Tensor | None = None) -> torch.Tensor:
    """REINFORCE loss used by DeepACO, with a per-instance mean-cost baseline."""
    costs_for_reward = costs if reward_costs is None else reward_costs
    advantages = costs_for_reward - costs_for_reward.mean()
    return torch.sum(advantages.detach() * log_probs.sum(dim=0)) / log_probs.size(1)


def train_instance(
    model,
    optimizer,
    data,
    n_ants,
    invtemp=1.0,
    use_ls_reward=False,
    it=0,
    local_search_params=None,
):
    model.train()

    train_mean_cost = 0.0
    train_min_cost = 0.0
    train_mean_reward_cost = 0.0
    train_min_reward_cost = 0.0
    train_entropy = 0.0

    sum_loss = torch.tensor(0.0, device=DEVICE)
    count = 0

    for pyg_data, demands, distances, positions, windows in data:
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS

        aco = ACO(
            distances=distances.to(DEVICE),
            demands=demands.to(DEVICE),
            windows=windows.to(DEVICE),
            n_ants=n_ants,
            heuristic=heu_mat.to(DEVICE),
            device=DEVICE,
            use_local_search=use_ls_reward,
            local_search_params=local_search_params,
            positions=positions,
        )

        costs, log_probs, paths = aco.sample(invtemp=invtemp)
        reward_costs = costs
        if use_ls_reward:
            paths_ls = aco.local_search(paths, inference=False)
            reward_costs = aco.gen_path_costs(paths_ls)

        loss = deepaco_reinforce_loss(costs, log_probs, reward_costs=reward_costs)
        sum_loss += loss
        count += 1

        if USE_WANDB:
            train_mean_cost += costs.mean().item()
            train_min_cost += costs.min().item()
            train_mean_reward_cost += reward_costs.mean().item()
            train_min_reward_cost += reward_costs.min().item()

            normed_heumat = heu_mat / heu_mat.sum(dim=1, keepdim=True)
            entropy = -(normed_heumat * torch.log(normed_heumat)).sum(dim=1).mean()
            train_entropy += entropy.item()

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
                "use_ls_reward": use_ls_reward,
            },
            step=it,
        )


@torch.no_grad()
def infer_instance(model, pyg_data, demands, distances, positions, windows, n_ants, local_search_params=None):
    model.eval()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS

    aco = ACO(
        distances=distances,
        demands=demands,
        windows=windows,
        positions=positions,
        n_ants=n_ants,
        heuristic=heu_mat,
        device=DEVICE,
        use_local_search=True,
        local_search_params=local_search_params,
    )

    costs = aco.sample()[0]
    baseline = costs.mean().item()
    best_sample_cost = costs.min().item()

    best_aco_1, diversity_1, _ = aco.run(n_iterations=1)
    best_aco_T, diversity_T, _ = aco.run(n_iterations=T - 1)
    return np.array([baseline, best_sample_cost, best_aco_1, best_aco_T, diversity_1, diversity_T])


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
):
    for i in tqdm(range(steps_per_epoch), desc="Train", dynamic_ncols=True):
        it = (epoch - 1) * steps_per_epoch + i
        data = generate_traindata(batch_size, n_node, k_sparse, vrptw=vrptw)
        train_instance(net, optimizer, data, n_ants, invtemp, use_ls_reward, it, local_search_params)


@torch.no_grad()
def validation(val_list, n_ants, net, epoch, steps_per_epoch, local_search_params=None):
    stats = []
    for data, demands, distances, positions, windows in tqdm(val_list, desc="Val", dynamic_ncols=True):
        stats.append(infer_instance(net, data, demands, distances, positions, windows, n_ants, local_search_params))
    avg_stats = [i.item() for i in np.stack(stats).mean(0)]

    print(f"epoch {epoch}:", avg_stats)
    if USE_WANDB and wandb is not None:
        wandb.log(
            {
                "val_baseline": avg_stats[0],
                "val_best_sample_cost": avg_stats[1],
                "val_best_aco_1": avg_stats[2],
                "val_best_aco_T": avg_stats[3],
                "val_diversity_1": avg_stats[4],
                "val_diversity_T": avg_stats[5],
                "epoch": epoch,
            },
            step=epoch * steps_per_epoch,
        )

    return avg_stats[3]


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
    savepath="../pretrained/cvrptw-deepaco",
    run_name="",
    invtemp_schedule_params=(1.0, 1.0, 5),
    use_ls_reward=False,
    local_search_params=None,
    vrptw=False,
):
    savepath = os.path.join(savepath, str(n_nodes), run_name)
    os.makedirs(savepath, exist_ok=True)

    net = Net(gfn=False).to(DEVICE)
    if pretrained:
        net.load_state_dict(torch.load(pretrained, map_location=DEVICE))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr * 0.1)

    val_list = load_val_dataset(n_nodes, k_sparse, DEVICE, TAM, vrptw=vrptw)
    val_list = val_list[:(val_size or len(val_list))]

    best_result = validation(val_list, n_val_ants, net, 0, steps_per_epoch, local_search_params)

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
        )
        sum_time += time.time() - start

        if epoch % val_interval == 0:
            curr_result = validation(val_list, n_val_ants, net, epoch, steps_per_epoch, local_search_params)
            if curr_result < best_result:
                torch.save(net.state_dict(), os.path.join(savepath, "best.pt"))
                best_result = curr_result

            torch.save(net.state_dict(), os.path.join(savepath, f"{epoch}.pt"))

        scheduler.step()

    print("\ntotal training duration:", sum_time)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a DeepACO-style REINFORCE model for CVRPTW using GFACS ACO sampling/inference.")
    parser.add_argument("nodes", metavar="N", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-l", "--lr", metavar="eta", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-d", "--device", type=str, default=("cuda:0" if torch.cuda.is_available() else "cpu"), help="The device to train NNs")
    parser.add_argument("-p", "--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("-a", "--ants", type=int, default=20, help="Number of ants for training")
    parser.add_argument("-va", "--val_ants", type=int, default=50, help="Number of ants for validation")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Steps per epoch")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Epochs to run")
    parser.add_argument("-v", "--val_size", type=int, default=10, help="Number of instances for validation")
    parser.add_argument("-o", "--output", type=str, default="../pretrained/cvrptw-deepaco", help="The directory to store checkpoints")
    parser.add_argument("--val_interval", type=int, default=5, help="The interval to validate model")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default="", help="Run name")
    parser.add_argument("--invtemp_min", type=float, default=1.0, help="ACO sampling inverse temperature min")
    parser.add_argument("--invtemp_max", type=float, default=1.0, help="ACO sampling inverse temperature max")
    parser.add_argument("--invtemp_flat_epochs", type=int, default=5, help="Inverse temperature flat epochs")
    parser.add_argument("--use_ls_reward", action="store_true", help="Use PyVRP local-search-improved paths to compute REINFORCE reward costs")
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")
    parser.add_argument("--vrptw", action="store_true", help="Use VRPTW mode by setting all customer demands to zero")
    parser.add_argument("--n_cpus", type=int, default=1, help="Number of cpus for local search")
    parser.add_argument("--max_trials", type=int, default=10, help="Number of local search trials")
    parser.add_argument("--load_penalty", type=int, default=20, help="Initial load_penalty in training phase")
    parser.add_argument("--tw_penalty", type=int, default=20, help="Initial tw_penalty in training phase")
    parser.add_argument("--nb_granular", type=int, default=None, help="Granularity of neighbourhood search")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 5
    if args.nb_granular is None:
        args.nb_granular = args.nodes // 2

    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    USE_WANDB = not args.disable_wandb
    TAM = args.tam

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
        wandb.config.update({"T": T, "model": "DeepACO", "tam": TAM, "vrptw": args.vrptw})

    local_search_params = {
        "n_cpus": args.n_cpus,
        "max_trials": args.max_trials,
        "neighbourhood_params": {"nb_granular": args.nb_granular},
        "cost_evaluator_params": {"load_penalty": args.load_penalty, "tw_penalty": args.tw_penalty},
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
    )
