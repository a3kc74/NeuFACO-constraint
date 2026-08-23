from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

from envs.cvrptw_env import build_solver, build_training_solver, generate_traindata, gen_pyg_data, load_val_dataset
from models.faco_net import Net
from solvers.faco import set_faco_cpp_threads
from trainers.ppo_replay import replay_logp_batch

EPS = 1e-10
T = 10
DEVICE = 'cpu'
USE_WANDB = False



def configure_cpp_threads(threads):
    if threads is None:
        return 'default'
    if threads < 1:
        raise ValueError('threads must be >= 1')
    set_faco_cpp_threads(int(threads))
    return str(int(threads))


def _to_tensor(x, device=None):
    return torch.as_tensor(x, dtype=torch.float32, device=device or DEVICE)


def train_instance(
    model,
    optimizer,
    data,
    n_ants,
    cost_w=0.0,
    invtemp=1.0,
    guided_exploration=False,
    shared_energy_norm=False,
    it=0,
    clip_ratio=0.2,
    ppo_epochs=4,
    value_coeff=0.5,
    entropy_coeff=0.01,
    cand_list_size=32,
    backup_list_size=64,
    min_new_edges=8,
    use_local_search=False,
    fixed_steps=0,
    extend_ls=False,
    nls=False,
    T_nls=10,
    deep_nls=False,
):
    model.train()
    all_experiences = []
    metrics = {
        'train_mean_cost': 0.0,
        'train_min_cost': 0.0,
        'train_mean_cost_raw': 0.0,
        'train_min_cost_raw': 0.0,
        'train_mean_reward': 0.0,
        'train_mean_reward_raw': 0.0,
        'train_mean_reward_ls': 0.0,
        'train_new_edges': 0.0,
        'train_feasibility_rate': 0.0,
    }
    sample_time_sec = 0.0
    backprop_time_sec = 0.0

    for pyg_data, instance in data:
        pyg_data = pyg_data.to(DEVICE)
        with torch.no_grad():
            heu_vec, value = model(pyg_data, return_value=True)
            prior_logits = model.reshape(pyg_data, heu_vec)

        solver = build_training_solver(
            instance,
            n_ants=n_ants,
            cand_list_size=cand_list_size,
            backup_list_size=backup_list_size,
            min_new_edges=min_new_edges,
            use_local_search=use_local_search,
            fixed_steps=fixed_steps,
            extend_ls=extend_ls,
            nls=nls,
            T_nls=T_nls,
            deep_nls=deep_nls,
            device=DEVICE,
        )
        sample_start = time.perf_counter()
        costs, routes, _, old_logps, traces, costs_raw, _, new_edges, _ = solver.sample(
            require_prob=True,
            prior=(invtemp * prior_logits).detach().cpu().numpy(),
            parallel_traced=True,
        )
        sample_time_sec += time.perf_counter() - sample_start
        source_cost = float(solver.source_cost)
        costs_t = _to_tensor(costs)
        costs_raw_t = _to_tensor(costs_raw if len(costs_raw) else costs)
        returns_raw = (source_cost - costs_raw_t) / max(source_cost, EPS)
        returns_ls = (source_cost - costs_t) / max(source_cost, EPS)
        returns = (1.0 - cost_w) * returns_raw + cost_w * returns_ls
        baseline = returns.mean() if shared_energy_norm else value.detach().expand(n_ants)
        advantages = returns - baseline
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + EPS)

        all_experiences.append({
            'pyg_data': pyg_data,
            'traces': traces,
            'old_log_probs': _to_tensor(old_logps).detach(),
            'returns': returns.detach(),
            'advantages': advantages.detach(),
        })

        metrics['train_mean_cost'] += costs_t.mean().item()
        metrics['train_min_cost'] += costs_t.min().item()
        metrics['train_mean_cost_raw'] += costs_raw_t.mean().item()
        metrics['train_min_cost_raw'] += costs_raw_t.min().item()
        metrics['train_mean_reward'] += returns.mean().item()
        metrics['train_mean_reward_raw'] += returns_raw.mean().item()
        metrics['train_mean_reward_ls'] += returns_ls.mean().item()
        metrics['train_new_edges'] += float(np.asarray(new_edges).mean())
        metrics['train_feasibility_rate'] += 1.0

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clip_fraction = 0.0
    total_ratio_mean = 0.0
    updates = 0

    backprop_start = time.perf_counter()
    for _ in range(ppo_epochs):
        for experience in all_experiences:
            heu_vec, value = model(experience['pyg_data'], return_value=True)
            prior_logits = model.reshape(experience['pyg_data'], heu_vec)
            new_log_probs, entropy = replay_logp_batch(prior_logits, experience['traces'], invtemp=invtemp)
            old_log_probs = experience['old_log_probs']
            returns = experience['returns']
            advantages = experience['advantages']

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value.expand_as(returns), returns)
            entropy_mean = entropy.mean()
            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0, norm_type=2)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean()
                clip_fraction = ((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)).float().mean()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()
            total_approx_kl += approx_kl.item()
            total_clip_fraction += clip_fraction.item()
            total_ratio_mean += ratio.mean().item()
            updates += 1
    if torch.cuda.is_available() and str(DEVICE).startswith('cuda'):
        torch.cuda.synchronize()
    backprop_time_sec = time.perf_counter() - backprop_start

    count = max(len(all_experiences), 1)
    for key in list(metrics):
        metrics[key] /= count
    updates = max(updates, 1)
    metrics.update({
        'train_policy_loss': total_policy_loss / updates,
        'train_value_loss': total_value_loss / updates,
        'train_entropy': total_entropy / updates,
        'train_approx_kl': total_approx_kl / updates,
        'train_clip_fraction': total_clip_fraction / updates,
        'train_ratio_mean': total_ratio_mean / updates,
        'train_instance_sample_time_sec': sample_time_sec,
        'train_instance_backprop_time_sec': backprop_time_sec,
        'train_instance_sample_time_per_problem_sec': sample_time_sec / count,
        'train_instance_backprop_time_per_update_sec': backprop_time_sec / updates,
        'cost_w': cost_w,
        'invtemp': invtemp,
    })
    if USE_WANDB and wandb is not None:
        wandb.log(metrics, step=it)
    return metrics


def train_epoch(n_node, k_sparse, n_ants, epoch, steps_per_epoch, net, optimizer, batch_size=20, **kwargs):
    stats = None
    for i in tqdm(range(steps_per_epoch), desc='Train', dynamic_ncols=True):
        data = generate_traindata(batch_size, n_node, k_sparse, seed=(epoch * steps_per_epoch + i) * batch_size)
        stats = train_instance(net, optimizer, data, n_ants, it=(epoch - 1) * steps_per_epoch + i, cand_list_size=k_sparse, **kwargs)
    return stats


@torch.no_grad()
def infer_instance(
    model,
    pyg_data,
    instance,
    n_ants,
    cand_list_size=32,
    backup_list_size=64,
    min_new_edges=8,
    val_n_iter=10,
    val_mini_H=1,
    seed=0,
    instance_idx=0,
    decay=0.9,
    alpha=1.0,
    p_best=0.05,
    use_local_search=True,
    disable_heuristic=False,
    extend_ls=False,
    smooth_mmas=False,
    fixed_steps=0,
    nls=False,
    T_nls=10,
    deep_nls=False,
    **_unused,
):
    if val_n_iter < 1:
        raise ValueError('val_n_iter must be >= 1')
    if val_mini_H < 1:
        raise ValueError('val_mini_H must be >= 1')
    model.eval()
    pyg_data = pyg_data.to(DEVICE)
    heu_vec = model(pyg_data)
    prior_logits = model.reshape(pyg_data, heu_vec).detach().cpu().numpy()

    faco_solver = build_solver(
        instance,
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
        deep_nls=deep_nls,
        device=DEVICE,
    )
    faco_solver.seed_rng(seed + instance_idx)
    raw_mean = math.inf
    raw_best = math.inf
    best_1 = math.inf
    best_t = math.inf
    for t in range(val_n_iter):
        for mini_t in range(val_mini_H):
            costs, routes, *_ = faco_solver.sample(prior=prior_logits)
            costs_np = np.asarray(costs, dtype=np.float32)
            if t == 0 and mini_t == 0:
                raw_mean = float(costs_np.mean())
                raw_best = float(costs_np.min())
            best_idx = int(costs_np.argmin())
            best_cost = float(costs_np[best_idx])
            best_t = min(best_t, best_cost)
            faco_solver.update_pheromone(routes[best_idx], best_cost)
        if t == 0:
            best_1 = best_t
    return [raw_mean, raw_best, raw_best, best_1, best_t, 1.0]


@torch.no_grad()
def validation(val_list, n_ants, net, epoch, steps_per_epoch, **kwargs):
    infer_kwargs = dict(kwargs)
    stats = [
        infer_instance(net, pyg_data, instance, n_ants, instance_idx=idx, **infer_kwargs)
        for idx, (pyg_data, instance) in enumerate(tqdm(val_list, desc='Val', dynamic_ncols=True))
    ]
    avg_stats = np.asarray(stats, dtype=np.float32).mean(0).tolist()
    print(f'epoch {epoch}:', avg_stats)
    if USE_WANDB and wandb is not None:
        wandb.log({
            'val_mean_raw_cost': avg_stats[0],
            'val_best_raw_cost': avg_stats[1],
            'val_best_sample_cost': avg_stats[2],
            'val_best_faco_1': avg_stats[3],
            'val_best_faco_T': avg_stats[4],
            'val_feasibility_rate': avg_stats[5],
            'epoch': epoch,
        }, step=epoch * steps_per_epoch)
    return avg_stats[4]


def train(n_nodes, k_sparse, n_ants, n_val_ants, steps_per_epoch, epochs, lr=1e-3, batch_size=20, val_size=20, val_interval=1, pretrained=None, savepath='../pretrained/cvrptw_ppo', run_name='', cost_w_schedule_params=(0.0, 1.0, 5), invtemp_schedule_params=(1.0, 1.0, 5), **kwargs):
    cand_list_size = kwargs.pop('cand_list_size', None) or k_sparse
    val_only_kwargs = {'val_n_iter', 'val_mini_H', 'seed', 'decay', 'alpha', 'p_best', 'smooth_mmas'}
    train_kwargs = {key: value for key, value in kwargs.items() if key not in val_only_kwargs}
    print(f'Focused PPO training device: {DEVICE}')
    savepath = os.path.join(savepath, str(n_nodes), run_name)
    os.makedirs(savepath, exist_ok=True)
    net = Net(value_head=True).to(DEVICE)
    if pretrained:
        net.load_state_dict(torch.load(pretrained, map_location=DEVICE))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr * 0.1)
    val_list = load_val_dataset(n_nodes, cand_list_size, DEVICE, val_size=val_size)
    best_result = validation(val_list, n_val_ants, net, 0, steps_per_epoch, cand_list_size=cand_list_size, **kwargs)

    for epoch in range(1, epochs + 1):
        cost_w_min, cost_w_max, cost_w_flat_epochs = cost_w_schedule_params
        cost_w = cost_w_min + (cost_w_max - cost_w_min) * min((epoch - 1) / max(epochs - cost_w_flat_epochs, 1), 1.0)
        invtemp_min, invtemp_max, invtemp_flat_epochs = invtemp_schedule_params
        invtemp = invtemp_min + (invtemp_max - invtemp_min) * min((epoch - 1) / max(epochs - invtemp_flat_epochs, 1), 1.0)
        train_epoch(n_nodes, cand_list_size, n_ants, epoch, steps_per_epoch, net, optimizer, batch_size, cost_w=cost_w, invtemp=invtemp, **train_kwargs)
        scheduler.step()
        if epoch % val_interval == 0:
            result = validation(val_list, n_val_ants, net, epoch, steps_per_epoch, cand_list_size=cand_list_size, **kwargs)
            if result < best_result:
                best_result = result
                torch.save(net.state_dict(), os.path.join(savepath, 'best.pt'))
            torch.save(net.state_dict(), os.path.join(savepath, f'epoch-{epoch}.pt'))
    return net


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {'1', 'true', 'yes', 'on'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train focused-operator PPO for CVRPTW')
    parser.add_argument('nodes', metavar='N', type=int)
    parser.add_argument('-k', '--k_sparse', type=int, default=None)
    parser.add_argument('-l', '--lr', type=float, default=1e-3)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-p', '--pretrained', type=str, default=None)
    parser.add_argument('-a', '--ants', type=int, default=32)
    parser.add_argument('-va', '--val_ants', type=int, default=64)
    parser.add_argument('-b', '--batch_size', type=int, default=20)
    parser.add_argument('-s', '--steps', type=int, default=20)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-v', '--val_size', type=int, default=20)
    parser.add_argument('-o', '--output', type=str, default='../pretrained/cvrptw_ppo')
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--invtemp_min', type=float, default=1.0)
    parser.add_argument('--invtemp_max', type=float, default=1.0)
    parser.add_argument('--invtemp_flat_epochs', type=int, default=5)
    parser.add_argument('--cost_w_min', type=float, default=0.0)
    parser.add_argument('--cost_w_max', type=float, default=1.0)
    parser.add_argument('--cost_w_flat_epochs', type=int, default=5)
    parser.add_argument('--disable_shared_energy_norm', action='store_true')
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--value_coeff', type=float, default=0.5)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--cand_list_size', type=int, default=None)
    parser.add_argument('--backup_list_size', type=int, default=64)
    parser.add_argument('--min_new_edges', type=int, default=8)
    parser.add_argument('--val_n_iter', type=int, default=10)
    parser.add_argument('--val_mini_H', type=int, default=1)
    parser.add_argument('--decay', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--p_best', type=float, default=0.05)
    parser.add_argument('--use_local_search', type=parse_bool, default=False)
    parser.add_argument('--fixed_steps', type=int, default=0)
    parser.add_argument('--extend_ls', action='store_true')
    parser.add_argument('--smooth_mmas', action='store_true')
    parser.add_argument('--nls', action='store_true')
    parser.add_argument('--T_nls', type=int, default=10)
    parser.add_argument('--deep_nls', action='store_true')
    parser.add_argument('--threads', type=int, default=None, help='C++ backend OpenMP thread count')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = min(args.nodes // 5, 32)
    if args.cand_list_size is None:
        args.cand_list_size = args.k_sparse
    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    cpp_threads = configure_cpp_threads(args.threads)
    print(f'Focused PPO C++ backend threads: {cpp_threads}')
    USE_WANDB = not args.disable_wandb
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_name = f'[{args.run_name}]' if args.run_name else ''
    run_name += f'cvrptw{args.nodes}_sd{args.seed}'
    if USE_WANDB and wandb is not None:
        wandb.init(project='neufaco-cvrptw', name=run_name)
        wandb.config.update(args)
        wandb.config.update({'T': T, 'model': 'FocusedPPO'})

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
        cost_w_schedule_params=(args.cost_w_min, args.cost_w_max, args.cost_w_flat_epochs),
        invtemp_schedule_params=(args.invtemp_min, args.invtemp_max, args.invtemp_flat_epochs),
        shared_energy_norm=(not args.disable_shared_energy_norm),
        clip_ratio=args.clip_ratio,
        ppo_epochs=args.ppo_epochs,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        cand_list_size=args.cand_list_size,
        backup_list_size=args.backup_list_size,
        min_new_edges=args.min_new_edges,
        val_n_iter=args.val_n_iter,
        val_mini_H=args.val_mini_H,
        seed=args.seed,
        decay=args.decay,
        alpha=args.alpha,
        p_best=args.p_best,
        use_local_search=args.use_local_search,
        fixed_steps=args.fixed_steps,
        extend_ls=args.extend_ls,
        smooth_mmas=args.smooth_mmas,
        nls=args.nls,
        T_nls=args.T_nls,
        deep_nls=args.deep_nls,
    )
