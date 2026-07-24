from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


EPS = 1e-10
DEVICE = 'cpu'
USE_WANDB = False


def parse_bool(value=True):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {'1', 'true', 'yes', 'on'}


def configure_cpp_threads(threads: int | None) -> str:
    from faco import set_faco_cpp_threads

    if threads is None:
        return 'default'
    if threads < 1:
        raise ValueError('threads must be >= 1')
    set_faco_cpp_threads(int(threads))
    return str(int(threads))


def _trace_tensor(traces, name: str, dtype, device):
    value = getattr(traces, name)
    if name == 'valid_mask':
        value = np.asarray(value, dtype=np.uint64).astype(np.int64)
    return torch.as_tensor(value, dtype=dtype, device=device)


def replay_logp_from_trace(
    traces,
    tau: torch.Tensor,
    eta: torch.Tensor,
    prior_logits: torch.Tensor,
    alpha: float = 1.0,
    disable_heuristic: bool = False,
):
    device = prior_logits.device
    starts = _trace_tensor(traces, 'starts', torch.long, device)
    curr_nodes = _trace_tensor(traces, 'curr_nodes', torch.long, device)
    is_stochastic = _trace_tensor(traces, 'is_stochastic', torch.bool, device)
    pick_j = _trace_tensor(traces, 'pick_j', torch.long, device)
    valid_mask = _trace_tensor(traces, 'valid_mask', torch.long, device)

    n_ants = int(starts.numel() - 1)
    k_sparse = int(prior_logits.shape[1])
    trajectory_logp = torch.zeros(n_ants, dtype=prior_logits.dtype, device=device)
    trajectory_entropy = torch.zeros(n_ants, dtype=prior_logits.dtype, device=device)
    stochastic_decisions = torch.zeros(n_ants, dtype=torch.long, device=device)

    if n_ants == 0 or int(curr_nodes.numel()) == 0:
        return trajectory_logp, trajectory_entropy, stochastic_decisions

    base_logits = float(alpha) * torch.log(tau.to(device).clamp_min(EPS))
    if not disable_heuristic:
        base_logits = base_logits + torch.log(eta.to(device).clamp_min(EPS))
    logits_all = base_logits + prior_logits

    decision_counts = starts[1:] - starts[:-1]
    ant_ids = torch.repeat_interleave(torch.arange(n_ants, device=device), decision_counts)
    valid_choice = (pick_j >= 0) & (pick_j < k_sparse) & is_stochastic
    if not bool(valid_choice.any().item()):
        return trajectory_logp, trajectory_entropy, stochastic_decisions

    decision_idx = torch.nonzero(valid_choice, as_tuple=False).squeeze(1)
    chosen_j = pick_j.index_select(0, decision_idx)
    curr = curr_nodes.index_select(0, decision_idx).clamp(max=logits_all.shape[0] - 1)
    decision_ant_ids = ant_ids.index_select(0, decision_idx)
    mask_values = valid_mask.index_select(0, decision_idx)
    bit_offsets = torch.arange(k_sparse, dtype=torch.long, device=device)
    masks = ((mask_values.unsqueeze(1) >> bit_offsets.unsqueeze(0)) & 1).bool()
    chosen_is_valid = masks.gather(1, chosen_j.unsqueeze(1)).squeeze(1)
    if not bool(chosen_is_valid.any().item()):
        return trajectory_logp, trajectory_entropy, stochastic_decisions

    chosen_j = chosen_j[chosen_is_valid]
    curr = curr[chosen_is_valid]
    decision_ant_ids = decision_ant_ids[chosen_is_valid]
    masks = masks[chosen_is_valid]

    masked_logits = logits_all.index_select(0, curr).masked_fill(~masks, torch.finfo(prior_logits.dtype).min)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    probs = torch.softmax(masked_logits, dim=1)
    decision_logp = log_probs.gather(1, chosen_j.unsqueeze(1)).squeeze(1)
    decision_entropy = -(probs * log_probs).masked_fill(~masks, 0.0).sum(dim=1)
    trajectory_logp.index_add_(0, decision_ant_ids, decision_logp)
    trajectory_entropy.index_add_(0, decision_ant_ids, decision_entropy)
    stochastic_decisions.index_add_(0, decision_ant_ids, torch.ones_like(decision_ant_ids, dtype=torch.long))
    return trajectory_logp, trajectory_entropy, stochastic_decisions


def solver_kwargs(args: argparse.Namespace, n_ants: int, cand_list_size: int) -> dict[str, Any]:
    return {
        'n_ants': n_ants,
        'cand_list_size': cand_list_size,
        'backup_list_size': args.backup_list_size,
        'min_new_edges': args.min_new_edges,
        'decay': args.decay,
        'alpha': args.alpha,
        'p_best': args.p_best,
        'use_local_search': args.use_local_search,
        'disable_heuristic': args.disable_heuristic,
        'extend_ls': args.extend_ls,
        'smooth_mmas': args.smooth_mmas,
        'fixed_steps': args.fixed_steps,
        'nls': args.nls,
        'T_nls': args.T_nls,
        'device': DEVICE,
    }


def train_instance_dynaco(
    model,
    optimizer: torch.optim.Optimizer,
    data,
    args: argparse.Namespace,
    epoch: int,
    step_idx: int,
):
    from utils_ppo import build_solver, gen_pyg_data

    model.train()
    metrics = {
        'train_mean_cost': 0.0,
        'train_best_cost': 0.0,
        'train_loss': 0.0,
        'train_entropy': 0.0,
        'train_approx_kl': 0.0,
        'train_clip_frac': 0.0,
        'train_prior_std': 0.0,
    }
    instances_seen = 0
    prior_updates = 0

    for batch_offset, (_pyg_data, instance) in enumerate(data):
        solver = build_solver(instance, **solver_kwargs(args, args.ants, args.cand_list_size))
        solver.seed_rng(args.seed + epoch * 100_000 + step_idx * args.batch_size + batch_offset)

        rollout = []
        best_seen = math.inf
        last_mean = math.inf

        for _outer in range(args.train_H):
            pyg_data = gen_pyg_data(
                instance,
                cand_list_size=args.cand_list_size,
                backup_list_size=args.backup_list_size,
                n_ants=args.ants,
                min_new_edges=args.min_new_edges,
                device=DEVICE,
                solver=solver,
            ).to(DEVICE)
            with torch.no_grad():
                prior_old = model.reshape(pyg_data, model(pyg_data)).detach()
                metrics['train_prior_std'] += float(prior_old.std(unbiased=False).item())
                prior_updates += 1
            for _mini in range(args.train_mini_H):
                tau = solver.pheromone_sparse.detach().clone().to(DEVICE)
                eta = solver.h_sparse_torch.detach().clone().to(DEVICE)
                costs, routes, _, _, traces, costs_raw, _, _, _ = solver.sample(
                    require_prob=True,
                    prior=prior_old.cpu().numpy(),
                    parallel_traced=args.parallel_traced,
                )
                costs_t = torch.as_tensor(costs, dtype=torch.float32, device=DEVICE)
                costs_raw_t = torch.as_tensor(costs_raw if len(costs_raw) else costs, dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    old_logp, _, ndec = replay_logp_from_trace(
                        traces, tau, eta, prior_old, alpha=args.alpha, disable_heuristic=args.disable_heuristic
                    )
                    ndec_f = ndec.to(dtype=old_logp.dtype).clamp_min(1.0)
                    old_logp = old_logp / ndec_f
                rollout.append({
                    'pyg_data': pyg_data.detach().clone(),
                    'tau': tau,
                    'eta': eta,
                    'traces': traces,
                    'old_logp': old_logp.detach(),
                    'ndec': ndec.detach(),
                    'costs': costs_t.detach(),
                    'costs_raw': costs_raw_t.detach(),
                })
                best_idx = int(costs_t.argmin().item())
                best_cost = float(costs_t[best_idx].item())
                best_seen = min(best_seen, best_cost)
                last_mean = float(costs_t.mean().item())
                solver.update_pheromone(routes[best_idx], best_cost)

        for _ in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            entropies = []
            approx_kls = []
            clip_fracs = []
            for item in rollout:
                item_pyg_data = item['pyg_data']
                prior_new = model.reshape(item_pyg_data, model(item_pyg_data))
                new_logp, entropy, ndec = replay_logp_from_trace(
                    item['traces'], item['tau'], item['eta'], prior_new,
                    alpha=args.alpha, disable_heuristic=args.disable_heuristic,
                )
                ndec_f = ndec.to(dtype=new_logp.dtype).clamp_min(1.0)
                new_logp = new_logp / ndec_f
                entropy = entropy / ndec_f
                stochastic_mask = ndec > 0
                if not bool(stochastic_mask.any().item()):
                    continue
                new_logp = new_logp[stochastic_mask]
                old_logp = item['old_logp'][stochastic_mask]
                entropy = entropy[stochastic_mask]
                if args.nls:
                    combined_costs = args.nls_beta * item['costs'] + (1.0 - args.nls_beta) * item['costs_raw']
                else:
                    combined_costs = item['costs']
                combined_costs = combined_costs[stochastic_mask]
                advantage = (combined_costs.mean() - combined_costs).detach()
                if not args.no_adv_norm:
                    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
                ratio = torch.exp(new_logp - old_logp)
                log_ratio = new_logp - old_logp
                approx_kls.append((0.5 * log_ratio.pow(2)).mean())
                clip_fracs.append(((ratio > 1.0 + args.ppo_clip) | (ratio < 1.0 - args.ppo_clip)).float().mean())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * advantage
                losses.append(-torch.min(surr1, surr2).mean())
                entropies.append(entropy.mean())
            if not losses:
                continue
            loss = torch.stack(losses).mean() - args.entropy_coeff * torch.stack(entropies).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            metrics['train_loss'] += float(loss.detach().item())
            metrics['train_entropy'] += float(torch.stack(entropies).mean().detach().item())
            metrics['train_approx_kl'] += float(torch.stack(approx_kls).mean().detach().item())
            metrics['train_clip_frac'] += float(torch.stack(clip_fracs).mean().detach().item())

        metrics['train_mean_cost'] += last_mean
        metrics['train_best_cost'] += best_seen
        instances_seen += 1

    denom = max(instances_seen, 1)
    metrics['train_mean_cost'] /= denom
    metrics['train_best_cost'] /= denom
    metrics['train_loss'] /= max(denom * args.ppo_epochs, 1)
    metrics['train_entropy'] /= max(denom * args.ppo_epochs, 1)
    metrics['train_approx_kl'] /= max(denom * args.ppo_epochs, 1)
    metrics['train_clip_frac'] /= max(denom * args.ppo_epochs, 1)
    metrics['train_prior_std'] /= max(prior_updates, 1)
    return metrics


def _prior_for_model(model, pyg_data):
    if model is None:
        return None
    model.eval()
    with torch.no_grad():
        pyg_data = pyg_data.to(DEVICE)
        return model.reshape(pyg_data, model(pyg_data)).detach().cpu().numpy()


def infer_validation_instance(model, pyg_data, instance, n_ants: int, mode: str, instance_idx: int, **kwargs):
    import faco_test
    import faco_test_ib
    from faco import MFACO_CVRPTW
    from utils_ppo import gen_pyg_data

    prior = _prior_for_model(model, pyg_data)
    positions = torch.as_tensor(instance['coords'], dtype=torch.float32).cpu()
    demands = torch.as_tensor(instance['demand'], dtype=torch.float32).cpu()
    windows = torch.as_tensor(instance['windows'], dtype=torch.float32).cpu()
    common = dict(
        demands=demands,
        positions=positions,
        windows=windows,
        n_ants=n_ants,
        n_iter=kwargs['val_n_iter'],
        mini_H=kwargs['val_mini_H'],
        threads=kwargs.get('threads'),
        seed=kwargs.get('seed', 0) + instance_idx,
        cand_list_size=kwargs['cand_list_size'],
        backup_list_size=kwargs['backup_list_size'],
        min_new_edges=kwargs['min_new_edges'],
        decay=kwargs['decay'],
        alpha=kwargs['alpha'],
        p_best=kwargs['p_best'],
        use_local_search=kwargs['use_local_search'],
        disable_heuristic=kwargs['disable_heuristic'],
        extend_ls=kwargs['extend_ls'],
        smooth_mmas=kwargs['smooth_mmas'],
        fixed_steps=kwargs['fixed_steps'],
        nls=kwargs['nls'],
        T_nls=kwargs['T_nls'],
        prior=prior,
    )
    if mode == 'faco_test':
        solver = MFACO_CVRPTW(
            positions,
            demands,
            windows,
            capacity=1.0,
            n_ants=n_ants,
            cand_list_size=kwargs['cand_list_size'],
            backup_list_size=kwargs['backup_list_size'],
            min_new_edges=kwargs['min_new_edges'],
            decay=kwargs['decay'],
            alpha=kwargs['alpha'],
            p_best=kwargs['p_best'],
            use_local_search=kwargs['use_local_search'],
            disable_heuristic=kwargs['disable_heuristic'],
            extend_ls=kwargs['extend_ls'],
            smooth_mmas=kwargs['smooth_mmas'],
            device='cpu',
            fixed_steps=kwargs['fixed_steps'],
            nls=kwargs['nls'],
            T_nls=kwargs['T_nls'],
        )
        solver.seed_rng(kwargs.get('seed', 0) + instance_idx)
        results = torch.zeros(size=(kwargs['val_n_iter'],), dtype=torch.float32)
        diversity = torch.zeros(size=(kwargs['val_n_iter'],), dtype=torch.float32)
        best_so_far = float('inf')
        global_best_route = None
        elite_archive = []
        for t in range(kwargs['val_n_iter']):
            dynamic_pyg = gen_pyg_data(
                instance,
                cand_list_size=kwargs['cand_list_size'],
                backup_list_size=kwargs['backup_list_size'],
                n_ants=n_ants,
                min_new_edges=kwargs['min_new_edges'],
                device=DEVICE,
                solver=solver,
            )
            iter_prior = _prior_for_model(model, dynamic_pyg)
            routes = None
            for mini_t in range(kwargs['val_mini_H']):
                if mini_t == kwargs['val_mini_H'] - 1 and len(elite_archive) > 1:
                    costs_all = []
                    routes_all = []
                    for source in elite_archive[: min(kwargs.get('source_count', 2), len(elite_archive))]:
                        solver.set_source_route(source['route'], source['cost'])
                        costs, sampled_routes, *_ = solver.sample(prior=iter_prior)
                        costs_all.append(np.asarray(costs, dtype=np.float32))
                        routes_all.extend(np.asarray(route, dtype=np.int32) for route in sampled_routes)
                    costs_np = np.concatenate(costs_all)
                    routes = np.asarray(routes_all, dtype=object)
                else:
                    elite_source = faco_test.select_elite_source(elite_archive, t) if mini_t == kwargs['val_mini_H'] - 1 else None
                    if elite_source is not None:
                        solver.set_source_route(elite_source['route'], elite_source['cost'])
                    elif mini_t == kwargs['val_mini_H'] - 1 and global_best_route is not None:
                        solver.set_source_route(global_best_route, best_so_far)
                    costs, routes, *_ = solver.sample(prior=iter_prior)
                    costs_np = np.asarray(costs, dtype=np.float32)
                best_idx = int(np.argmin(costs_np))
                best_cost = float(costs_np[best_idx])
                best_route = np.asarray(routes[best_idx], dtype=np.int32)
                if best_cost < best_so_far:
                    best_so_far = best_cost
                    global_best_route = best_route.copy()
                pinned_elite = {'route': global_best_route, 'cost': best_so_far} if kwargs.get('pin_global_best_elite', False) and global_best_route is not None else None
                elite_archive = faco_test.update_elite_archive(
                    elite_archive,
                    routes,
                    costs_np,
                    elite_k=kwargs.get('elite_k', 8),
                    elite_min_diversity=kwargs.get('elite_min_diversity', 0.15),
                    elite_cost_tolerance=kwargs.get('elite_cost_tolerance', 1.05),
                    pinned_elite=pinned_elite,
                )
                solver.update_pheromone(best_route, best_cost)
            results[t] = best_so_far
            if t == kwargs['val_n_iter'] - 1:
                diversity[t] = faco_test.route_diversity(routes)
    elif mode == 'faco_test_ib':
        results, diversity, _ = faco_test_ib.infer_instance(**common)
    else:
        raise ValueError(f'unknown validation infer mode: {mode}')
    return [float(results[0]), float(results[0]), float(results[0]), float(results[0]), float(results[-1]), float(diversity[-1])]


def validate_model(
    model,
    val_list,
    n_ants: int,
    mode: str,
    epoch: int,
    **kwargs,
):
    modes = ['faco_test', 'faco_test_ib'] if mode == 'both' else [mode]
    metrics = {}
    for infer_mode in modes:
        stats = [
            infer_validation_instance(model, pyg_data, instance, n_ants, infer_mode, idx, **kwargs)
            for idx, (pyg_data, instance) in enumerate(tqdm(val_list, desc=f'Val-{infer_mode}', dynamic_ncols=True))
        ]
        avg = np.asarray(stats, dtype=np.float32).mean(0).tolist()
        prefix = infer_mode
        metrics[f'{prefix}_best_1'] = float(avg[3])
        metrics[f'{prefix}_best_T'] = float(avg[4])
        metrics[f'{prefix}_diversity_T'] = float(avg[5])
        print(f'epoch {epoch} {infer_mode}:', avg)
    return metrics


def selected_metric(metrics: dict[str, float], select_mode: str) -> float:
    return float(metrics[f'{select_mode}_best_T'])


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return (
        f'dynaco_cvrptw{args.nodes}_sd{args.seed}'
        f'_minnew{args.min_new_edges}_k{args.cand_list_size}'
        f'_H{args.train_H}_miniH{args.train_mini_H}'
    )


def write_report(report_path: Path, args: argparse.Namespace, history: list[dict[str, Any]], best_epoch: int, best_value: float):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
        branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    except Exception:
        commit = 'unknown'
        branch = 'unknown'
    val_keys = sorted(k for row in history for k in row if k.endswith('_best_T'))
    val_keys = list(dict.fromkeys(val_keys))
    with report_path.open('w', encoding='utf-8') as f:
        f.write('# neural_DyNACO PPO Experiment Report\n\n')
        f.write(f'- Branch: `{branch}`\n')
        f.write(f'- Commit: `{commit}`\n')
        f.write(f'- Best epoch: `{best_epoch}`\n')
        f.write(f'- Best selected metric: `{best_value:.6f}`\n')
        f.write(f'- Validation mode: `{args.val_infer_mode}`; selection mode: `{args.select_metric_mode}`\n\n')
        f.write('## Command\n\n')
        f.write('```powershell\nuv run .\\train_dynaco_ppo.py ' + ' '.join(os.sys.argv[1:]) + '\n```\n\n')
        f.write('## Per-Epoch Metrics\n\n')
        diag_keys = ['train_loss', 'train_entropy', 'train_approx_kl', 'train_clip_frac', 'train_prior_std']
        headers = ['epoch', 'train_best_cost', 'train_mean_cost'] + diag_keys + val_keys
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for row in history:
            values = []
            for key in headers:
                value = row.get(key, '')
                values.append(f'{value:.6f}' if isinstance(value, float) else str(value))
            f.write('| ' + ' | '.join(values) + ' |\n')


def train(args: argparse.Namespace):
    from net import Net
    from utils_ppo import generate_traindata, load_val_dataset

    global DEVICE, USE_WANDB
    DEVICE = args.device if torch.cuda.is_available() and str(args.device).startswith('cuda') else 'cpu'
    USE_WANDB = not args.disable_wandb and wandb is not None
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.k_sparse is None:
        args.k_sparse = min(args.nodes // 5, 32)
    if args.cand_list_size is None:
        args.cand_list_size = args.k_sparse
    args.cand_list_size = int(args.cand_list_size)
    args.run_name = resolve_run_name(args)

    print(f'DyNACO PPO training device: {DEVICE}')
    print(f'DyNACO PPO C++ backend threads: {configure_cpp_threads(args.threads)}')
    if USE_WANDB:
        wandb.init(project='neufaco-cvrptw', name=args.run_name)
        wandb.config.update(vars(args))

    model = Net(value_head=False).to(DEVICE)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    val_list = load_val_dataset(args.nodes, args.cand_list_size, DEVICE, val_size=args.val_size)
    save_dir = Path(args.output) / str(args.nodes) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    val_kwargs = vars(args).copy()
    history = []
    best_value = math.inf
    best_epoch = -1
    baseline_metrics = validate_model(None, val_list, args.val_ants, args.val_infer_mode, -1, **val_kwargs)
    history.append({'epoch': -1, 'train_best_cost': math.nan, 'train_mean_cost': math.nan, **baseline_metrics})
    initial_metrics = validate_model(model, val_list, args.val_ants, args.val_infer_mode, 0, **val_kwargs)
    history.append({'epoch': 0, 'train_best_cost': math.nan, 'train_mean_cost': math.nan, **initial_metrics})
    best_value = selected_metric(initial_metrics, args.select_metric_mode)
    best_epoch = 0
    torch.save(model.state_dict(), save_dir / 'best.pt')

    for epoch in range(1, args.epochs + 1):
        epoch_metrics = []
        for step in tqdm(range(args.steps), desc='Train', dynamic_ncols=True):
            data = generate_traindata(args.batch_size, args.nodes, args.cand_list_size, seed=(epoch * args.steps + step) * args.batch_size)
            metrics = train_instance_dynaco(model, optimizer, data, args, epoch, step)
            epoch_metrics.append(metrics)
        train_summary = {
            key: float(np.mean([m[key] for m in epoch_metrics]))
            for key in epoch_metrics[0]
        } if epoch_metrics else {'train_best_cost': math.nan, 'train_mean_cost': math.nan, 'train_loss': math.nan, 'train_entropy': math.nan}
        val_metrics = validate_model(model, val_list, args.val_ants, args.val_infer_mode, epoch, **val_kwargs)
        row = {'epoch': epoch, **train_summary, **val_metrics}
        history.append(row)
        current_value = selected_metric(val_metrics, args.select_metric_mode)
        if current_value < best_value:
            best_value = current_value
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir / 'best.pt')
        torch.save(model.state_dict(), save_dir / f'epoch-{epoch}.pt')
        if USE_WANDB:
            wandb.log(row, step=epoch)
        write_report(Path(args.report_path), args, history, best_epoch, best_value)
    write_report(Path(args.report_path), args, history, best_epoch, best_value)
    return history


def parse_args():
    parser = argparse.ArgumentParser(description='Train DyNACO-style PPO neural prior for CVRPTW FACO.')
    parser.add_argument('nodes', type=int)
    parser.add_argument('-k', '--k_sparse', type=int, default=None)
    parser.add_argument('--cand_list_size', type=int, default=None)
    parser.add_argument('--backup_list_size', type=int, default=64)
    parser.add_argument('--min_new_edges', type=int, default=8)
    parser.add_argument('--ants', type=int, default=32)
    parser.add_argument('--val_ants', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--ppo_clip', type=float, default=0.1)
    parser.add_argument('--entropy_coeff', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--no_adv_norm', action='store_true')
    parser.add_argument('--train_H', type=int, default=1)
    parser.add_argument('--train_mini_H', type=int, default=4)
    parser.add_argument('--val_size', type=int, default=20)
    parser.add_argument('--val_n_iter', type=int, default=10)
    parser.add_argument('--val_mini_H', type=int, default=10)
    parser.add_argument('--val_infer_mode', choices=['faco_test', 'faco_test_ib', 'both'], default='faco_test_ib')
    parser.add_argument('--select_metric_mode', choices=['faco_test', 'faco_test_ib'], default='faco_test_ib')
    parser.add_argument('--decay', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--p_best', type=float, default=0.05)
    parser.add_argument('--use_local_search', nargs='?', const=True, default=False, type=parse_bool)
    parser.add_argument('--disable_heuristic', action='store_true')
    parser.add_argument('--extend_ls', action='store_true')
    parser.add_argument('--smooth_mmas', action='store_true')
    parser.add_argument('--fixed_steps', type=int, default=0)
    parser.add_argument('--nls', action='store_true')
    parser.add_argument('--nls_beta', type=float, default=0.2)
    parser.add_argument('--T_nls', type=int, default=10)
    parser.add_argument('--elite_k', type=int, default=8)
    parser.add_argument('--elite_min_diversity', type=float, default=0.15)
    parser.add_argument('--elite_cost_tolerance', type=float, default=1.05)
    parser.add_argument('--source_count', type=int, default=2)
    parser.add_argument('--pin_global_best_elite', action='store_true')
    parser.add_argument('--parallel_traced', action='store_true')
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--output', type=str, default='../pretrained/cvrptw_dynaco_ppo')
    parser.add_argument('--report_path', type=str, default='experiments/neural_DyNACO_report.md')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--disable_wandb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
