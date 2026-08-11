from __future__ import annotations

import argparse
import dis
import inspect
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
    from solvers.faco import set_faco_cpp_threads

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


class ReplayLogpResult:
    def __init__(self, logp, entropy, stochastic_decisions):
        self.logp = logp
        self.entropy = entropy
        self.stochastic_decisions = stochastic_decisions

    def __iter__(self):
        expected = 3
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if caller is not None:
            for instruction in dis.get_instructions(caller.f_code):
                if instruction.offset == caller.f_lasti and instruction.opname == 'UNPACK_SEQUENCE':
                    expected = int(instruction.arg or 3)
                    break
        yield self.logp
        yield self.entropy
        if expected != 2:
            yield self.stochastic_decisions


def _format_replay_result(logp, entropy, stochastic_decisions, return_decision_counts: bool):
    if return_decision_counts:
        return logp, entropy, stochastic_decisions
    return ReplayLogpResult(logp, entropy, stochastic_decisions)


def replay_logp_from_trace(
    traces,
    tau: torch.Tensor,
    eta: torch.Tensor,
    prior_logits: torch.Tensor,
    alpha: float = 1.0,
    disable_heuristic: bool = False,
    return_decision_counts: bool = False,
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
        return _format_replay_result(trajectory_logp, trajectory_entropy, stochastic_decisions, return_decision_counts)

    base_logits = float(alpha) * torch.log(tau.to(device).clamp_min(EPS))
    if not disable_heuristic:
        base_logits = base_logits + torch.log(eta.to(device).clamp_min(EPS))
    logits_all = base_logits + prior_logits

    decision_counts = starts[1:] - starts[:-1]
    ant_ids = torch.repeat_interleave(torch.arange(n_ants, device=device), decision_counts)
    valid_choice = (pick_j >= 0) & (pick_j < k_sparse) & is_stochastic
    if not bool(valid_choice.any().item()):
        return _format_replay_result(trajectory_logp, trajectory_entropy, stochastic_decisions, return_decision_counts)

    decision_idx = torch.nonzero(valid_choice, as_tuple=False).squeeze(1)
    chosen_j = pick_j.index_select(0, decision_idx)
    curr = curr_nodes.index_select(0, decision_idx).clamp(max=logits_all.shape[0] - 1)
    decision_ant_ids = ant_ids.index_select(0, decision_idx)
    mask_values = valid_mask.index_select(0, decision_idx)
    bit_offsets = torch.arange(k_sparse, dtype=torch.long, device=device)
    masks = ((mask_values.unsqueeze(1) >> bit_offsets.unsqueeze(0)) & 1).bool()
    chosen_is_valid = masks.gather(1, chosen_j.unsqueeze(1)).squeeze(1)
    if not bool(chosen_is_valid.any().item()):
        return _format_replay_result(trajectory_logp, trajectory_entropy, stochastic_decisions, return_decision_counts)

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
    return _format_replay_result(trajectory_logp, trajectory_entropy, stochastic_decisions, return_decision_counts)


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


def _combined_costs(item: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    if args.nls:
        return args.nls_beta * item['costs'] + (1.0 - args.nls_beta) * item['costs_raw']
    return item['costs']


def _annotate_advantages(rollout_groups: list[dict[str, Any]], args: argparse.Namespace) -> float:
    raw_advantages = []
    for group in rollout_groups:
        if args.advantage_mode == 'macro_relative':
            relative_advantages = []
            for item in group['items']:
                costs = _combined_costs(item, args)
                mean_cost = costs.mean()
                relative_advantages.append(((mean_cost - costs) / (mean_cost.abs() + 1e-6)).detach())
            if relative_advantages:
                outer_advantage = torch.cat(relative_advantages)
                raw_advantages.append(outer_advantage)
                if not args.no_adv_norm:
                    std = outer_advantage.std(unbiased=False)
                    if float(std.item()) > 1e-4:
                        outer_advantage = (outer_advantage - outer_advantage.mean()) / (std + 1e-6)
                    else:
                        outer_advantage = torch.zeros_like(outer_advantage)
                if args.adv_clip > 0:
                    outer_advantage = outer_advantage.clamp(-args.adv_clip, args.adv_clip)
                offset = 0
                for item in group['items']:
                    width = int(item['costs'].numel())
                    item['advantage'] = outer_advantage[offset:offset + width]
                    offset += width
        else:
            for item in group['items']:
                costs = _combined_costs(item, args)
                advantage = (costs.mean() - costs).detach()
                raw_advantages.append(advantage)
                if not args.no_adv_norm:
                    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
                if args.adv_clip > 0:
                    advantage = advantage.clamp(-args.adv_clip, args.adv_clip)
                item['advantage'] = advantage
    if not raw_advantages:
        return 0.0
    return float(torch.cat(raw_advantages).std(unbiased=False).item())


def _mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.tensor(0.0, device=DEVICE)
    return torch.stack(values).mean()


def update_ema_state(ema_state: dict[str, torch.Tensor] | None, model, decay: float):
    if ema_state is None or decay <= 0:
        return None
    with torch.no_grad():
        model_state = model.state_dict()
        for key, value in model_state.items():
            if torch.is_floating_point(value):
                ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_state[key].copy_(value)
    return ema_state


def evaluate_with_optional_ema(model, ema_state: dict[str, torch.Tensor] | None, fn):
    was_training = model.training
    if ema_state is None:
        model.eval()
        try:
            return fn(model)
        finally:
            model.train(was_training)
    live_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(ema_state, strict=True)
    model.eval()
    try:
        return fn(model)
    finally:
        model.load_state_dict(live_state, strict=True)
        model.train(was_training)


def train_instance_dynaco(
    model,
    optimizer: torch.optim.Optimizer,
    data,
    args: argparse.Namespace,
    epoch: int,
    step_idx: int,
):
    from envs.cvrptw_env import build_solver, gen_pyg_data

    model.train()
    metrics = {
        'train_mean_cost': 0.0,
        'train_best_cost': 0.0,
        'train_loss': 0.0,
        'train_entropy': 0.0,
        'train_approx_kl': 0.0,
        'train_clip_frac': 0.0,
        'train_prior_std': 0.0,
        'train_prior_mean': 0.0,
        'train_prior_max': 0.0,
        'train_grad_norm': 0.0,
        'train_raw_adv_std': 0.0,
        'train_stochastic_decisions': 0.0,
        'train_no_stochastic_frac': 0.0,
        'train_tau_cv': 0.0,
        'train_ppo_epochs_used': 0.0,
    }
    instances_seen = 0
    prior_updates = 0
    replay_items = 0
    ppo_updates = 0
    rollout_groups = []

    for batch_offset, (_pyg_data, instance) in enumerate(data):
        solver = build_solver(instance, **solver_kwargs(args, args.ants, args.cand_list_size))
        solver.seed_rng(args.seed + epoch * 100_000 + step_idx * args.batch_size + batch_offset)

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
                edge_feature_mode=args.edge_feature_mode,
            ).to(DEVICE)
            with torch.no_grad():
                prior_old = model.reshape(pyg_data, model(pyg_data)).detach()
                metrics['train_prior_std'] += float(prior_old.std(unbiased=False).item())
                metrics['train_prior_mean'] += float(prior_old.mean().item())
                metrics['train_prior_max'] += float(prior_old.max().item())
                prior_updates += 1
            rollout_group = {
                'pyg_data': pyg_data.detach().clone(),
                'items': [],
            }
            for _mini in range(args.train_mini_H):
                tau = solver.pheromone_sparse.detach().clone().to(DEVICE)
                eta = solver.h_sparse_torch.detach().clone().to(DEVICE)
                tau_mean = tau.mean().clamp_min(EPS)
                metrics['train_tau_cv'] += float((tau.std(unbiased=False) / tau_mean).item())
                costs, routes, _, _, traces, costs_raw, _, _, _ = solver.sample(
                    require_prob=True,
                    prior=prior_old.cpu().numpy(),
                    parallel_traced=args.parallel_traced,
                )
                costs_t = torch.as_tensor(costs, dtype=torch.float32, device=DEVICE)
                costs_raw_t = torch.as_tensor(costs_raw if len(costs_raw) else costs, dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    old_logp, _, ndec = replay_logp_from_trace(
                        traces, tau, eta, prior_old, alpha=args.alpha,
                        disable_heuristic=args.disable_heuristic,
                        return_decision_counts=True,
                    )
                    ndec_f = ndec.to(dtype=old_logp.dtype).clamp_min(1.0)
                    old_logp = old_logp / ndec_f
                metrics['train_stochastic_decisions'] += float(ndec.float().mean().item())
                metrics['train_no_stochastic_frac'] += float((ndec == 0).float().mean().item())
                replay_items += 1
                rollout_group['items'].append({
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
            rollout_groups.append(rollout_group)

        metrics['train_mean_cost'] += last_mean
        metrics['train_best_cost'] += best_seen
        instances_seen += 1

    metrics['train_raw_adv_std'] = _annotate_advantages(rollout_groups, args)
    for _ in range(args.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        entropies = []
        approx_kls = []
        clip_fracs = []
        for group in rollout_groups:
            group_pyg_data = group['pyg_data']
            prior_new = model.reshape(group_pyg_data, model(group_pyg_data))
            for item in group['items']:
                new_logp, entropy, ndec = replay_logp_from_trace(
                    item['traces'], item['tau'], item['eta'], prior_new,
                    alpha=args.alpha, disable_heuristic=args.disable_heuristic,
                    return_decision_counts=True,
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
                advantage = item['advantage'][stochastic_mask]
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        ppo_updates += 1
        mean_entropy = _mean_or_zero(entropies)
        mean_kl = _mean_or_zero(approx_kls)
        mean_clip_frac = _mean_or_zero(clip_fracs)
        metrics['train_loss'] += float(loss.detach().item())
        metrics['train_entropy'] += float(mean_entropy.detach().item())
        metrics['train_approx_kl'] += float(mean_kl.detach().item())
        metrics['train_clip_frac'] += float(mean_clip_frac.detach().item())
        metrics['train_grad_norm'] += float(torch.as_tensor(grad_norm).detach().item())
        if args.target_kl > 0 and float(mean_kl.detach().item()) > 1.5 * args.target_kl:
            break

    denom = max(instances_seen, 1)
    metrics['train_mean_cost'] /= denom
    metrics['train_best_cost'] /= denom
    metrics['train_loss'] /= max(ppo_updates, 1)
    metrics['train_entropy'] /= max(ppo_updates, 1)
    metrics['train_approx_kl'] /= max(ppo_updates, 1)
    metrics['train_clip_frac'] /= max(ppo_updates, 1)
    metrics['train_grad_norm'] /= max(ppo_updates, 1)
    metrics['train_prior_std'] /= max(prior_updates, 1)
    metrics['train_prior_mean'] /= max(prior_updates, 1)
    metrics['train_prior_max'] /= max(prior_updates, 1)
    metrics['train_stochastic_decisions'] /= max(replay_items, 1)
    metrics['train_no_stochastic_frac'] /= max(replay_items, 1)
    metrics['train_tau_cv'] /= max(replay_items, 1)
    metrics['train_ppo_epochs_used'] = float(ppo_updates)
    return metrics


def _prior_for_model(model, pyg_data):
    if model is None:
        return None
    model.eval()
    with torch.no_grad():
        pyg_data = pyg_data.to(DEVICE)
        return model.reshape(pyg_data, model(pyg_data)).detach().cpu().numpy()


def infer_validation_instance(model, pyg_data, instance, n_ants: int, mode: str, instance_idx: int, **kwargs):
    from evaluation import faco_test, faco_test_ib
    from solvers.faco import MFACO_CVRPTW
    from envs.cvrptw_env import gen_pyg_data

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
                edge_feature_mode=kwargs.get('edge_feature_mode', 'full'),
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


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_value: float,
    best_step: int = 0,
    stopped_reason: str = '',
):
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
        f.write(f'- Best step: `{best_step}`\n')
        f.write(f'- Best selected metric: `{best_value:.6f}`\n')
        f.write(f'- Validation mode: `{args.val_infer_mode}`; selection mode: `{args.select_metric_mode}`\n')
        f.write(f'- Validation frequency: every `{args.val_every_epochs}` epoch(s), plus final epoch\n\n')
        if stopped_reason:
            f.write(f'- Stopped early: `{stopped_reason}`\n\n')
        f.write('## Command\n\n')
        f.write('```powershell\nuv run .\\train_dynaco_ppo.py ' + ' '.join(os.sys.argv[1:]) + '\n```\n\n')
        f.write('## Per-Step Metrics\n\n')
        diag_keys = [
            'train_loss', 'train_entropy', 'train_approx_kl', 'train_clip_frac',
            'train_prior_mean', 'train_prior_std', 'train_prior_max', 'train_grad_norm',
            'train_raw_adv_std', 'train_stochastic_decisions', 'train_no_stochastic_frac',
            'train_tau_cv', 'train_ppo_epochs_used', 'lr',
        ]
        headers = ['step', 'epoch', 'train_best_cost', 'train_mean_cost'] + diag_keys + val_keys
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for row in history:
            values = []
            for key in headers:
                value = row.get(key, '')
                values.append(f'{value:.6f}' if isinstance(value, float) else str(value))
            f.write('| ' + ' | '.join(values) + ' |\n')


def train(args: argparse.Namespace):
    from models.faco_net import Net
    from envs.cvrptw_env import generate_traindata, load_val_dataset

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
    args.val_every_epochs = max(1, int(args.val_every_epochs))
    args.run_name = resolve_run_name(args)

    print(f'DyNACO PPO training device: {DEVICE}')
    print(f'DyNACO PPO C++ backend threads: {configure_cpp_threads(args.threads)}')
    if USE_WANDB:
        wandb.init(project='neufaco-cvrptw', name=args.run_name)
        wandb.config.update(vars(args))

    model = Net(value_head=False, norm_type=args.norm_type).to(DEVICE)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1)) if args.lr_scheduler == 'cosine' else None
    ema_state = None
    if args.ema_decay > 0:
        ema_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    val_list = load_val_dataset(args.nodes, args.cand_list_size, DEVICE, val_size=args.val_size)
    save_dir = Path(args.output) / str(args.nodes) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    val_kwargs = vars(args).copy()
    history = []
    best_value = math.inf
    best_epoch = -1
    best_step = 0
    best_for_patience = math.inf
    validation_checks_without_improvement = 0
    stopped_reason = ''
    baseline_metrics = validate_model(None, val_list, args.val_ants, args.val_infer_mode, -1, **val_kwargs)
    history.append({'step': 0, 'epoch': -1, 'train_best_cost': math.nan, 'train_mean_cost': math.nan, **baseline_metrics})
    initial_metrics = evaluate_with_optional_ema(
        model,
        ema_state,
        lambda eval_model: validate_model(eval_model, val_list, args.val_ants, args.val_infer_mode, 0, **val_kwargs),
    )
    history.append({'step': 0, 'epoch': 0, 'train_best_cost': math.nan, 'train_mean_cost': math.nan, **initial_metrics})
    best_value = selected_metric(initial_metrics, args.select_metric_mode)
    best_epoch = 0
    best_step = 0
    best_for_patience = best_value
    torch.save(ema_state if ema_state is not None else model.state_dict(), save_dir / 'best.pt')

    for epoch in range(1, args.epochs + 1):
        epoch_metrics = []
        for step in tqdm(range(args.steps), desc='Train', dynamic_ncols=True):
            data = generate_traindata(args.batch_size, args.nodes, args.cand_list_size, seed=(epoch * args.steps + step) * args.batch_size)
            metrics = train_instance_dynaco(model, optimizer, data, args, epoch, step)
            update_ema_state(ema_state, model, args.ema_decay)
            epoch_metrics.append(metrics)
        train_summary = {
            key: float(np.mean([m[key] for m in epoch_metrics]))
            for key in epoch_metrics[0]
        } if epoch_metrics else {'train_best_cost': math.nan, 'train_mean_cost': math.nan, 'train_loss': math.nan, 'train_entropy': math.nan}
        if scheduler is not None:
            scheduler.step()
        train_summary['lr'] = float(optimizer.param_groups[0]['lr'])
        global_step = epoch * args.steps
        should_validate = (epoch % args.val_every_epochs == 0) or (epoch == args.epochs)
        val_metrics = {}
        if should_validate:
            val_metrics = evaluate_with_optional_ema(
                model,
                ema_state,
                lambda eval_model: validate_model(eval_model, val_list, args.val_ants, args.val_infer_mode, epoch, **val_kwargs),
            )
        row = {'step': global_step, 'epoch': epoch, **train_summary, **val_metrics}
        history.append(row)
        if should_validate:
            current_value = selected_metric(val_metrics, args.select_metric_mode)
            if current_value < best_value:
                best_value = current_value
                best_epoch = epoch
                best_step = global_step
                torch.save(ema_state if ema_state is not None else model.state_dict(), save_dir / 'best.pt')
            if current_value < best_for_patience - args.early_stop_min_delta:
                best_for_patience = current_value
                validation_checks_without_improvement = 0
            else:
                validation_checks_without_improvement += 1
        torch.save(ema_state if ema_state is not None else model.state_dict(), save_dir / f'epoch-{epoch}.pt')
        if USE_WANDB:
            wandb.log(row, step=global_step)
        if should_validate and args.early_stop_patience > 0 and validation_checks_without_improvement >= args.early_stop_patience:
            stopped_reason = (
                f'no selected-metric improvement > {args.early_stop_min_delta:g} '
                f'for {args.early_stop_patience} validation checks'
            )
            write_report(Path(args.report_path), args, history, best_epoch, best_value, best_step, stopped_reason)
            break
        write_report(Path(args.report_path), args, history, best_epoch, best_value, best_step, stopped_reason)
    write_report(Path(args.report_path), args, history, best_epoch, best_value, best_step, stopped_reason)
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
    parser.add_argument('--advantage_mode', choices=['per_step', 'macro_relative'], default='per_step')
    parser.add_argument('--adv_clip', type=float, default=0.0)
    parser.add_argument('--target_kl', type=float, default=0.0)
    parser.add_argument('--lr_scheduler', choices=['none', 'cosine'], default='none')
    parser.add_argument('--ema_decay', type=float, default=0.0)
    parser.add_argument('--norm_type', choices=['batch', 'layer'], default='batch')
    parser.add_argument('--edge_feature_mode', choices=['full', 'static'], default='full')
    parser.add_argument('--train_H', type=int, default=1)
    parser.add_argument('--train_mini_H', type=int, default=4)
    parser.add_argument('--val_size', type=int, default=20)
    parser.add_argument('--val_every_epochs', type=int, default=1)
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
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0)
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
