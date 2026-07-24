from __future__ import annotations

import numpy as np
import torch


def _trace_tensor(traces, name, dtype, device):
    value = getattr(traces, name)
    if name == 'valid_mask':
        value = np.asarray(value, dtype=np.uint64).astype(np.int64)
    return torch.as_tensor(value, dtype=dtype, device=device)


def replay_logp_batch(prior_logits, traces, invtemp=1.0):
    device = prior_logits.device
    starts = _trace_tensor(traces, 'starts', torch.long, device)
    curr_nodes = _trace_tensor(traces, 'curr_nodes', torch.long, device)
    is_stochastic = _trace_tensor(traces, 'is_stochastic', torch.bool, device)
    pick_j = _trace_tensor(traces, 'pick_j', torch.long, device)
    valid_mask = _trace_tensor(traces, 'valid_mask', torch.long, device)

    n_ants = int(starts.numel() - 1)
    trajectory_logp = torch.zeros(n_ants, dtype=prior_logits.dtype, device=device)
    trajectory_entropy = torch.zeros(n_ants, dtype=prior_logits.dtype, device=device)
    k_sparse = int(prior_logits.shape[1])

    if n_ants == 0 or int(curr_nodes.numel()) == 0:
        return trajectory_logp, trajectory_entropy

    decision_counts = starts[1:] - starts[:-1]
    ant_ids = torch.repeat_interleave(torch.arange(n_ants, device=device), decision_counts)
    valid_choice = (pick_j >= 0) & (pick_j < k_sparse) & is_stochastic
    if not bool(valid_choice.any().item()):
        return trajectory_logp, trajectory_entropy

    decision_idx = torch.nonzero(valid_choice, as_tuple=False).squeeze(1)
    chosen_j = pick_j.index_select(0, decision_idx)
    curr = curr_nodes.index_select(0, decision_idx).clamp(max=prior_logits.shape[0] - 1)
    decision_ant_ids = ant_ids.index_select(0, decision_idx)
    mask_values = valid_mask.index_select(0, decision_idx)
    bit_offsets = torch.arange(k_sparse, dtype=torch.long, device=device)
    masks = ((mask_values.unsqueeze(1) >> bit_offsets.unsqueeze(0)) & 1).bool()
    chosen_is_valid = masks.gather(1, chosen_j.unsqueeze(1)).squeeze(1)
    if not bool(chosen_is_valid.any().item()):
        return trajectory_logp, trajectory_entropy

    chosen_j = chosen_j[chosen_is_valid]
    curr = curr[chosen_is_valid]
    decision_ant_ids = decision_ant_ids[chosen_is_valid]
    masks = masks[chosen_is_valid]

    masked_logits = (invtemp * prior_logits.index_select(0, curr)).masked_fill(~masks, torch.finfo(prior_logits.dtype).min)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    probs = torch.softmax(masked_logits, dim=1)
    decision_logp = log_probs.gather(1, chosen_j.unsqueeze(1)).squeeze(1)
    decision_entropy = -(probs * log_probs).masked_fill(~masks, 0.0).sum(dim=1)
    trajectory_logp.index_add_(0, decision_ant_ids, decision_logp)
    trajectory_entropy.index_add_(0, decision_ant_ids, decision_entropy)
    return trajectory_logp, trajectory_entropy
