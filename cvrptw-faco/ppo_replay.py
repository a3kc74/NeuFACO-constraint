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
    k_sparse = prior_logits.shape[1]

    for ant in range(n_ants):
        start = int(starts[ant].item())
        end = int(starts[ant + 1].item())
        for decision in range(start, end):
            if not bool(is_stochastic[decision].item()):
                continue
            chosen_j = int(pick_j[decision].item())
            if chosen_j < 0:
                continue
            curr = int(curr_nodes[decision].item())
            mask_value = int(valid_mask[decision].item())
            mask = torch.tensor(
                [((mask_value >> j) & 1) == 1 for j in range(k_sparse)],
                dtype=torch.bool,
                device=device,
            )
            if chosen_j >= k_sparse or not bool(mask[chosen_j].item()):
                continue
            logits = invtemp * prior_logits[curr]
            masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            log_probs = torch.log_softmax(masked_logits, dim=0)
            probs = torch.softmax(masked_logits, dim=0)
            trajectory_logp[ant] = trajectory_logp[ant] + log_probs[chosen_j]
            trajectory_entropy[ant] = trajectory_entropy[ant] - (probs[mask] * log_probs[mask]).sum()
    return trajectory_logp, trajectory_entropy
