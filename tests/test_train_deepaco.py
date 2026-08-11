import torch

from trainers.deepaco_trainer import deepaco_reinforce_loss


def test_deepaco_reinforce_loss_uses_mean_baseline_and_path_log_probs():
    costs = torch.tensor([10.0, 14.0, 20.0])
    log_probs = torch.tensor([
        [-0.1, -0.2, -0.3],
        [-0.4, -0.5, -0.6],
    ])

    loss = deepaco_reinforce_loss(costs, log_probs)

    advantages = costs - costs.mean()
    expected = torch.sum(advantages.detach() * log_probs.sum(dim=0)) / costs.numel()
    assert torch.allclose(loss, expected)


def test_deepaco_reinforce_loss_accepts_external_reward_costs():
    sampled_costs = torch.tensor([10.0, 14.0, 20.0])
    reward_costs = torch.tensor([9.0, 12.0, 18.0])
    log_probs = torch.tensor([
        [-0.1, -0.2, -0.3],
        [-0.4, -0.5, -0.6],
    ])

    loss = deepaco_reinforce_loss(sampled_costs, log_probs, reward_costs=reward_costs)

    advantages = reward_costs - reward_costs.mean()
    expected = torch.sum(advantages.detach() * log_probs.sum(dim=0)) / sampled_costs.numel()
    assert torch.allclose(loss, expected)
