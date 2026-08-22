import torch

from trainers import deepaco_trainer
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


def test_deepaco_reinforce_loss_accepts_replayed_path_log_probs():
    costs = torch.tensor([10.0, 14.0, 20.0])
    replay_logp = torch.tensor([-0.5, -0.7, -0.9])

    loss = deepaco_reinforce_loss(costs, replay_logp)

    advantages = costs - costs.mean()
    expected = torch.sum(advantages.detach() * replay_logp) / costs.numel()
    assert torch.allclose(loss, expected)


def test_train_instance_samples_with_faco_and_replay(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, pyg_data):
            return self.weight * torch.ones(3, 3)

        def reshape(self, pyg_data, values):
            return values

    class FakeSolver:
        def __init__(self, *args, **kwargs):
            self.nn_list = torch.tensor([[0, 1], [0, 2], [1, 2]]).numpy()
            self.pheromone_sparse = torch.ones(3, 2)
            self.h_sparse_torch = torch.ones(3, 2)

        def sample(self, require_prob=False, prior=None, parallel_traced=False):
            calls["require_prob"] = require_prob
            calls["prior_shape"] = prior.shape
            calls["parallel_traced"] = parallel_traced
            return [1.0, 2.0], [], None, None, {"starts": [0, 0, 0], "curr_nodes": [], "is_stochastic": [], "pick_j": [], "valid_mask": []}, [], [], 0, None

    def fake_replay(traces, tau, eta, prior_logits, alpha=1.0, disable_heuristic=False, return_decision_counts=False):
        calls["replay_prior_requires_grad"] = prior_logits.requires_grad
        return prior_logits.sum() * torch.tensor([0.1, 0.2]), None, torch.ones(2, dtype=torch.long)

    calls = {}
    monkeypatch.setattr(deepaco_trainer, "MFACO_CVRPTW", FakeSolver)
    monkeypatch.setattr(deepaco_trainer, "replay_logp_from_trace", fake_replay)
    monkeypatch.setattr(deepaco_trainer, "DEVICE", "cpu")
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data = [(None, torch.zeros(3), torch.zeros(3, 3), torch.zeros(3, 2), torch.zeros(3, 2))]

    deepaco_trainer.train_instance(
        model,
        optimizer,
        data,
        n_ants=2,
        faco_params={"parallel_traced": True, "use_local_search": False},
        k_sparse=2,
    )

    assert calls["require_prob"] is True
    assert calls["prior_shape"] == (3, 2)
    assert calls["parallel_traced"] is True
    assert calls["replay_prior_requires_grad"] is True


def test_train_instance_can_force_granular_gnn_graph(monkeypatch):
    class GranularPyg:
        pass

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, pyg_data):
            calls["pyg_data"] = pyg_data
            return self.weight * torch.ones(3, 3)

        def reshape(self, _pyg_data, values):
            return values

    class FakeSolver:
        def __init__(self, *args, **kwargs):
            self.nn_list = torch.tensor([[0, 1], [0, 2], [1, 2]]).numpy()
            self.pheromone_sparse = torch.ones(3, 2)
            self.h_sparse_torch = torch.ones(3, 2)

        def sample(self, **_kwargs):
            return [1.0, 2.0], [], None, None, {"starts": [0, 0, 0], "curr_nodes": [], "is_stochastic": [], "pick_j": [], "valid_mask": []}, [], [], 0, None

    def fake_granular_pyg(_demands, _distances, _windows, solver, _device):
        calls["granular_solver"] = solver
        return granular_pyg

    def fake_replay(_traces, _tau, _eta, prior_logits, **_kwargs):
        return prior_logits.sum() * torch.tensor([0.1, 0.2]), None, torch.ones(2, dtype=torch.long)

    calls = {}
    granular_pyg = GranularPyg()
    monkeypatch.setattr(deepaco_trainer, "MFACO_CVRPTW", FakeSolver)
    monkeypatch.setattr(deepaco_trainer.faco_test, "gen_granular_pyg_data", fake_granular_pyg)
    monkeypatch.setattr(deepaco_trainer, "replay_logp_from_trace", fake_replay)
    monkeypatch.setattr(deepaco_trainer, "DEVICE", "cpu")

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data = [(object(), torch.zeros(3), torch.ones(3, 3), torch.zeros(3, 2), torch.zeros(3, 2))]

    deepaco_trainer.train_instance(
        model,
        optimizer,
        data,
        n_ants=2,
        faco_params={"deepaco_graph_mode": "granular", "use_local_search": False},
        k_sparse=2,
    )

    assert calls["pyg_data"] is granular_pyg
    assert isinstance(calls["granular_solver"], FakeSolver)


def test_validation_uses_faco_infer_final_cost(monkeypatch):
    def fake_infer_instance(*args, **kwargs):
        calls["deepaco_model"] = kwargs["deepaco_model"]
        calls["n_iter"] = kwargs["n_iter"]
        calls["deepaco_graph_mode"] = kwargs["deepaco_graph_mode"]
        return torch.tensor([3.0, 2.0]), torch.tensor([0.1, 0.2]), 0.0, [1, 2], {}

    calls = {}
    monkeypatch.setattr(deepaco_trainer.faco_test, "infer_instance", fake_infer_instance)
    monkeypatch.setattr(deepaco_trainer, "DEVICE", "cpu")
    val_list = [(None, torch.zeros(3), torch.zeros(3, 3), torch.zeros(3, 2), torch.zeros(3, 2))]
    model = torch.nn.Linear(1, 1)

    result = deepaco_trainer.validation(
        val_list,
        n_ants=2,
        net=model,
        epoch=0,
        steps_per_epoch=1,
        faco_params={"val_n_iter": 2, "use_local_search": False, "deepaco_graph_mode": "granular"},
        k_sparse=2,
    )

    assert result == 2.0
    assert calls["deepaco_model"] is model
    assert calls["n_iter"] == 2
    assert calls["deepaco_graph_mode"] == "granular"
