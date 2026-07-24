import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_dynaco_ppo as dynaco_ppo
from train_dynaco_ppo import replay_logp_from_trace, resolve_run_name
import utils_ppo
from utils_ppo import build_solver, gen_pyg_data, generate_cvrptw_instance, generate_traindata


def test_generate_traindata_treats_nodes_as_customer_count():
    pyg_data, instance = generate_traindata(1, 50, 10, seed=123)[0]

    assert tuple(instance["coords"].shape) == (51, 2)
    assert pyg_data.n_nodes == 51


def test_replay_logp_reports_stochastic_decision_counts():
    traces = SimpleNamespace(
        starts=[0, 2],
        curr_nodes=[0, 1],
        is_stochastic=[True, True],
        pick_j=[0, 1],
        valid_mask=[3, 3],
    )
    tau = torch.ones((2, 2), dtype=torch.float32)
    eta = torch.ones((2, 2), dtype=torch.float32)
    prior = torch.zeros((2, 2), dtype=torch.float32)

    logp, _entropy, ndec = replay_logp_from_trace(traces, tau, eta, prior)

    assert torch.equal(ndec, torch.tensor([2]))
    assert torch.allclose(logp / ndec.clamp_min(1), torch.tensor([-0.69314718]))


def test_resolve_run_name_adds_dynaco_training_parameters_when_unset():
    args = SimpleNamespace(
        run_name='',
        nodes=100,
        seed=0,
        min_new_edges=8,
        cand_list_size=20,
        train_H=1,
        train_mini_H=4,
    )

    assert resolve_run_name(args) == 'dynaco_cvrptw100_sd0_minnew8_k20_H1_miniH4'


def test_resolve_run_name_keeps_explicit_name():
    args = SimpleNamespace(
        run_name='custom',
        nodes=100,
        seed=0,
        min_new_edges=8,
        cand_list_size=20,
        train_H=1,
        train_mini_H=4,
    )

    assert resolve_run_name(args) == 'custom'


def test_gen_pyg_data_reflects_solver_dynamic_state():
    instance = generate_cvrptw_instance(5, seed=321)
    solver = build_solver(instance, n_ants=4, cand_list_size=3, backup_list_size=4, min_new_edges=1)

    before = gen_pyg_data(instance, cand_list_size=3, solver=solver).edge_attr.clone()
    costs, routes, *_ = solver.sample()
    best_idx = int(costs.argmin() if hasattr(costs, "argmin") else min(range(len(costs)), key=lambda i: costs[i]))
    solver.update_pheromone(routes[best_idx], float(costs[best_idx]))
    after = gen_pyg_data(instance, cand_list_size=3, solver=solver).edge_attr

    assert not before.equal(after)


def test_train_instance_replays_each_rollout_with_its_own_dynamic_state(monkeypatch):
    class FakePygData:
        def __init__(self, state_marker):
            self.state_marker = float(state_marker)

        def to(self, _device):
            return self

        def clone(self):
            return FakePygData(self.state_marker)

        def detach(self):
            return self

    class FakeSolver:
        def __init__(self):
            self.state_marker = 1
            self.h_sparse_torch = torch.ones((2, 2), dtype=torch.float32)

        @property
        def pheromone_sparse(self):
            return torch.full((2, 2), float(self.state_marker), dtype=torch.float32)

        def seed_rng(self, _seed):
            return None

        def sample(self, **_kwargs):
            costs = torch.tensor([2.0, 1.0])
            routes = [[0, 1, 0], [0, 2, 0]]
            return costs, routes, None, None, object(), costs, None, None, None

        def update_pheromone(self, _route, _cost):
            self.state_marker += 1

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, pyg_data):
            return self.scale * torch.full((4,), pyg_data.state_marker, dtype=torch.float32)

        @staticmethod
        def reshape(_pyg_data, vector):
            return vector.reshape(2, 2)

    replayed_pairs = []

    def fake_build_solver(_instance, **_kwargs):
        return FakeSolver()

    def fake_gen_pyg_data(_instance, solver, **_kwargs):
        return FakePygData(solver.state_marker)

    def fake_replay(_traces, tau, _eta, prior_logits, **_kwargs):
        if torch.is_grad_enabled():
            replayed_pairs.append((float(tau[0, 0].item()), float(prior_logits[0, 0].detach().item())))
        logp = prior_logits.mean(dim=1)
        entropy = prior_logits.new_zeros(2)
        ndec = torch.ones(2, dtype=torch.long)
        return logp, entropy, ndec

    monkeypatch.setattr(utils_ppo, 'build_solver', fake_build_solver)
    monkeypatch.setattr(utils_ppo, 'gen_pyg_data', fake_gen_pyg_data)
    monkeypatch.setattr(dynaco_ppo, 'replay_logp_from_trace', fake_replay)

    args = SimpleNamespace(
        ants=2,
        cand_list_size=2,
        backup_list_size=2,
        min_new_edges=1,
        decay=0.9,
        alpha=1.0,
        p_best=0.05,
        use_local_search=False,
        disable_heuristic=False,
        extend_ls=False,
        smooth_mmas=False,
        fixed_steps=0,
        nls=False,
        nls_beta=0.2,
        T_nls=10,
        parallel_traced=False,
        seed=0,
        batch_size=1,
        train_H=2,
        train_mini_H=1,
        ppo_epochs=1,
        ppo_clip=0.1,
        entropy_coeff=0.0,
        no_adv_norm=True,
        max_grad_norm=1.0,
    )

    model = FakeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    dynaco_ppo.train_instance_dynaco(model, optimizer, [(None, {})], args, epoch=1, step_idx=0)

    assert replayed_pairs == [(1.0, 1.0), (2.0, 2.0)]
