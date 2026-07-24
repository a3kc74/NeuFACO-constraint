import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_dynaco_ppo import replay_logp_from_trace, resolve_run_name
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
