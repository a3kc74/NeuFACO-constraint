from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation import faco_test


def test_load_ppo_prior_model_loads_local_net_checkpoint(tmp_path):
    model = faco_test.PPONet(value_head=False, norm_type="batch")
    checkpoint_path = tmp_path / "best.pt"
    torch.save(model.state_dict(), checkpoint_path)

    loaded = faco_test.load_ppo_prior_model(checkpoint_path, "cpu", norm_type="batch")

    assert isinstance(loaded, faco_test.PPONet)
    assert not loaded.training


def test_ppo_prior_uses_active_solver_and_edge_feature_mode(monkeypatch):
    captured = {}

    class FakePygData:
        n_nodes = 3
        k_sparse = 2

    class FakePPO(torch.nn.Module):
        def forward(self, pyg_data):
            assert pyg_data is captured["pyg_data"]
            return torch.tensor([1.0, 3.0, 2.0, 6.0, 4.0, 8.0])

        @staticmethod
        def reshape(pyg_data, vector):
            return vector.reshape(pyg_data.n_nodes, pyg_data.k_sparse)

    class FakeSolver:
        nn_list = np.array([[2, 1], [0, 2], [1, 0]], dtype=np.int32)

    def fake_gen_pyg_data(instance, **kwargs):
        captured["instance"] = instance
        captured["kwargs"] = kwargs
        captured["pyg_data"] = FakePygData()
        return captured["pyg_data"]

    monkeypatch.setattr(faco_test.ppo_utils, "gen_pyg_data", fake_gen_pyg_data)

    instance = {"coords": torch.zeros((3, 2)), "demand": torch.zeros(3), "windows": torch.ones((3, 2)), "capacity": 1.0}
    prior = faco_test.ppo_prior(
        FakePPO(),
        instance,
        solver=FakeSolver(),
        cand_list_size=2,
        backup_list_size=4,
        n_ants=5,
        min_new_edges=1,
        device="cpu",
        edge_feature_mode="static",
        prior_scale=0.5,
        prior_center=True,
    )

    assert prior.shape == (3, 2)
    assert np.allclose(prior, np.array([[-0.5, 0.5], [-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32))
    assert captured["kwargs"]["solver"].nn_list.tolist() == [[2, 1], [0, 2], [1, 0]]
    assert captured["kwargs"]["edge_feature_mode"] == "static"
    assert captured["kwargs"]["cand_list_size"] == 2


def test_sample_multisource_recomputes_ppo_prior_after_each_source_route():
    calls = []

    class FakeSolver:
        def set_source_route(self, route, _cost):
            calls.append(("set", int(route[1])))

        def sample(self, prior=None):
            calls.append(("sample", float(prior[0, 0])))
            return np.array([1.0], dtype=np.float32), [np.array([0, 1, 0], dtype=np.int32)]

    def prior_fn():
        source_values = [item[1] for item in calls if item[0] == "set"]
        return np.array([[float(source_values[-1])]], dtype=np.float32)

    archive = [
        {"route": np.array([0, 3, 0], dtype=np.int32), "cost": 3.0},
        {"route": np.array([0, 7, 0], dtype=np.int32), "cost": 7.0},
    ]

    faco_test.sample_multisource(FakeSolver(), archive, source_count=2, prior_fn=prior_fn)

    assert calls == [("set", 3), ("sample", 3.0), ("set", 7), ("sample", 7.0)]


def test_infer_instance_reuses_ppo_prior_by_outer_iteration(monkeypatch):
    priors = []

    class FakeSolver:
        nn_list = np.array([[1], [0]], dtype=np.int32)

        def __init__(self, *_args, **_kwargs):
            self.state = 0

        def seed_rng(self, _seed):
            return None

        def sample(self, prior=None):
            priors.append(float(prior[0, 0]))
            costs = np.array([2.0], dtype=np.float32)
            routes = [np.array([0, 1, 0], dtype=np.int32)]
            return costs, routes, None, None, None, costs, None, None, None

        def update_pheromone(self, _route, _cost):
            self.state += 1

        def set_source_route(self, _route, _cost):
            return None

    def fake_ppo_prior(_model, _instance, solver, **_kwargs):
        return np.array([[float(solver.state)], [float(solver.state)]], dtype=np.float32)

    monkeypatch.setattr(faco_test, "MFACO_CVRPTW", FakeSolver)
    monkeypatch.setattr(faco_test, "ppo_prior", fake_ppo_prior)

    faco_test.infer_instance(
        demands=torch.zeros(2),
        distances=torch.ones((2, 2)),
        positions=torch.zeros((2, 2)),
        windows=torch.ones((2, 2)),
        n_ants=1,
        n_iter=1,
        mini_H=3,
        threads=1,
        seed=0,
        cand_list_size=1,
        backup_list_size=1,
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
        T_nls=10,
        elite_k=0,
        ppo_model=object(),
        ppo_device="cpu",
    )

    assert priors == [0.0, 0.0, 0.0]


def test_infer_instance_can_recompute_ppo_prior_for_each_sample(monkeypatch):
    priors = []

    class FakeSolver:
        nn_list = np.array([[1], [0]], dtype=np.int32)

        def __init__(self, *_args, **_kwargs):
            self.state = 0

        def seed_rng(self, _seed):
            return None

        def sample(self, prior=None):
            priors.append(float(prior[0, 0]))
            costs = np.array([2.0], dtype=np.float32)
            routes = [np.array([0, 1, 0], dtype=np.int32)]
            return costs, routes, None, None, None, costs, None, None, None

        def update_pheromone(self, _route, _cost):
            self.state += 1

        def set_source_route(self, _route, _cost):
            return None

    def fake_ppo_prior(_model, _instance, solver, **_kwargs):
        return np.array([[float(solver.state)], [float(solver.state)]], dtype=np.float32)

    monkeypatch.setattr(faco_test, "MFACO_CVRPTW", FakeSolver)
    monkeypatch.setattr(faco_test, "ppo_prior", fake_ppo_prior)

    faco_test.infer_instance(
        demands=torch.zeros(2),
        distances=torch.ones((2, 2)),
        positions=torch.zeros((2, 2)),
        windows=torch.ones((2, 2)),
        n_ants=1,
        n_iter=1,
        mini_H=3,
        threads=1,
        seed=0,
        cand_list_size=1,
        backup_list_size=1,
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
        T_nls=10,
        elite_k=0,
        ppo_model=object(),
        ppo_device="cpu",
        ppo_prior_refresh="sample",
    )

    assert priors == [0.0, 1.0, 2.0]


def test_main_rejects_gfacs_and_ppo_together(tmp_path, monkeypatch):
    monkeypatch.setattr(faco_test, "load_dataset", lambda *_args, **_kwargs: [])

    with pytest.raises(ValueError, match="gfacs_pretrained.*ppo_pretrained"):
        faco_test.main(
            n_nodes=2,
            gfacs_pretrained=tmp_path / "gfacs.pt",
            ppo_pretrained=tmp_path / "ppo.pt",
        )
