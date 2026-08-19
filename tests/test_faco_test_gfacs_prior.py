import numpy as np
import torch

from evaluation import faco_test


def test_sample_multisource_forwards_prior_to_each_source():
    prior = np.arange(6, dtype=np.float32).reshape(2, 3)

    class FakeSolver:
        def __init__(self):
            self.priors = []

        def set_source_route(self, _route, _cost):
            return None

        def sample(self, prior=None):
            self.priors.append(prior)
            return np.array([1.0], dtype=np.float32), [np.array([0, 1, 0], dtype=np.int32)]

    solver = FakeSolver()
    archive = [
        {"route": np.array([0, 1, 0], dtype=np.int32), "cost": 1.0},
        {"route": np.array([0, 2, 0], dtype=np.int32), "cost": 2.0},
    ]

    faco_test.sample_multisource(solver, archive, source_count=2, prior=prior)

    assert solver.priors == [prior, prior]


def test_deepaco_prior_gathers_model_matrix_by_solver_candidate_list(monkeypatch):
    captured = {}

    class FakePygData:
        pass

    class FakeDeepACO(torch.nn.Module):
        def forward(self, pyg_data):
            assert pyg_data is captured["pyg_data"]
            return torch.arange(9, dtype=torch.float32)

        @staticmethod
        def reshape(_pyg_data, _vector):
            return torch.tensor(
                [
                    [0.0, 0.2, 0.8],
                    [0.5, 0.0, 0.0],
                    [0.0, 0.4, 0.0],
                ],
                dtype=torch.float32,
            )

    class FakeSolver:
        nn_list = np.array([[2, 1], [0, 2], [1, 0]], dtype=np.int32)

    def fake_gen_deepaco_pyg_data(demands, distances, windows, device, k_sparse):
        captured["args"] = (demands, distances, windows, device, k_sparse)
        captured["pyg_data"] = FakePygData()
        return captured["pyg_data"]

    monkeypatch.setattr(faco_test, "gen_deepaco_pyg_data", fake_gen_deepaco_pyg_data)

    prior = faco_test.deepaco_prior(
        FakeDeepACO(),
        {
            "demand": torch.zeros(3),
            "distances": torch.ones((3, 3)),
            "windows": torch.ones((3, 2)),
        },
        solver=FakeSolver(),
        k_sparse=2,
        device="cpu",
        prior_scale=1.0,
        prior_center=False,
    )

    assert prior.shape == (3, 2)
    assert np.allclose(prior[0], np.log(np.array([0.8, 0.2], dtype=np.float32)))
    assert prior[1, 0] == np.log(np.float32(0.5))
    assert prior[1, 1] == 0.0
    assert captured["args"][4] == 2


def test_deepaco_prior_can_row_center_and_scale(monkeypatch):
    class FakePygData:
        pass

    class FakeDeepACO(torch.nn.Module):
        def forward(self, _pyg_data):
            return torch.arange(4, dtype=torch.float32)

        @staticmethod
        def reshape(_pyg_data, _vector):
            return torch.tensor([[0.25, 1.0], [0.5, 0.125]], dtype=torch.float32)

    class FakeSolver:
        nn_list = np.array([[0, 1], [0, 1]], dtype=np.int32)

    monkeypatch.setattr(faco_test, "gen_deepaco_pyg_data", lambda *_args, **_kwargs: FakePygData())

    prior = faco_test.deepaco_prior(
        FakeDeepACO(),
        {"demand": torch.zeros(2), "distances": torch.ones((2, 2)), "windows": torch.ones((2, 2))},
        solver=FakeSolver(),
        k_sparse=2,
        device="cpu",
        prior_scale=0.5,
        prior_center=True,
    )

    expected = np.array([[-0.3465736, 0.3465736], [0.3465736, -0.3465736]], dtype=np.float32)
    assert np.allclose(prior, expected)


def test_load_deepaco_prior_model_loads_deepaco_trainer_checkpoint(tmp_path):
    model = faco_test.OriginalGFACSNet(gfn=False)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(model.state_dict(), checkpoint_path)

    loaded = faco_test.load_deepaco_prior_model(checkpoint_path, "cpu")

    assert isinstance(loaded, faco_test.OriginalGFACSNet)
    assert not loaded.training

