import torch

from evaluation import gfacs_test


def test_load_checkpoint_model_uses_deepaco_net_without_logz(monkeypatch, tmp_path):
    calls = []

    class FakeNet:
        def __init__(self, gfn=False, Z_out_dim=1):
            calls.append((gfn, Z_out_dim))

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(gfacs_test, "Net", FakeNet)
    monkeypatch.setattr(torch, "load", lambda path, map_location=None: {"weight": torch.tensor([1.0])})

    model = gfacs_test.load_checkpoint_model(str(ckpt_path), guided_exploration=True, deepaco=True)

    assert isinstance(model, FakeNet)
    assert calls == [(False, 1)]
    assert model.state_dict["weight"].item() == 1.0


def test_load_checkpoint_model_keeps_gfacs_logz_head(monkeypatch, tmp_path):
    calls = []

    class FakeNet:
        def __init__(self, gfn=False, Z_out_dim=1):
            calls.append((gfn, Z_out_dim))

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(gfacs_test, "Net", FakeNet)
    monkeypatch.setattr(torch, "load", lambda path, map_location=None: {"weight": torch.tensor([1.0])})

    gfacs_test.load_checkpoint_model(str(ckpt_path), guided_exploration=True, deepaco=False)

    assert calls == [(True, 2)]
