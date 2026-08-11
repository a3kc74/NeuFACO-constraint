import numpy as np
import pytest

from solvers import pyvrp_local_search


def test_batched_local_search_uses_threads_for_parallel_calls(monkeypatch):
    calls = []

    def fake_local_search(path, **kwargs):
        calls.append(path.tolist())
        return path

    class FailingProcessPool:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ProcessPoolExecutor should not be used for local-search batching")

    monkeypatch.setattr(pyvrp_local_search, "pyvrp_local_search", fake_local_search)
    monkeypatch.setattr(pyvrp_local_search.concurrent.futures, "ProcessPoolExecutor", FailingProcessPool)

    paths = np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]])
    positions = np.zeros((4, 2))
    demands = np.zeros(4)
    windows = np.zeros((4, 2))
    distances = np.ones((4, 4))

    new_paths = pyvrp_local_search.pyvrp_batched_local_search(
        paths,
        positions,
        demands,
        windows,
        distances,
        n_cpus=2,
    )

    assert calls == paths.tolist()
    np.testing.assert_array_equal(new_paths, paths)
