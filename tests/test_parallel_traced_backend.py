import unittest

import numpy as np

from solvers.faco import set_faco_cpp_threads
from envs.cvrptw_env import build_solver, generate_cvrptw_instance


class ParallelTracedBackendTest(unittest.TestCase):
    def test_parallel_traced_matches_serial_traced_for_same_seed(self):
        set_faco_cpp_threads(4)
        instance = generate_cvrptw_instance(30, seed=123)
        kwargs = dict(
            n_ants=8,
            cand_list_size=16,
            backup_list_size=32,
            min_new_edges=8,
            decay=0.9,
            alpha=1.0,
            p_best=0.05,
            use_local_search=True,
            disable_heuristic=False,
            extend_ls=True,
            smooth_mmas=True,
            fixed_steps=0,
            nls=False,
            T_nls=10,
            device='cpu',
        )
        prior = np.zeros((30, 16), dtype=np.float32)

        def run(parallel_traced):
            solver = build_solver(instance, **kwargs)
            solver.seed_rng(777)
            return solver.sample(require_prob=True, prior=prior, parallel_traced=parallel_traced)

        serial = run(False)
        parallel = run(True)
        for serial_arr, parallel_arr in zip((serial[0], serial[5], serial[7]), (parallel[0], parallel[5], parallel[7])):
            np.testing.assert_array_equal(serial_arr, parallel_arr)
        self.assertEqual([list(route) for route in serial[1]], [list(route) for route in parallel[1]])
        self.assertEqual([list(route) for route in serial[6]], [list(route) for route in parallel[6]])

        serial_trace = serial[4]
        parallel_trace = parallel[4]
        for name in ('starts', 'start_nodes', 'curr_nodes', 'chosen_nodes', 'is_stochastic', 'pick_j', 'valid_mask', 'is_new_edge'):
            np.testing.assert_array_equal(getattr(serial_trace, name), getattr(parallel_trace, name))


if __name__ == '__main__':
    unittest.main()
