import unittest
from unittest.mock import patch

import numpy as np
import torch

import trainers.ppo_trainer as train_ppo


class DummyPygData:
    def to(self, device):
        return self


class DummyModel:
    def eval(self):
        pass

    def __call__(self, pyg_data):
        return 'heu'

    def reshape(self, pyg_data, heu_vec):
        return torch.zeros((3, 3), dtype=torch.float32)


class FakeSolver:
    instances = []

    def __init__(self):
        self.sample_calls = 0
        self.update_calls = []
        self.seed = None
        self.set_source_calls = 0
        FakeSolver.instances.append(self)

    def seed_rng(self, seed):
        self.seed = seed

    def sample(self, prior=None):
        self.sample_calls += 1
        costs = np.array([10.0 - self.sample_calls, 20.0], dtype=np.float32)
        routes = np.array([[0, 1, 0], [0, 2, 0]], dtype=np.int32)
        return costs, routes

    def update_pheromone(self, route, cost):
        self.update_calls.append((tuple(route.tolist()), float(cost)))

    def set_source_route(self, route, cost):
        self.set_source_calls += 1


class TrainPpoInferTest(unittest.TestCase):
    def test_infer_matches_faco_test_loop_shape(self):
        FakeSolver.instances = []

        with patch.object(train_ppo, 'build_solver', return_value=FakeSolver()):
            stats = train_ppo.infer_instance(
                DummyModel(),
                DummyPygData(),
                {'coords': [], 'demand': [], 'windows': [], 'capacity': 1.0},
                n_ants=2,
                cand_list_size=3,
                val_n_iter=3,
                val_mini_H=2,
                seed=7,
                instance_idx=4,
                smooth_mmas=True,
                extend_ls=True,
            )

        solver = FakeSolver.instances[0]
        self.assertEqual(solver.seed, 11)
        self.assertEqual(solver.sample_calls, 6)
        self.assertEqual(len(solver.update_calls), 6)
        self.assertEqual(solver.set_source_calls, 0)
        self.assertEqual(stats[3], 8.0)
        self.assertEqual(stats[4], 4.0)

    def test_train_filters_validation_only_kwargs(self):
        captured = {}

        def fake_validation(*args, **kwargs):
            return 1.0

        def fake_train_epoch(*args, **kwargs):
            captured.update(kwargs)

        with patch.object(train_ppo, 'load_val_dataset', return_value=[]), \
             patch.object(train_ppo, 'validation', side_effect=fake_validation), \
             patch.object(train_ppo, 'train_epoch', side_effect=fake_train_epoch), \
             patch.object(train_ppo.torch, 'save'), \
             patch.object(train_ppo, 'Net') as net_cls:
            net_cls.return_value.to.return_value.parameters.return_value = [torch.nn.Parameter(torch.zeros(()))]
            train_ppo.train(
                100,
                20,
                32,
                100,
                1,
                1,
                val_n_iter=10,
                val_mini_H=10,
                seed=0,
                decay=0.9,
                alpha=1.0,
                p_best=0.05,
                smooth_mmas=True,
                extend_ls=True,
            )

        self.assertNotIn('val_n_iter', captured)
        self.assertNotIn('val_mini_H', captured)
        self.assertNotIn('seed', captured)
        self.assertNotIn('decay', captured)
        self.assertNotIn('alpha', captured)
        self.assertNotIn('p_best', captured)
        self.assertNotIn('smooth_mmas', captured)
        self.assertTrue(captured['extend_ls'])


if __name__ == '__main__':
    unittest.main()
