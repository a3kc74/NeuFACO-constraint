import math
import unittest
from unittest.mock import patch

import numpy as np
import torch


class DummyTrace:
    starts = np.array([0, 2], dtype=np.int32)
    curr_nodes = np.array([0, 1], dtype=np.int32)
    is_stochastic = np.array([1, 1], dtype=np.uint8)
    pick_j = np.array([0, 1], dtype=np.int16)
    valid_mask = np.array([0b11, 0b10], dtype=np.uint64)


class DyNACOReplayTest(unittest.TestCase):
    def test_replay_logp_uses_tau_eta_prior_and_valid_mask(self):
        import train_dynaco_ppo

        tau = torch.ones((2, 2), dtype=torch.float32)
        eta = torch.ones((2, 2), dtype=torch.float32)
        prior = torch.zeros((2, 2), dtype=torch.float32)

        logp, entropy = train_dynaco_ppo.replay_logp_from_trace(
            DummyTrace(), tau, eta, prior, alpha=1.0, disable_heuristic=False
        )

        self.assertEqual(tuple(logp.shape), (1,))
        self.assertAlmostEqual(float(logp[0]), -math.log(2.0), places=5)
        self.assertGreater(float(entropy[0]), 0.0)


class DyNACOValidationModeTest(unittest.TestCase):
    def test_both_validation_modes_are_reported_separately(self):
        import train_dynaco_ppo

        val_list = [(object(), {'coords': [], 'demand': [], 'windows': [], 'capacity': 1.0})]

        with patch.object(train_dynaco_ppo, 'infer_validation_instance', side_effect=[[1, 2, 2, 3, 4, 1], [5, 6, 6, 7, 8, 1]]) as infer:
            metrics = train_dynaco_ppo.validate_model(
                model=object(),
                val_list=val_list,
                n_ants=2,
                mode='both',
                epoch=0,
                cand_list_size=2,
                seed=0,
            )

        self.assertEqual(infer.call_count, 2)
        self.assertIn('faco_test_best_T', metrics)
        self.assertIn('faco_test_ib_best_T', metrics)
        self.assertEqual(metrics['faco_test_best_T'], 4.0)
        self.assertEqual(metrics['faco_test_ib_best_T'], 8.0)


if __name__ == '__main__':
    unittest.main()
