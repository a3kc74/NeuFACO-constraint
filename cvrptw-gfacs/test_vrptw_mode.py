import unittest

import torch

from utils import gen_instance


class VRPTWModeTest(unittest.TestCase):
    def test_vrptw_generation_sets_all_demands_to_zero(self):
        demands, distances, positions, windows = gen_instance(10, "cpu", vrptw=True)

        self.assertTrue(torch.allclose(demands, torch.zeros_like(demands)))
        self.assertEqual(distances.shape, (11, 11))
        self.assertEqual(positions.shape, (11, 2))
        self.assertEqual(windows.shape, (11, 2))

    def test_cvrptw_generation_keeps_positive_customer_demands(self):
        demands, *_ = gen_instance(10, "cpu", vrptw=False)

        self.assertEqual(demands[0].item(), 0.0)
        self.assertGreater(demands[1:].sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
