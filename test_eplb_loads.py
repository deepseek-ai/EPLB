import unittest

import torch

from eplb import rebalance_experts


class ExpertLoadTests(unittest.TestCase):
    def test_invalid_loads_are_rejected(self):
        for value in (float('nan'), float('inf'), float('-inf'), -1.0, 1e100):
            with self.subTest(value=value):
                # The last value is finite in float64 but overflows on conversion.
                weight = torch.tensor([[value, 9, 8, 7]], dtype=torch.float64)
                with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
                    rebalance_experts(weight, 8, 1, 1, 2)

    def test_valid_loads_return_consistent_maps(self):
        for groups, nodes in ((4, 2), (3, 2)):
            for weight in (torch.zeros((2, 12)), torch.arange(24).reshape(2, 12)):
                with self.subTest(groups=groups, zeros=not weight.any()):
                    phy2log, log2phy, count = rebalance_experts(weight, 16, groups, nodes, 8)
                    self.assertTrue(torch.equal(count.sum(dim=1), torch.full((2,), 16)))
                    for layer in range(2):
                        for expert in range(12):
                            physical = log2phy[layer, expert, :count[layer, expert]]
                            self.assertTrue(torch.all(phy2log[layer, physical] == expert))
                            self.assertTrue(torch.all(log2phy[layer, expert, count[layer, expert]:] == -1))


if __name__ == '__main__':
    unittest.main()
