import unittest

import torch

from eplb import rebalance_experts


class TopologyTests(unittest.TestCase):
    def test_invalid_node_gpu_counts_rejected_for_both_policies(self):
        weight = torch.arange(1, 7).reshape(1, 6)
        for groups in (2, 3):
            for nodes, gpus in ((0, 2), (2, 0), (-1, 2), (2, -1), (2, 3), (4, 2)):
                with self.subTest(groups=groups, nodes=nodes, gpus=gpus):
                    with self.assertRaisesRegex(ValueError, 'num_gpus must be a multiple of num_nodes'):
                        rebalance_experts(weight, 6, groups, nodes, gpus)

    def test_valid_hierarchical_and_global_topologies(self):
        weight = torch.arange(1, 7).reshape(1, 6)
        for groups in (2, 3):
            with self.subTest(groups=groups):
                phy2log, log2phy, count = rebalance_experts(weight, 6, groups, 2, 2)
                self.assertEqual(phy2log.shape, (1, 6))
                self.assertEqual(log2phy.shape[:2], (1, 6))
                self.assertEqual(count.sum().item(), 6)
                self.assertEqual(sorted(phy2log[0].tolist()), list(range(6)))


if __name__ == '__main__':
    unittest.main()
