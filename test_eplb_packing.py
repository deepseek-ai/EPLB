import unittest
from unittest.mock import patch

import torch

import eplb


def reference_packing(weight, num_packs):
    """Slow tensor-scalar implementation of the original greedy policy."""
    num_layers, num_groups = weight.shape
    groups_per_pack = num_groups // num_packs
    if groups_per_pack == 1:
        return (torch.arange(num_groups, dtype=torch.int64).expand(num_layers, -1),
                torch.zeros_like(weight, dtype=torch.int64))
    order = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full((num_layers, num_groups), -1, dtype=torch.int64)
    rank = torch.full_like(pack_index, -1)
    for layer in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in order[layer]:
            pack = min((i for i in range(num_packs) if pack_items[i] < groups_per_pack),
                       key=pack_weights.__getitem__)
            pack_index[layer, group] = pack
            rank[layer, group] = pack_items[pack]
            pack_weights[pack] += weight[layer, group]
            pack_items[pack] += 1
    return pack_index, rank


class TestBalancedPacking(unittest.TestCase):
    def assert_same_packing(self, weight, num_packs):
        expected = reference_packing(weight, num_packs)
        actual = eplb.balanced_packing(weight, num_packs)
        for x, y in zip(actual, expected):
            self.assertTrue(torch.equal(x, y))

    def test_cpu_float32_matches_original_policy(self):
        generator = torch.Generator().manual_seed(42)
        for layers in (1, 3):
            for groups in (4, 16, 32):
                for packs in (1, 2, 4):
                    samples = (
                        torch.rand((layers, groups), generator=generator) * 1e8,
                        torch.randint(0, 4, (layers, groups), generator=generator).float(),
                        torch.rand((layers, groups), generator=generator) * 2e38,
                        torch.rand((layers, groups * 2), generator=generator)[:, ::2],
                    )
                    for weight in samples:
                        with self.subTest(layers=layers, groups=groups, packs=packs):
                            self.assert_same_packing(weight, packs)

    def test_other_dtypes_and_empty_batch(self):
        for dtype in (torch.int64, torch.float64):
            self.assert_same_packing(torch.tensor([[3, 1, 4, 1], [0, 8, 0, 2]], dtype=dtype), 2)
        pack, rank = eplb.balanced_packing(torch.empty(0, 8), 2)
        self.assertEqual(pack.shape, (0, 8))
        self.assertEqual(rank.shape, (0, 8))

    def test_cpu_output_ignores_default_device(self):
        weight = torch.tensor([[4, 3, 2, 1]], dtype=torch.float32, device='cpu')
        expected = reference_packing(weight, 2)
        previous = torch.get_default_device()
        try:
            torch.set_default_device('meta')
            actual = eplb.balanced_packing(weight, 2)
        finally:
            torch.set_default_device(previous)
        for x, y in zip(actual, expected):
            self.assertEqual(x.device.type, 'cpu')
            self.assertTrue(torch.equal(x, y))

    def test_public_policies_match_reference(self):
        weight = torch.randint(0, 10000, (3, 32), generator=torch.Generator().manual_seed(7))
        for groups, nodes in ((4, 2), (3, 2)):
            with self.subTest(groups=groups, nodes=nodes):
                with patch.object(eplb, 'balanced_packing', reference_packing):
                    expected = eplb.rebalance_experts(weight, 40, groups, nodes, 8)
                actual = eplb.rebalance_experts(weight, 40, groups, nodes, 8)
                for x, y in zip(actual, expected):
                    self.assertTrue(torch.equal(x, y))


if __name__ == '__main__':
    unittest.main()
