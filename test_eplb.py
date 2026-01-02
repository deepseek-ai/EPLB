"""
Tests for EPLB (Expert Parallelism Load Balancer)

These tests verify the correctness of expert replication and placement,
including regression tests for known issues.
"""

import torch
import eplb


def check_no_duplicate_experts_per_gpu(phy2log: torch.Tensor, num_gpus: int) -> bool:
    """
    Verify that no GPU has duplicate logical experts assigned to it.

    Args:
        phy2log: [num_layers, num_physical_experts], logical expert id for each physical expert
        num_gpus: total number of GPUs

    Returns:
        True if no duplicates found, False otherwise
    """
    num_layers, num_phy = phy2log.shape
    experts_per_gpu = num_phy // num_gpus

    for layer in range(num_layers):
        for gpu in range(num_gpus):
            start = gpu * experts_per_gpu
            end = start + experts_per_gpu
            experts_on_gpu = phy2log[layer, start:end].tolist()
            if len(experts_on_gpu) != len(set(experts_on_gpu)):
                duplicates = [x for x in experts_on_gpu if experts_on_gpu.count(x) > 1]
                print(f"Duplicate experts found! Layer {layer}, GPU {gpu}: {experts_on_gpu}")
                print(f"Duplicates: {set(duplicates)}")
                return False
    return True


def test_readme_example():
    """Test the example from the README works correctly."""
    weight = torch.tensor([[ 90, 132,  40,  61, 104, 165,  39,   4,  73,  56, 183,  86],
                           [ 20, 107, 104,  64,  19, 197, 187, 157, 172,  86,  16,  27]])

    num_replicas = 16
    num_groups = 4
    num_nodes = 2
    num_gpus = 8

    phy2log, log2phy, logcnt = eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

    # Check shapes
    assert phy2log.shape == (2, 16), f"Expected (2, 16), got {phy2log.shape}"
    assert log2phy.shape[0] == 2 and log2phy.shape[1] == 12, f"Unexpected log2phy shape: {log2phy.shape}"
    assert logcnt.shape == (2, 12), f"Expected (2, 12), got {logcnt.shape}"

    # Check no duplicate experts per GPU
    assert check_no_duplicate_experts_per_gpu(phy2log, num_gpus), "Found duplicate experts on same GPU!"

    # Check all logical experts are covered
    for layer in range(2):
        logical_experts = set(phy2log[layer].tolist())
        assert len(logical_experts) >= 12, f"Layer {layer}: Not all logical experts are assigned"

    print("README example test passed!")


def test_no_duplicate_experts_issue_22():
    """
    Regression test for issue #22: phy2log can generate duplicate hot expert ids on same rank.

    This test uses a configuration similar to the bug report where 256 logical experts
    with 32 redundant experts are distributed across 32 GPUs on 2 nodes.
    """
    # Simplified version of the bug report scenario
    num_logical_experts = 256
    num_redundant = 32
    num_physical = num_logical_experts + num_redundant  # 288
    num_gpus = 32
    num_nodes = 2
    num_groups = 8
    num_layers = 4

    # Create realistic load distribution with some experts being much hotter than others
    torch.manual_seed(42)
    weight = torch.rand(num_layers, num_logical_experts) * 100
    # Make some experts much hotter (to trigger replication)
    hot_experts = torch.randint(0, num_logical_experts, (20,))
    weight[:, hot_experts] *= 10

    phy2log, log2phy, logcnt = eplb.rebalance_experts(
        weight, num_physical, num_groups, num_nodes, num_gpus
    )

    # Main assertion: no duplicate logical experts on the same GPU
    assert check_no_duplicate_experts_per_gpu(phy2log, num_gpus), \
        "Issue #22 regression: Found duplicate logical experts on same GPU!"

    # Verify basic properties
    assert phy2log.shape == (num_layers, num_physical)
    assert logcnt.shape == (num_layers, num_logical_experts)

    # Check that replication happened (some experts should have count > 1)
    assert (logcnt > 1).any(), "Expected some experts to be replicated"

    # Total replicas should match
    assert logcnt.sum(dim=-1).tolist() == [num_physical] * num_layers

    print("Issue #22 regression test passed!")


def test_high_replication_scenario():
    """
    Test scenario with very high replication where duplicates are more likely.
    """
    num_logical_experts = 64
    num_physical = 128  # 2x replication on average
    num_gpus = 16
    num_nodes = 2
    num_groups = 4
    num_layers = 2

    # Create highly skewed load where some experts will be heavily replicated
    weight = torch.ones(num_layers, num_logical_experts)
    weight[:, :8] = 100  # First 8 experts are 100x hotter

    phy2log, log2phy, logcnt = eplb.rebalance_experts(
        weight, num_physical, num_groups, num_nodes, num_gpus
    )

    # Check no duplicates
    assert check_no_duplicate_experts_per_gpu(phy2log, num_gpus), \
        "Found duplicate logical experts in high replication scenario!"

    # Check hot experts are replicated more
    assert (logcnt[:, :8] > 1).all(), "Hot experts should be replicated"

    print("High replication scenario test passed!")


def test_global_load_balancing():
    """Test the global load balancing policy (when num_groups % num_nodes != 0)."""
    weight = torch.tensor([[10, 20, 30, 40, 50, 60],
                           [60, 50, 40, 30, 20, 10]])

    num_replicas = 12
    num_groups = 3  # Not divisible by num_nodes=2
    num_nodes = 2
    num_gpus = 4

    phy2log, log2phy, logcnt = eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

    assert phy2log.shape == (2, 12)
    assert check_no_duplicate_experts_per_gpu(phy2log, num_gpus), \
        "Found duplicate experts in global load balancing mode!"

    print("Global load balancing test passed!")


def test_single_replica_per_expert():
    """Test when there are no redundant experts (1:1 mapping)."""
    weight = torch.rand(2, 16)

    num_replicas = 16  # Same as logical experts
    num_groups = 4
    num_nodes = 2
    num_gpus = 8

    phy2log, log2phy, logcnt = eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

    # All counts should be 1 (no replication)
    assert (logcnt == 1).all(), "Expected no replication"

    # Each logical expert appears exactly once
    for layer in range(2):
        assert sorted(phy2log[layer].tolist()) == list(range(16))

    print("Single replica per expert test passed!")


if __name__ == "__main__":
    test_readme_example()
    test_no_duplicate_experts_issue_22()
    test_high_replication_scenario()
    test_global_load_balancing()
    test_single_replica_per_expert()
    print("\nAll tests passed!")
