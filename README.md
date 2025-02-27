# Expert Parallelism Load Balancer (EPLB)

When using expert parallelism (EP), different experts are assigned to different GPUs. Because the load of different 
experts may vary depending on the current workload, it is important to keep the load of different GPUs balanced. 
As described in the DeepSeek-V3 paper, we adopt a **redundant experts** strategy that duplicates heavy-loaded experts. 
Then, we heuristically pack the duplicated experts to GPUs to ensure load balancing across different GPUs. Moreover, 
thanks to the **group-limited expert routing** used in DeepSeek-V3, we also attempt to place the experts of the same 
group to the same node to reduce inter-node data traffic, whenever possible.

To facilitate reproduction and deployment, we open-source our deployed EP load balancing algorithm in `eplb.py`. 
The algorithm computes a balanced expert replication and placement plan based on the estimated expert loads. Note 
that the exact method to predict the loads of experts is out of this repo's scope. A common method is to use 
moving average of historical statistics. 

## The Algorithm

The load balancing algorithm comes with two policies used for different cases.

### Hierarchical Load Balancing

When the number of server nodes divides the number of expert groups, we use the hierarchical load balancing policy to
harness the group-limited expert routing. We first pack the expert groups to nodes evenly, ensuring the loads of 
different nodes are balanced. Then, we replicate the experts within each node. Finally, we pack the replicated experts 
to individual GPUs to ensure different GPUs are load-balanced. The hierarchical load balancing policy can be used in 
prefilling stage with a smaller expert-parallel size.

### Global Load Balancing

In other cases, we use the global load balancing policy that replicates the experts globally regardless of expert 
groups, and pack the replicated experts to individual GPUs. This policy can be adopted in decoding stage with a larger 
expert-parallel size.

## Interface and Example

The main function of the load balancer is `eplb.rebalance_experts`.

The following code illustrates an example of a two-layer MoE model, and each layer contains 12 experts. We introduce 4 redundant experts per layer, and the total 16 replicas are placed on 2 nodes, and each node contains 4 GPUs.

``` python
import torch
import eplb

weight = torch.tensor([[ 90, 132,  40,  61, 104, 165,  39,   4,  73,  56, 183,  86],
                       [ 20, 107, 104,  64,  19, 197, 187, 157, 172,  86,  16,  27]])

num_replicas = 16
num_groups = 4
num_nodes = 2
num_gpus = 8

phy2log, log2phy, logcnt = eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)
print(phy2log)

# Output:
# tensor([[ 5,  6,  5,  7,  8,  4,  3,  4, 10,  9, 10,  2,  0,  1, 11,  1],
#         [ 7, 10,  6,  8,  6, 11,  8,  9,  2,  4,  5,  1,  5,  0,  3,  1]])
```

The output, generated by the hierarchical load balancing policy, indicates the following 
expert replication and placement plan.

![](example.png)


## License

This code repository is released under the MIT License.
