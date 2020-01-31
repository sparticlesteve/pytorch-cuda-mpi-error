import torch
import torch.distributed as dist

# Configuration
ranks_per_node = 8
shape = 2**17
dtype = torch.float32

# Test UCX workaround to initialize cuda context before MPI init
import os
local_rank = int(os.getenv('SLURM_LOCALID'))
device = torch.device('cuda', local_rank)

# Allocate a small tensor on every gpu from every rank.
# This is an attempt to force creation of all device contexts.
for i in range(ranks_per_node):
    _ = torch.randn(1).to(torch.device('cuda', i))

# Initialize MPI
dist.init_process_group(backend='mpi')
rank, n_ranks = dist.get_rank(), dist.get_world_size()
local_rank = rank % ranks_per_node

# Select our gpu
#device = torch.device('cuda', local_rank)
print('MPI rank', rank, 'size', n_ranks, 'device', device)

# Allocate a tensor on the gpu
x = torch.randn(shape, dtype=dtype).to(device)
print('local result:', x.sum())

# Do a broadcast from rank 0
dist.broadcast(x, 0)
print('broadcast result:', x.sum())

# Do an all-reduce
dist.all_reduce(x)
print('allreduce result:', x.sum())
