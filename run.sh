#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -t 10

module load gcc/7.3.0
module load cuda/10.1.168
module load nccl/2.4.8
module load openmpi/4.0.1-ucx-1.6
module load pytorch/v1.3.1-gpu

ucx_info -f -d > ucx_info.txt

export UCX_LOG_LEVEL=debug
export UCX_NET_DEVICES=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1
export OMPI_MCA_btl_smcuda_use_cuda_ipc=0
export OMPI_MCA_btl=^openib,tcp
export OMPI_MCA_osc=ucx

# Try preload UCX 1.7
export LD_LIBRARY_PATH=/global/cfs/cdirs/nstaff/sfarrell/software/ucx-1.7.0/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/global/cfs/cdirs/nstaff/sfarrell/software/ucx-1.7.0/lib/libucp.so

srun -n 2 -c 10 -u -l python test_mpi_cuda.py
