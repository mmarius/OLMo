#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=8                               
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=4:00:00
#SBATCH -o /network/scratch/a/arkil.patel/olmo/slurm_logs/olmo_eval_267M_decay-%j.out

# 1. Load the required modules
module load python/3.10 cuda/12.6.0/cudnn

# 2. Load your environment
source ~/envs/scalenv/bin/activate
cd ~/OLMo

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/eval.py configs/slim_pajama/evaluate/OLMo-267M-132_decay.yaml --seed=479 --load_path=/network/scratch/a/arkil.patel/olmo/checkpoints/granular/OLMo_267M_479_decay_0-244ce7be579fab8b/latest-unsharded

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/eval.py configs/slim_pajama/evaluate/OLMo-267M-132_decay.yaml --seed=479 --load_path=/network/scratch/a/arkil.patel/olmo/checkpoints/granular/OLMo_267M_479_decay_50-7b30d016975d90d4/latest-unsharded

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/eval.py configs/slim_pajama/evaluate/OLMo-267M-132_decay.yaml --seed=479 --load_path=/network/scratch/a/arkil.patel/olmo/checkpoints/granular/OLMo_267M_479_decay_100-1e264e4c91343ec4/latest-unsharded

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/eval.py configs/slim_pajama/evaluate/OLMo-267M-132_decay.yaml --seed=479 --load_path=/network/scratch/a/arkil.patel/olmo/checkpoints/granular/OLMo_267M_479_decay_150-5cb3e21737a164a0/latest-unsharded

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/eval.py configs/slim_pajama/evaluate/OLMo-267M-132_decay.yaml --seed=479 --load_path=/network/scratch/a/arkil.patel/olmo/checkpoints/granular/OLMo_267M_479_decay_200-eadafc03a137a9a5/latest-unsharded