#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=8                               
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/a/arkil.patel/olmo/slurm_logs/olmo_train_267M_decay_G2-%j.out

# 1. Load the required modules
module load python/3.10 cuda/12.6.0/cudnn

# 2. Load your environment
source ~/envs/scalenv/bin/activate
cd ~/OLMo

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py configs/slim_pajama/OLMo-267M.yaml --seed=479 --run_name=OLMo_267M_479_G2 --device_train_microbatch_size=24 --max_duration=500