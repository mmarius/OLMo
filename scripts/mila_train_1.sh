#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=8                               
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/a/arkil.patel/olmo/slurm_logs/olmo_train_267M_decay-%j.out

# 1. Load the required modules
module load python/3.10 cuda/12.6.0/cudnn

# 2. Load your environment
source ~/envs/scalenv/bin/activate
cd ~/OLMo

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/train.py configs/slim_pajama/OLMo-267M.yaml --seed=132 --run_name=OLMo_267M_132

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/train.py configs/slim_pajama/OLMo-267M_decay.yaml --seed=479 --device_train_microbatch_size=24 --save_folder=/network/scratch/a/arkil.patel/olmo/checkpoints/granular --load_path=/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_267M_479_Granular-e260889bdc1acd21/step4000-unsharded --max_duration=5000 --scheduler.t_decay=1000 --run_name=OLMo_267M_479_decay_4000

# 0.8m = u, m = 1.25u, t = 0.25u