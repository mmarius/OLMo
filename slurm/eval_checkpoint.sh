#!/bin/bash

#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=11:59:00
#SBATCH -o /network/scratch/m/marius.mosbach/slurm-logs/olmo/slurm-%j.out

# Load the required modules
module --quiet load python/3.10

# Look at GPUs
nvidia-smi

# CD to project dir
PROJECT_DIR=$HOME/projects/olmo
cd $HOME/projects/olmo

# Activate virtual env
source $HOME/venvs/olmo/bin/activate

# Update default cache dir of huggingface transformers and datasets
export HF_HOME=$SCRATCH/hf-cache-dir
export HF_DATASETS_CACHE=$SCRATCH/hf-cache-dir

# Load arguments
CONFIG_PATH=$1

echo "Starting evaluation..."

# Run python script
torchrun --nproc_per_node=1 \
    scripts/eval.py $CONFIG_PATH

# Copy whatever you want to save on $SCRATCH
# touch $SLURM_TMPDIR/test_tmp.txt
# cp $SLURM_TMPDIR/test_tmp.txt $SCRATCH/