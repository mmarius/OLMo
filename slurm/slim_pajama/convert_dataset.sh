#!/bin/bash

#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:59:00
#SBATCH -o /network/scratch/m/marius.mosbach/slurm-logs/olmo/slurm-%j.out

# Load the required modules
module --quiet load python/3.10

# CD to project dir
PROJECT_DIR=$HOME/projects/olmo
cd $HOME/projects/olmo

# Update default cache dir of huggingface transformers and datasets
export HF_HOME=$SCRATCH/hf-cache-dir
export HF_DATASETS_CACHE=$SCRATCH/hf-cache-dir

# Activate virtual env
source $HOME/venvs/olmo/bin/activate

# Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
# python main.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR

echo "Preparing data..."

INPUT_DIR=$SCRATCH/hf-cache-dir/DKYoon___slim_pajama-6_b/default/0.0.0/b5f90f419b7489cdba26fdbc8c022fcb5562f968
OUTPUT_DIR=$SCRATCH/olmo/training_data/slim_pajama

python scripts/convert_arrow_datasets.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --pattern "slim_pajama-6_b-train" \
    --num_workers 10

echo "Done!"