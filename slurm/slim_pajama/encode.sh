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

echo "Encoding data with dolma toolkit ..."

TOKENIZER_PATH=allenai/dolma2-tokenizer # olmo2 defaults
EOS_TOKEN_ID=100257 # olmo2 defaults
PAD_TOKEN_ID=100277 # olmo2 defaults
DTYPE=uint32 # olmo2 defaults

INPUT_FILE=$SCRATCH/olmo/training_data/olmo2/slim_pajama/json/*.jsonl.gz
OUTPUT_DIR=$SCRATCH/olmo/training_data/olmo2/slim_pajama/dolma

# see: https://github.com/allenai/dolma/blob/main/docs/tokenize.md
# for how to use the dolma toolkit to tokenize data

dolma tokens \
    --documents $INPUT_FILE \
    --destination $OUTPUT_DIR \
    --tokenizer.name_or_path $TOKENIZER_PATH \
    --tokenizer.eos_token_id $EOS_TOKEN_ID \
    --tokenizer.pad_token_id $PAD_TOKEN_ID \
    --dtype $DTYPE \
    --processes 8 \
    --seed 123

echo "Done!"