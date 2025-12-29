#!/bin/bash

mkdir $SLURM_TMPDIR/validation_data
cp -r /network/scratch/m/marius.mosbach/olmo/validation_data/olmo2/slim_pajama/dolma/* $SLURM_TMPDIR/validation_data/

# Path to the granular directory
BASE_DIR="/network/scratch/a/arkil.patel/olmo/checkpoints"

# Loop through each directory in granular/
for dir in "$BASE_DIR"/*/; do
	# Remove trailing slash and get the directory name
	dir_name=$(basename "$dir")
		    
	# Construct the path to the potential latest-unsharded directory
	unsharded_path="${dir}latest-unsharded"
			    
	# Check if the latest-unsharded directory exists
	if [ -d "$unsharded_path" ]; then
		echo "Found latest-unsharded in $dir_name, running evaluation..."
		CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/eval.py \
			configs/tiny/evaluate/OLMo-136M.yaml \
			--seed=456 \
			--load_path="/network/scratch/a/arkil.patel/olmo/checkpoints/${dir_name}/latest-unsharded"
	else
		echo "No latest-unsharded directory in $dir_name, skipping..."
	fi
done

