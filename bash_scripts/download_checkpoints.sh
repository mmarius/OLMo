#!/bin/bash

OUTPUT_DIR=$1

for step in 10000 50000 100000 500000
do
    echo "Downloading checkpoint files at step $step"
    wget --directory-prefix=$OUTPUT_DIR/step$step-unsharded https://olmo-checkpoints.org/ai2-llm/olmo-small/s7wptaol/step$step-unsharded/config.yaml
    wget --directory-prefix=$OUTPUT_DIR/step$step-unsharded https://olmo-checkpoints.org/ai2-llm/olmo-small/s7wptaol/step$step-unsharded/model.pt
    wget --directory-prefix=$OUTPUT_DIR/step$step-unsharded https://olmo-checkpoints.org/ai2-llm/olmo-small/s7wptaol/step$step-unsharded/optim.pt
    wget --directory-prefix=$OUTPUT_DIR/step$step-unsharded https://olmo-checkpoints.org/ai2-llm/olmo-small/s7wptaol/step$step-unsharded/train.pt
done
