#!/bin/bash

LINKS=$1
OUTPUT_DIR=$2
PARALLEL_DOWNLOADS=$3

# -n 1 tells xargs to use one URL per wget command
# -P n tells xargs to run n wget commands in parallel
cat $LINKS | xargs -n 1 -P $PARALLEL_DOWNLOADS wget -c -P $OUTPUT_DIR
