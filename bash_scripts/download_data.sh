#!/bin/bash

FILE=$1
OUTPUT_DIR=$2

for i in $(cat $FILE)
do
    wget --directory-prefix=$OUTPUT_DIR $i
done