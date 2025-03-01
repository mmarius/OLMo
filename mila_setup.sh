#!/bin/bash

# (Optional) Setup git on the terminal
# git config user.name "TODO" # TODO: fill in
# git config user.email "TODO" # TODO: fill in

# Install virtual environment
PROJECT_NAME="olmo"
PYTHON_VERSION="3.10"

## Create a virtual environment if it doesn't exist
if [ ! -d "$HOME/venvs/$PROJECT_NAME" ]; then
    module load python/$PYTHON_VERSION
    python3 -m venv $HOME/venvs/$PROJECT_NAME

    # Modify the virtual environment to load the python module
    echo "module load python/$PYTHON_VERSION" >> $HOME/venvs/$PROJECT_NAME/bin/activate
fi

## Activate the virtual environment
source $HOME/venvs/$PROJECT_NAME/bin/activate

# Install basic dependencies
pip install --upgrade pip
pip install wheel
pip install -e .[all]
pip install -r requirements_mila.txt

# (Optional) Install torch
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# (Optional) Install flash-attention
# module load cudatoolkit/12.1
# pip install triton "flash-attn>=2.5.0" --no-build-isolation