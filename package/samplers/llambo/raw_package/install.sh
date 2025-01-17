#!/bin/bash

# Step 1: Initialize Conda (necessary to enable 'conda activate' in non-login shells)
eval "$(conda shell.bash hook)"

# Step 2: Create a conda environment
conda create -n optunahub_llambo python=3.10 -y

# Step 3: Activate the environment
conda activate optunahub_llambo

echo "Current Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
sleep 5


# Step 4: Install packages
pip install -e .

echo "Setup complete!"