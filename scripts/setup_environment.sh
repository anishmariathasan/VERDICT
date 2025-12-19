#!/bin/bash
# Script to set up the development environment

echo "Setting up VERDICT environment..."

# Create conda environment
conda create -n verdict python=3.10 -y
source activate verdict

# Install dependencies
pip install -e ".[dev]"

echo "Environment setup complete. Activate with 'conda activate verdict'"
