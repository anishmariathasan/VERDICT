#!/bin/bash
# Script to run the full experiment pipeline

CONFIG="configs/base_config.yaml"

echo "Running VERDICT experiments..."

# 1. Generate Attributions
echo "Generating attributions..."
python experiments/generate_attributions.py --config $CONFIG

# 2. Train Baseline (optional)
# echo "Training baseline..."
# python experiments/train_baseline.py --config $CONFIG

# 3. Evaluate
echo "Evaluating..."
python experiments/run_evaluation.py --config $CONFIG

echo "Experiments complete."
