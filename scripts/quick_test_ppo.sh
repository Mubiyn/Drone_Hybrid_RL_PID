#!/bin/bash

# Quick test script for PPO training
# Usage: ./scripts/quick_test_ppo.sh

echo "Testing PPO training on hover task (short run)"
echo "=============================================="

python scripts/train_ppo.py \
    --task hover \
    --timesteps 50000 \
    --learning-rate 3e-4

echo ""
echo "Training complete! Check models/ppo/hover/ for saved models"
echo "Check logs/ for TensorBoard logs"
