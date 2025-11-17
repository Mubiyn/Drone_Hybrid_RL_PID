#!/bin/bash
# Complete PID Baseline Data Collection Script
# Run this to collect all Week 1 baseline results

set -e  # Exit on error

echo "=================================================="
echo "PID Baseline Data Collection - Week 1"
echo "=================================================="
echo ""

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate drone-rl-pid
cd /Users/MMD/Desktop/MLR/Drone_Hybrid_RL_PID

# Create results directories
mkdir -p results/data results/figures

echo "STEP 1: Individual Task Tests (6 tasks)"
echo "=================================================="

echo "[1/6] Testing hover..."
python scripts/train_pid.py --task hover

echo "[2/6] Testing hover_extended (30s)..."
python scripts/train_pid.py --task hover_extended

echo "[3/6] Testing waypoint_delivery..."
python scripts/train_pid.py --task waypoint_delivery

echo "[4/6] Testing figure8..."
python scripts/train_pid.py --task figure8

echo "[5/6] Testing circle..."
python scripts/train_pid.py --task circle

echo "[6/6] Testing emergency_landing..."
python scripts/train_pid.py --task emergency_landing

echo ""
echo "STEP 2: OOD Robustness Tests (6 tasks × 7 scenarios)"
echo "=================================================="

echo "[1/6] OOD test: hover (5 trials per scenario)"
python scripts/test_ood.py --task hover --trials 5

echo "[2/6] OOD test: hover_extended (5 trials per scenario)"
python scripts/test_ood.py --task hover_extended --trials 5

echo "[3/6] OOD test: waypoint_delivery (5 trials per scenario)"
python scripts/test_ood.py --task waypoint_delivery --trials 5

echo "[4/6] OOD test: figure8 (5 trials per scenario)"
python scripts/test_ood.py --task figure8 --trials 5

echo "[5/6] OOD test: circle (5 trials per scenario)"
python scripts/test_ood.py --task circle --trials 5

echo "[6/6] OOD test: emergency_landing (5 trials per scenario)"
python scripts/test_ood.py --task emergency_landing --trials 5

echo ""
echo "=================================================="
echo "✅ Data Collection Complete!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - JSON: results/data/pid_*.json"
echo "  - CSV:  results/data/pid_*.csv"
echo "  - Plots: results/figures/pid_*.png"
echo ""
echo "Total files generated:"
ls -1 results/data/*.json | wc -l | xargs -I {} echo "  - {} JSON files"
ls -1 results/data/*.csv | wc -l | xargs -I {} echo "  - {} CSV files"
ls -1 results/figures/*.png | wc -l | xargs -I {} echo "  - {} PNG plots"
echo ""
echo "Next step: Analyze results and proceed to Week 2 (PPO training)"
