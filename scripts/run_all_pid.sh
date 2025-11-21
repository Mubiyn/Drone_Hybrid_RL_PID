#!/bin/bash

# Automated PID Baseline Data Collection Script
# This script runs all 6 PID tasks and their corresponding OOD tests sequentially.
# Total estimated time: ~1-2 hours depending on the machine.

set -e # Exit immediately if a command exits with a non-zero status.

echo "=========================================="
echo "PID Baseline - Complete Data Collection"
echo "=========================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/.." || exit
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"
echo ""

# Activate conda environment.
# This is a more robust way to activate conda, which works in most shell environments.
echo "Activating conda environment: drone-rl-pid..."
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please ensure conda is installed and in your PATH."
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drone-rl-pid
echo ""

# Create results directories
mkdir -p results/data results/figures
echo "Results will be saved to: results/data/"
echo ""

# Phase 1: Individual task testing
echo "=========================================="
echo "PHASE 1: Individual Task Testing (6 tasks)"
echo "=========================================="
echo ""

TASKS=("hover" "hover_extended" "waypoint_delivery" "figure8" "circle" "emergency_landing")

for task in "${TASKS[@]}"; do
    echo "-------------------------------------------"
    echo "Running PID on task: $task"
    echo "-------------------------------------------"
    python scripts/train_pid.py --task "$task"
    echo ""
    echo "✓ Completed PID test for: $task"
    echo ""
    sleep 2  # Brief pause between tasks
done

echo "=========================================="
echo "✓ PHASE 1 COMPLETE: All 6 tasks tested"
echo "=========================================="
echo ""

# Phase 2: OOD robustness testing
echo "=========================================="
echo "PHASE 2: OOD Robustness Testing (6 tasks × 7 scenarios)"
echo "=========================================="
echo ""

for task in "${TASKS[@]}"; do
    echo "-------------------------------------------"
    echo "Running OOD tests for: $task"
    echo "-------------------------------------------"
    python scripts/test_ood.py --task "$task" --trials 3 # Reduced trials for faster runs
    echo ""
    echo "✓ Completed OOD: $task"
    echo ""
    sleep 2
done

echo "=========================================="
echo "✓ PHASE 2 COMPLETE: All OOD tests done"
echo "=========================================="
echo ""

# Summary
echo "=========================================="
echo "PID BASELINE COLLECTION COMPLETE!"
echo "=========================================="
echo ""
echo "Data files saved to: results/data/"
echo "  - Individual tasks: pid_<task>_*.json/csv"
echo "  - OOD tests:      pid_ood_<task>_*.json/csv"
echo ""
echo "Total files expected: ~24 (12 JSON + 12 CSV)"
echo ""

# Count files
JSON_COUNT=$(find results/data -name 'pid_*.json' 2>/dev/null | wc -l | tr -d ' ')
CSV_COUNT=$(find results/data -name 'pid_*.csv' 2>/dev/null | wc -l | tr -d ' ')
echo "Files created: $JSON_COUNT JSON, $CSV_COUNT CSV"
echo ""

echo "Next steps:"
echo "  1. Review results in results/data/"
echo "  2. Proceed to Week 2: PPO training (e.g., using scripts/train_ppo.py)"
echo ""
