#!/bin/bash

# Automated PID Baseline Data Collection
# Runs all 6 tasks + OOD tests sequentially
# Total estimated time: ~2-3 hours

set -e  # Exit on error

echo "=========================================="
echo "PID Baseline - Complete Data Collection"
echo "=========================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"
echo ""

# Activate conda environment
echo "Activating conda environment: drone-rl-pid"
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
    echo "✓ Completed: $task"
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
    python scripts/test_ood.py --task "$task" --trials 5
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
echo "  - OOD tests: pid_ood_<task>_*.json/csv"
echo ""
echo "Total files expected: 24 (12 JSON + 12 CSV)"
echo ""

# Count files
JSON_COUNT=$(ls results/data/pid_*.json 2>/dev/null | wc -l | tr -d ' ')
CSV_COUNT=$(ls results/data/pid_*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "Files created: $JSON_COUNT JSON, $CSV_COUNT CSV"
echo ""

echo "Next steps:"
echo "  1. Review results in results/data/"
echo "  2. Proceed to Week 2: PPO training (scripts/run_all_ppo.sh)"
echo ""
