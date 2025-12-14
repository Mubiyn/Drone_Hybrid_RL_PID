#!/bin/bash
#
# Train All Hybrid RL Models
#
# Trains Hybrid controllers for all trajectory types using autonomous flight data.
# Uses Behavioral Cloning + RL fine-tuning pipeline for each trajectory.
#
# Usage: ./train_all_hybrid.sh
#

set -e  # Exit on error

echo "======================================================================="
echo "TRAINING ALL HYBRID RL MODELS"
echo "======================================================================="
echo ""

# Configuration
DATA_DIR="data/flight_logs"
BC_EPOCHS=50
FINETUNE_STEPS=500000
NUM_ENVS=4

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    exit 1
fi

# All trajectory types (RL-only training doesn't depend on autonomous data)
TRAJECTORIES=("circle" "square" "figure8" "spiral" "hover")

# Track timing
TOTAL_START=$(date +%s)

# Train each trajectory
for TRAJ in "${TRAJECTORIES[@]}"; do
    echo ""
    echo "======================================================================="
    echo "TRAINING: $TRAJ"
    echo "======================================================================="
    echo ""
    
    TRAJ_START=$(date +%s)
    
    # Check if trajectory file exists
    TRAJ_FILE="data/expert_trajectories/perfect_${TRAJ}_trajectory.pkl"
    if [ ! -f "$TRAJ_FILE" ]; then
        echo "⚠️  Trajectory file not found: $TRAJ_FILE"
        echo "   Skipping $TRAJ..."
        continue
    fi
    
    echo "   Training with RL only (PID baseline + residual corrections)..."
    echo ""
    
    # Train with RL only (no BC needed - PID provides baseline behavior)
    python scripts/train_hybrid_rl_only.py \
        --trajectory "$TRAJ" \
        --train-steps "$FINETUNE_STEPS" \
        --num-envs "$NUM_ENVS" \
        --domain-randomization
    
    TRAJ_END=$(date +%s)
    TRAJ_DURATION=$((TRAJ_END - TRAJ_START))
    
    echo ""
    echo "✓ $TRAJ training complete (${TRAJ_DURATION}s)"
    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo ""
echo "======================================================================="
echo "ALL TRAINING COMPLETE"
echo "======================================================================="
echo "Total time: ${TOTAL_DURATION}s ($((TOTAL_DURATION / 60))m $((TOTAL_DURATION % 60))s)"
echo ""
echo "Trained models saved in: logs/hybrid/"
echo ""
echo "Next steps:"
echo "  1. Test models: python scripts/test_hybrid.py --model logs/hybrid/circle/.../final_model"
echo "  2. Deploy to Tello: python scripts/deploy_hybrid_to_tello.py --model ..."
echo "  3. Compare: Open-loop vs Hybrid with added weight/perturbations"
echo ""
echo "======================================================================="
