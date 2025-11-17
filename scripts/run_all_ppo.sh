#!/bin/bash

# Automated PPO Training for All Tasks
# Trains PPO models for all 6 tasks sequentially
# Total estimated time: 12-18 hours (2-3 hours per task)

set -e  # Exit on error

echo "=========================================="
echo "PPO Training - All 6 Tasks"
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

# Create model directories
mkdir -p models/ppo logs
echo "Models will be saved to: models/ppo/"
echo "Logs will be saved to: logs/"
echo ""

# Configuration
TIMESTEPS=1000000  # 1M steps per task
TASKS=("hover" "hover_extended" "waypoint_delivery" "figure8" "circle" "emergency_landing")

echo "Configuration:"
echo "  Timesteps per task: $TIMESTEPS"
echo "  Total tasks: ${#TASKS[@]}"
echo "  Estimated time: 12-18 hours"
echo ""

# Start TensorBoard in background
echo "Starting TensorBoard server..."
tensorboard --logdir logs --port 6006 --bind_all > /dev/null 2>&1 &
TENSORBOARD_PID=$!
echo "✓ TensorBoard running at: http://localhost:6006"
echo "  PID: $TENSORBOARD_PID"
echo ""
sleep 3

# Training loop
TASK_NUM=1
TOTAL_TASKS=${#TASKS[@]}
START_TIME=$(date +%s)

for task in "${TASKS[@]}"; do
    echo "=========================================="
    echo "Task $TASK_NUM/$TOTAL_TASKS: $task"
    echo "=========================================="
    echo ""
    
    TASK_START=$(date +%s)
    
    echo "Starting training at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Expected duration: 2-3 hours"
    echo ""
    
    # Train the model
    python scripts/train_ppo.py --task "$task" --timesteps "$TIMESTEPS"
    
    TASK_END=$(date +%s)
    TASK_DURATION=$((TASK_END - TASK_START))
    TASK_HOURS=$((TASK_DURATION / 3600))
    TASK_MINS=$(((TASK_DURATION % 3600) / 60))
    
    echo ""
    echo "✓ Completed: $task"
    echo "  Duration: ${TASK_HOURS}h ${TASK_MINS}m"
    echo ""
    
    # Run evaluation
    echo "Running evaluation..."
    MODEL_PATH="models/ppo/$task/ppo_${task}_final.zip"
    if [ -f "$MODEL_PATH" ]; then
        python scripts/train_ppo.py --task "$task" --eval-only --model-path "$MODEL_PATH"
        echo ""
    else
        echo "⚠ Warning: Model file not found at $MODEL_PATH"
        echo ""
    fi
    
    TASK_NUM=$((TASK_NUM + 1))
    
    # Progress update
    ELAPSED=$(($(date +%s) - START_TIME))
    ELAPSED_HOURS=$((ELAPSED / 3600))
    ELAPSED_MINS=$(((ELAPSED % 3600) / 60))
    REMAINING_TASKS=$((TOTAL_TASKS - TASK_NUM + 1))
    
    if [ $TASK_NUM -le $TOTAL_TASKS ]; then
        echo "-------------------------------------------"
        echo "Progress: $((TASK_NUM - 1))/$TOTAL_TASKS tasks complete"
        echo "Elapsed time: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
        echo "Remaining tasks: $REMAINING_TASKS"
        echo "-------------------------------------------"
        echo ""
        sleep 5  # Brief pause between tasks
    fi
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINS=$(((TOTAL_DURATION % 3600) / 60))

echo "=========================================="
echo "PPO TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo "Tasks completed: $TOTAL_TASKS"
echo ""
echo "Models saved to: models/ppo/"
echo "Logs saved to: logs/"
echo ""

# List created models
echo "Trained models:"
for task in "${TASKS[@]}"; do
    MODEL_PATH="models/ppo/$task/ppo_${task}_final.zip"
    if [ -f "$MODEL_PATH" ]; then
        SIZE=$(du -h "$MODEL_PATH" | cut -f1)
        echo "  ✓ $task: $SIZE"
    else
        echo "  ✗ $task: MISSING"
    fi
done
echo ""

echo "TensorBoard logs available at: http://localhost:6006"
echo "  TensorBoard PID: $TENSORBOARD_PID (still running)"
echo ""
echo "To stop TensorBoard: kill $TENSORBOARD_PID"
echo ""

echo "Next steps:"
echo "  1. Review training curves in TensorBoard"
echo "  2. Implement Hybrid controller (Phase 2b)"
echo "  3. Train Hybrid models"
echo ""
