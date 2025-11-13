# Figures and Plots

This directory contains generated figures from training and evaluation.

## Figure Types

### Training Curves
- **reward_curve.png:** Episode rewards over time
- **loss_curve.png:** Policy and value loss
- **success_rate.png:** Success rate progression
- **learning_rate.png:** Learning rate schedule

### Evaluation Metrics
- **position_error.png:** Position tracking error distributions
- **trajectory_comparison.png:** Desired vs actual trajectories
- **control_effort.png:** Action magnitudes over time
- **energy_consumption.png:** Total energy per episode

### Ablation Studies
- **pid_vs_rl_vs_hybrid.png:** Performance comparison
- **domain_randomization_impact.png:** With/without DR
- **architecture_comparison.png:** Different network architectures

### Real Drone Results
- **sim_vs_real.png:** Sim-to-real transfer analysis
- **real_flight_trajectory.png:** Actual Tello flight paths

## Generating Figures

Figures are automatically generated during training and evaluation:

```bash
# Training generates figures in results/figures/
python scripts/train_rl.py --timesteps 500000

# Evaluation generates comparison plots
python scripts/evaluate.py --model models/hybrid/best_model.zip --n-episodes 50
```

## Manual Figure Generation

```python
from src.utils.visualization import plot_training_curves, plot_trajectory

# Plot training curves
plot_training_curves(
    log_dir='logs/tensorboard/',
    save_path='results/figures/training_curves.png'
)

# Plot trajectory comparison
plot_trajectory(
    desired_traj='data/trajectories/figure_eight.npy',
    actual_traj='results/actual_trajectory.npy',
    save_path='results/figures/trajectory_comparison.png'
)
```

## Figure Quality

All figures are saved in high resolution:
- **Format:** PNG (300 DPI)
- **Size:** 1920x1080 pixels
- **Style:** Seaborn default

For publications, use vector formats:
```python
plt.savefig('results/figures/plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
```

## Organization

```
results/figures/
├── training/           # Training-related plots
├── evaluation/         # Evaluation results
├── ablation/          # Ablation study plots
└── real_drone/        # Real-world results
```
