# Experimental Results Hub

Welcome to the results gallery for the Drone Hybrid RL-PID project. All experimental data, figures, videos, and analysis are organized here.

---

##  Quick Navigation

| Resource | Description | Link |
|----------|-------------|------|
| ** Phase 1 Results** | Simulation validation with domain randomization | [Phase 1 Directory](#phase-1-simulation-results) |
| ** Phase 2 Results** | Real Tello drone deployment | [Phase 2 Directory](#phase-2-hardware-results) |
| ** Figures Gallery** | All plots and visualizations | [Figures](#figures-gallery) |
| **Videos** | Flight demonstrations | [Videos](#video-demonstrations) |
| **📋 Detailed Analysis** | Complete results document | [Results Summary](../docs/05_results.md) |

---

## Phase 1: Simulation Results

**Directory**: [figures/phase1_simulation/](figures/phase1_simulation/)

**Configuration**:
- Platform: PyBullet CF2X simulation
- Models: `models/hybrid_robust/`
- Residual Scale: 200 RPM
- Domain Randomization: ±20% mass/inertia, 0.05N wind

### Performance Summary

| Trajectory | PID RMSE | Hybrid RMSE | Improvement | p-value |
|------------|----------|-------------|-------------|---------|
| **Spiral** | 0.260m | **0.071m** | **+73.7%** | p < 0.001 ✓ |
| **Circle** | 0.192m | **0.096m** | **+50.3%** | p < 0.01 ✓ |
| **Waypoint** | 0.156m | **0.111m** | **+28.7%** | p < 0.01 ✓ |
| **Hover** | 0.157m | **0.123m** | **+21.5%** | p < 0.05 ✓ |
| Figure8 | 0.146m | **0.136m** | **+6.8%** | p < 0.10 ~ |

### Key Figures

**Trajectory Comparisons**:
- [Spiral Tracking](figures/phase1_simulation/trajectory_plots/spiral_comparison.png)
- [Circle Tracking](figures/phase1_simulation/trajectory_plots/circle_comparison.png)
- [Figure8 Tracking](figures/phase1_simulation/trajectory_plots/figure8_comparison.png)
- [Hover Tracking](figures/phase1_simulation/trajectory_plots/hover_comparison.png)
- [Waypoint Tracking](figures/phase1_simulation/trajectory_plots/waypoint_comparison.png)

**Analysis Plots**:
- [Perturbation Test Results](figures/phase1_simulation/perturbation_tests/)
- [Training Curves](figures/phase1_simulation/training_curves/)

---

## Phase 2: Hardware Results

**Directory**: [figures/phase2_real_drone/](figures/phase2_real_drone/)

**Configuration**:
- Platform: DJI Tello EDU (80g quadrotor)
- Models: `logs/hybrid_tello_drone/*/rl_only_*/`
- Residual Scale: 100 RPM (reduced for stability)
- Domain Randomization: ±30% mass/inertia, 0.15N wind

### Performance Summary

| Trajectory | PID RMSE | Hybrid RMSE | Improvement | Status |
|------------|----------|-------------|-------------|--------|
| **Spiral** | 0.142m | **0.113m** | **+20.7%** |  Success |
| **Hover** | 0.168m | **0.134m** | **+20.2%** |  Success |
| Circle | - | - | - | ⚠️ Too fast for Tello |

### Key Figures

**Real Flight Analysis**:
- [Spiral Wind Perturbation](figures/phase2_real_drone/perturbation_analysis/spiral/comparison_spiral_wind.png)
- [Hover Wind Perturbation](figures/phase2_real_drone/perturbation_analysis/hover/comparison_hover_wind.png)
- [All Trajectories Summary](figures/phase2_real_drone/perturbation_analysis/summary_all_trajectories.png)

**Sim-to-Real Analysis**:
- [Sim vs Real Performance Gap](figures/phase2_real_drone/sim_to_real_comparison.png)

---

## Figures Gallery

### By Category

**Phase 1 Simulation**:
```
figures/phase1_simulation/
├── trajectory_plots/        # Trajectory tracking comparisons
├── perturbation_tests/      # Robustness analysis
├── training_curves/         # Learning progress
└── ablation_studies/        # Component analysis
```

**Phase 2 Hardware**:
```
figures/phase2_real_drone/
├── perturbation_analysis/   # Wind disturbance tests
├── autonomous_analysis/     # Autonomous flight data
└── model_analysis/          # Model performance
```

**Additional Figures**:
```
figures/
├── architecture_diagrams/   # System architecture
├── methodology_diagrams/    # Research approach
└── comparative_analysis/    # Controller comparisons
```

---

## Video Demonstrations

### Simulation Videos

**PID Baseline** (with perturbations):
- [Circle](videos/simulation/pid/circle.mp4) - 656 KB
- [Figure8](videos/simulation/pid/figure8.mp4) - 648 KB
- [Hover](videos/simulation/pid/hover.mp4) - 136 KB
- [Spiral](videos/simulation/pid/spiral.mp4)
- [Waypoint](videos/simulation/pid/waypoint.mp4) - 615 KB

**Hybrid RL-PID** (with perturbations):
- [Circle](videos/simulation/hybrid/circle.mp4) - 661 KB
- [Figure8](videos/simulation/hybrid/figure8.mp4) - 653 KB
- [Hover](videos/simulation/hybrid/hover.mp4) - 598 KB
- [Spiral](videos/simulation/hybrid/spiral.mp4) - 617 KB
- [Waypoint](videos/simulation/hybrid/waypoint.mp4)

### Hardware Videos

**Real Tello Flights**:
- Hover baseline + wind tests
- Spiral baseline + wind tests

 Videos available in repository: [videos/hardware/](videos/hardware/)

---

## Data Files

### Flight Logs

**Simulation Data**:
```
phase1_simulation/
└── perturbation_tests/      # JSON logs with trajectories
```

**Hardware Data**:
```
phase2_real_drone/
├── perturbation_analysis/   # Wind test data
└── autonomous_analysis/     # Autonomous flight logs
```

### Metrics Tables

**CSV Files**:
- [Phase 1 Metrics](tables/phase1_metrics.csv) - Simulation performance
- [Phase 2 Metrics](tables/phase2_metrics.csv) - Hardware performance
- [Statistical Tests](tables/statistical_analysis.csv) - Significance tests

---

## Configuration Comparison

### Why Parameters Changed Between Phases

The evolution from Phase 1 to Phase 2 reflects systematic sim-to-real transfer:

| Parameter | Phase 1 | Phase 2 | Reason |
|-----------|---------|---------|--------|
| **Residual Scale** | 200 RPM | 100 RPM | Tello has limited control authority; lower residual prevents oscillations |
| **Mass DR** | ±20% | ±30% | Real world has more uncertainty; increased robustness needed |
| **Inertia DR** | ±20% | ±30% | Better generalization to varying real-world conditions |
| **Wind DR** | 0.05N | 0.15N | Real indoor air currents stronger than expected |
| **Trajectories** | All 5 | Hover, Spiral | Circle/Figure8/Square too fast for 80g Tello |

**This is intentional engineering**, not inconsistency. See [Methodology](../docs/03_methodology.md) for detailed rationale.

---

## Browse Results

### By Trajectory

**Hover**:
- [Phase 1 Simulation](figures/phase1_simulation/trajectory_plots/hover_comparison.png)
- [Phase 2 Hardware](figures/phase2_real_drone/perturbation_analysis/hover/)

**Circle**:
- [Phase 1 Simulation](figures/phase1_simulation/trajectory_plots/circle_comparison.png)
- Phase 2: Too fast for Tello hardware

**Spiral**:
- [Phase 1 Simulation](figures/phase1_simulation/trajectory_plots/spiral_comparison.png)
- [Phase 2 Hardware](figures/phase2_real_drone/perturbation_analysis/spiral/)

**Figure8**:
- [Phase 1 Simulation](figures/phase1_simulation/trajectory_plots/figure8_comparison.png)
- Phase 2: Too fast for Tello hardware

**Waypoint**:
- [Phase 1 Simulation](figures/phase1_simulation/trajectory_plots/waypoint_comparison.png)
- Phase 2: Not tested on hardware

---

## Detailed Analysis

For complete experimental analysis, methodology, and discussion:

📄 **[Results Summary Document](../docs/05_results.md)**

Topics covered:
- Statistical significance tests
- Ablation studies
- Failure case analysis
- Sim-to-real transfer lessons
- Future improvements

---

## Reproduce Results

### Phase 1: Simulation

```bash
# Run perturbation tests
python scripts/phase1_simulation/test_simulation_perturbations.py

# Generate plots
python scripts/phase1_simulation/analyze_simulation_results.py

# Results saved to: results/phase1_simulation/
```

### Phase 2: Hardware

```bash
# Connect to Tello WiFi first
python scripts/phase2_real_drone/test_all_with_perturbations.py

# Analyze results
python scripts/phase2_real_drone/analyze_perturbation_tests.py

# Results saved to: results/phase2_real_drone/
```

---

## Data Organization

```
results/
├── README.md                    # This file - results hub
│
├── figures/                     # All plots and visualizations
│   ├── phase1_simulation/       # Simulation results
│   │   ├── trajectory_plots/    # Trajectory comparisons
│   │   ├── perturbation_tests/  # Robustness analysis
│   │   └── training_curves/     # Learning progress
│   └── phase2_real_drone/       # Hardware results
│       ├── perturbation_analysis/
│       └── autonomous_analysis/
│
├── videos/                      # Flight demonstrations
│   ├── simulation/              # PyBullet videos
│   │   ├── pid/                 # PID baseline
│   │   └── hybrid/              # Hybrid controller
│   └── hardware/                # Real Tello videos
│
├── tables/                      # Numeric results (CSV)
│   ├── phase1_metrics.csv
│   ├── phase2_metrics.csv
│   └── statistical_analysis.csv
│
└── logs/                        # Raw experiment logs
    ├── training/                # Training logs
    └── evaluation/              # Evaluation logs
```

---

## Citation

If you use these results in your research, please cite:

```bibtex
@misc{drone_hybrid_rl_pid_2024,
  title={Hybrid RL-PID Control for Quadrotor Trajectory Tracking},
  author={Bokono Bennett Nathan, Emanuel Israel Okpara, Adzembeh Joshua, Mubin Sheidu},
  year={2025},
  howpublished={\url{https://github.com/Mubiyn/Drone_Hybrid_RL_PID}}
}
```

---

**Last Updated**: December 18, 2025  
**Need Help?** See [Documentation Hub](../docs/README.md)
