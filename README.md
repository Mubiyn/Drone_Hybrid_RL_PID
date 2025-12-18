# Hybrid RL-PID Drone Control System

> **A two-phase research project combining Reinforcement Learning with PID control for robust quadrotor trajectory tracking, validated in simulation and deployed on real DJI Tello hardware.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

##  Quick Links

| Category | Links |
|----------|-------|
| ** Documentation** | [Full Documentation Hub](docs/README.md) • [Getting Started](docs/01_getting_started.md) • [Installation](docs/02_installation.md) |
| ** Research** | [Methodology](docs/03_methodology.md) • [Architecture](docs/04_architecture.md) |
| ** Results** | [Results Summary](docs/05_results.md) • [Figures Gallery](results/README.md) • [Videos](results/videos/) |
| ** Hardware** | [Hardware Setup](docs/06_hardware_setup.md) • [Docker Guide](docs/07_docker_guide.md) • [Advanced Topics](docs/advanced/) |

---

##  Key Features

- **Hybrid Control Architecture**: PID provides stable baseline, RL learns residual corrections
- **Domain Randomization**: Robust training with ±30% mass/inertia variation and 0.15N wind
- **Two-Phase Validation**: Simulation → Real hardware deployment
- **20%+ Performance Gains**: Improved tracking over PID baseline on real Tello drone
- **Open Source**: Complete implementation with trained models and documentation

---

##  Results Highlight

### Phase 1: Simulation (with Domain Randomization)

| Trajectory | PID Baseline | Hybrid RL-PID | Improvement | Significance |
|------------|--------------|---------------|-------------|--------------|
| **Spiral** | 0.260m | **0.071m** | **+73.7%** | p < 0.001 ✓ |
| **Circle** | 0.192m | **0.096m** | **+50.3%** | p < 0.01 ✓ |
| **Waypoint** | 0.156m | **0.111m** | **+28.7%** | p < 0.01 ✓ |
| Hover | 0.157m | **0.123m** | **+21.5%** | p < 0.05 ✓ |

[ Full Results & Analysis →](docs/05_results.md)

### Phase 2: Hardware Deployment (DJI Tello)

| Trajectory | Tello Hardware | Improvement | Status |
|------------|----------------|-------------|--------|
| **Spiral** | **0.113m** | **+20.7%** |  Success |
| **Hover** | **0.134m** | **+20.2%** |  Success |

[ Hardware Results Details →](docs/05_results.md#phase-2-hardware)

### Key Visualization

<details>
<summary><b> Spiral Trajectory Tracking Comparison</b></summary>

![Spiral Comparison](results/figures/phase1_simulation/trajectory_plots/spiral_comparison.png)

*Hybrid RL-PID (blue) vs PID baseline (orange) under domain randomization*

</details>

[Video Demonstrations →](results/videos/README.md) | [ More Figures →](results/README.md)

---

##  Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Mubiyn/Drone_Hybrid_RL_PID.git
cd Drone_Hybrid_RL_PID

# Create environment (conda recommended)
conda env create -f environment.yml
conda activate drone-hybrid-rl

# Install simulation environment
cd gym-pybullet-drones && pip install -e . && cd ..

# Verify installation
python scripts/test_installation.py
```

**Detailed Instructions**: [Installation Guide](docs/02_installation.md)

### Run Phase 1: Simulation Tests

```bash
# Test hybrid model on circle trajectory
python src/testing/demo_simulation.py --controller hybrid --trajectory circle

# Test with domain randomization
python src/testing/demo_simulation.py --controller hybrid --trajectory spiral --dr

# Run full evaluation comparison (all trajectories)
python src/testing/eval_comparison.py

# Run perturbation analysis
python scripts/phase1_simulation/test_simulation_perturbations.py
```

### Run Phase 2: Real Drone Deployment

```bash
# Connect to Tello WiFi (TELLO-XXXXXX), then:
python scripts/phase2_real_drone/test_hybrid_on_tello.py --trajectory spiral

# Test with wind perturbations
python scripts/phase2_real_drone/test_all_with_perturbations.py
```

**Complete Guide**: [Getting Started](docs/01_getting_started.md)

---

##  Project Structure

```
Drone_Hybrid_RL_PID/
├── README.md                    # This file - project overview
├── docs/                        #  All documentation
│   ├── README.md                #    Documentation hub
│   ├── 01_getting_started.md    #    Quick start guide
│   ├── 02_installation.md       #    Installation guide
│   ├── 03_methodology.md        #    Research methodology
│   ├── 04_architecture.md       #    System architecture
│   ├── 05_results.md            #    Results analysis
│   ├── 07_docker_guide.md       #    Docker setup
│   └── advanced/                #    Advanced topics (MoCap, etc.)
│
├── src/                         # Source code
│   ├── controllers/             #    Control algorithms (PID)
│   ├── envs/                    #    RL environments
│   ├── training/                #    Training pipelines
│   ├── testing/                 #    Testing utilities
│   ├── hardware/                #    Hardware interface (Tello, MoCap)
│   └── utils/                   #    Shared utilities
│
├── scripts/                     #  Executable scripts
│   ├── test_installation.py     #    Installation verification
│   ├── phase1_simulation/       #    Simulation experiments
│   ├── phase2_real_drone/       #    Hardware deployment
│   ├── training_scripts/        #    Model training
│   ├── data_generation/         #    Trajectory generation
│   └── shared/                  #    Analysis tools
│
├── models/                      #  Trained models
│   └── hybrid_robust/           #    Phase 1 models (5 trajectories)
│
├── results/                     #  Experimental results
│   ├── README.md                #    Results hub
│   ├── figures/                 #    Plots and visualizations
│   │   ├── phase1_simulation/   #    Simulation results
│   │   └── phase2_real_drone/   #    Hardware results
│   └── videos/                  #    Flight demonstrations
│       ├── hybrid/              #    Hybrid controller videos
│       └── pid/                 #    PID baseline videos
│
├── data/                        # 📁 Data files
│   ├── expert_trajectories/     #    Reference trajectories
│   └── flight_logs/             #    Flight recordings
│
└── gym-pybullet-drones/         #  Simulation environment
```

[Detailed Structure →](REFACTORING_PLAN.md#proposed-structure-after-refactoring)

---

##  Two-Phase Methodology

### Phase 1: Simulation Validation
- **Environment**: PyBullet CF2X simulation
- **Residual Scale**: 200 RPM (strong RL corrections)
- **Domain Randomization**: ±20% mass/inertia, 0.05N wind
- **Models**: `models/hybrid_robust/`

**Result**: Hybrid outperforms PID on all dynamic trajectories with strong robustness

### Phase 2: Hardware Deployment
- **Hardware**: DJI Tello EDU (80g quadrotor)
- **Residual Scale**: 100 RPM (conservative for stability)
- **Domain Randomization**: ±30% mass/inertia, 0.15N wind
- **Models**: `logs/hybrid_tello_drone/*/rl_only_*/`

**Result**: Successful sim-to-real transfer with 20%+ improvements

[ Complete Methodology →](docs/03_methodology.md)

---

## 🛠️ Technologies

- **Simulation**: [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
- **RL Framework**: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO)
- **Hardware**: [djitellopy](https://djitellopy.readthedocs.io/) (DJI Tello SDK)
- **Control**: Custom PID + RL residual architecture
- **Python**: 3.10+ with PyTorch, NumPy, Matplotlib

---

## Citation

```bibtex
@misc{drone_hybrid_rl_pid_2025,
  title={Hybrid RL-PID Control for Quadrotor Trajectory Tracking},
  author={Bokono Bennett Nathan, Emanuel Israel Okpara, Adzembeh Joshua, Mubin Sheidu},
  year={2025},
  howpublished={\url{https://github.com/Mubiyn/Drone_Hybrid_RL_PID}}
}
```

---

## Acknowledgments

- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) for simulation environment
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL algorithms
- Course instructors and TAs for guidance

---

## 📞 Contact

- **Repository**: [github.com/Mubiyn/Drone_Hybrid_RL_PID](https://github.com/Mubiyn/Drone_Hybrid_RL_PID)
- **Issues**: [GitHub Issues](https://github.com/Mubiyn/Drone_Hybrid_RL_PID/issues)

---

<div align="center">

**[ Documentation](docs/README.md)** • **[ Results](docs/05_results.md)** • **[ Hardware](docs/06_hardware_setup.md)**

*Built with  for robust drone control*

</div>
