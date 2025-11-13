# Drone Hybrid RL+PID Control System

## Project Overview

This project implements a hybrid control system for quadrotor drones that combines traditional PID control with Reinforcement Learning (PPO) for robust autonomous flight. The system is trained in PyBullet simulation with domain randomization and can be deployed to real DJI Tello drones.

**Team Size:** 4 members  
**Duration:** 3 weeks  
**Technologies:** gym-pybullet-drones, Stable-Baselines3, PyTorch, Domain Randomization

## Key Features

- **Hybrid Architecture:** PID for low-level stability + RL for high-level decision making
- **Domain Randomization:** Robust training with randomized physics parameters
- **Progressive Complexity:** Start with hovering → waypoint navigation → trajectory tracking
- **Sim-to-Real Transfer:** Tested pipeline from PyBullet to DJI Tello
- **Comprehensive Testing:** Unit tests, integration tests, and real-world validation

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.3+ (optional, for GPU acceleration)
- DJI Tello drone (for real deployment)

### Setup with pip

```bash
# Clone the repository
git clone <your-repo-url>
cd Task1_Drone_Hybrid_RL_PID

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup with conda

```bash
# Create conda environment
conda env create -f environment.yml
conda activate drone-hybrid-rl
```

### Verify Installation

```bash
python scripts/test_installation.py
```

## Quick Start

### 1. Train PID Controller (Week 1)

```bash
# Train PID for hovering
python scripts/train_pid.py --task hover --episodes 100

# Tune PID parameters
python scripts/tune_pid.py --method grid_search
```

### 2. Train RL Agent (Week 2)

```bash
# Train PPO with domain randomization
python scripts/train_rl.py \
    --task waypoint \
    --timesteps 500000 \
    --domain-randomization \
    --save-freq 10000

# Monitor training with TensorBoard
tensorboard --logdir logs/
```

### 3. Train Hybrid System (Week 2)

```bash
# Train hybrid controller
python scripts/train_hybrid.py \
    --pid-model models/ppo/best_hover.pkl \
    --timesteps 300000 \
    --hybrid-mode adaptive
```

### 4. Evaluate and Deploy (Week 3)

```bash
# Evaluate in simulation
python scripts/evaluate.py \
    --model models/hybrid/best_model.zip \
    --n-episodes 50 \
    --render

# Deploy to real Tello drone
python scripts/deploy_real.py \
    --model models/hybrid/best_model.zip \
    --test-mode safe
```

## Project Structure

```
Task1_Drone_Hybrid_RL_PID/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── .gitignore                   # Git exclusions
├── LICENSE                      # MIT License
├── Dockerfile                   # Container setup
├── PROJECT_STRUCTURE.md         # Detailed structure guide
├── QUICKSTART.md               # Quick reference
│
├── config/                      # Configuration files
│   ├── pid_hover_config.yaml
│   ├── rl_waypoint_config.yaml
│   ├── hybrid_trajectory_config.yaml
│   └── domain_randomization.yaml
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── controllers/             # Control algorithms
│   │   ├── __init__.py
│   │   ├── pid_controller.py
│   │   ├── rl_policy.py
│   │   └── hybrid_controller.py
│   ├── training/                # Training logic
│   │   ├── __init__.py
│   │   ├── ppo_trainer.py
│   │   ├── domain_randomizer.py
│   │   └── callbacks.py
│   ├── testing/                 # Testing utilities
│   │   ├── __init__.py
│   │   ├── sim_tester.py
│   │   └── metrics.py
│   ├── real_drone/              # Real drone interface
│   │   ├── __init__.py
│   │   ├── tello_interface.py
│   │   └── safe_deploy.py
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── visualization.py
│       └── logging_utils.py
│
├── scripts/                     # Executable scripts
│   ├── test_installation.py
│   ├── train_pid.py
│   ├── tune_pid.py
│   ├── train_rl.py
│   ├── train_hybrid.py
│   ├── evaluate.py
│   └── deploy_real.py
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_PID_Tuning.ipynb
│   ├── 02_RL_Training_Analysis.ipynb
│   ├── 03_Hybrid_Comparison.ipynb
│   └── Training_on_Colab.ipynb
│
├── tests/                       # Unit tests
│   ├── test_pid.py
│   ├── test_rl_policy.py
│   ├── test_hybrid.py
│   └── test_domain_randomization.py
│
├── models/                      # Trained models
│   ├── ppo/                     # Pure RL models
│   │   └── README.md
│   ├── hybrid/                  # Hybrid models
│   │   └── README.md
│   └── best_model.zip          # Final model
│
├── data/                        # Training data
│   ├── expert_trajectories/
│   │   └── README.md
│   └── flight_logs/
│       └── README.md
│
├── results/                     # Experiment results
│   ├── figures/                 # Plots and charts
│   │   └── README.md
│   ├── videos/                  # Flight recordings
│   │   └── README.md
│   └── metrics.csv             # Performance metrics
│
├── logs/                        # Training logs
│   ├── tensorboard/
│   └── training_logs.txt
│
└── docs/                        # Additional documentation
    ├── ARCHITECTURE.md
    ├── PID_TUNING_GUIDE.md
    ├── RL_TRAINING_GUIDE.md
    └── DEPLOYMENT_GUIDE.md
```

## Usage Guide

### Training PID Controller

The PID controller provides low-level stability. Train it first:

```python
from src.controllers.pid_controller import PIDController
from src.training.ppo_trainer import train_pid

# Initialize PID
pid = PIDController(kp=[0.4, 0.4, 1.0], ki=[0.0, 0.0, 0.1], kd=[0.2, 0.2, 0.5])

# Train on hovering task
train_pid(pid, task='hover', episodes=100)
```

### Training RL Agent

Train PPO agent with domain randomization:

```python
from stable_baselines3 import PPO
from src.training.domain_randomizer import DomainRandomizedEnv

# Create randomized environment
env = DomainRandomizedEnv('waypoint-aviary-v0', randomize_physics=True)

# Train PPO
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='logs/tensorboard/')
model.learn(total_timesteps=500000)
model.save('models/ppo/waypoint_model')
```

### Training Hybrid System

Combine PID and RL:

```python
from src.controllers.hybrid_controller import HybridController
from src.training.ppo_trainer import train_hybrid

# Load PID model
pid = PIDController.load('models/ppo/best_hover.pkl')

# Initialize hybrid controller
hybrid = HybridController(pid_controller=pid, rl_policy=None, mode='adaptive')

# Train hybrid system
train_hybrid(hybrid, task='trajectory', timesteps=300000)
```

### Evaluation

Evaluate trained models:

```python
from src.testing.sim_tester import SimulationTester

# Load model
model = PPO.load('models/hybrid/best_model.zip')

# Test in simulation
tester = SimulationTester(model, n_episodes=50)
metrics = tester.evaluate()
print(f"Success Rate: {metrics['success_rate']:.2%}")
print(f"Avg Return: {metrics['avg_return']:.2f}")
```

### Real Drone Deployment

Deploy to DJI Tello:

```python
from src.real_drone.tello_interface import TelloInterface
from src.real_drone.safe_deploy import SafeDeployer

# Initialize Tello
tello = TelloInterface()
deployer = SafeDeployer(tello, model_path='models/hybrid/best_model.zip')

# Safe deployment with emergency stop
deployer.deploy_safe(max_duration=60, emergency_stop_enabled=True)
```

## Configuration

All hyperparameters are defined in YAML files in the `config/` directory:

### PID Configuration (`config/pid_hover_config.yaml`)

```yaml
pid:
  kp: [0.4, 0.4, 1.0]  # Proportional gains [x, y, z]
  ki: [0.0, 0.0, 0.1]  # Integral gains
  kd: [0.2, 0.2, 0.5]  # Derivative gains
  output_limits: [-1.0, 1.0]

task:
  name: hover
  target_position: [0.0, 0.0, 1.0]
  tolerance: 0.05
  max_episode_steps: 500
```

### RL Configuration (`config/rl_waypoint_config.yaml`)

```yaml
rl:
  algorithm: PPO
  policy: MlpPolicy
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01

environment:
  task: waypoint
  obs_type: kin
  act_type: rpm
  domain_randomization: true
  
domain_randomization:
  mass_range: [0.8, 1.2]  # Multiplier
  inertia_range: [0.8, 1.2]
  motor_noise: 0.05
  wind_disturbance: 0.1
```

### Hybrid Configuration (`config/hybrid_trajectory_config.yaml`)

```yaml
hybrid:
  mode: adaptive  # Options: 'adaptive', 'switching', 'weighted'
  pid_weight: 0.3
  rl_weight: 0.7
  switching_threshold: 0.5
  adaptation_rate: 0.01

task:
  name: trajectory
  trajectory_type: figure_eight
  speed: 0.5
  duration: 30.0
```

## Team Roles and Responsibilities

### Week 1: PID Development & Setup
- **Member 1:** Environment setup, PID implementation
- **Member 2:** PID tuning, hyperparameter optimization
- **Member 3:** Testing framework, evaluation metrics
- **Member 4:** Documentation, visualization tools

### Week 2: RL Training
- **Member 1:** PPO training with domain randomization
- **Member 2:** Hybrid controller architecture
- **Member 3:** Training monitoring and debugging
- **Member 4:** Comparative analysis, ablation studies

### Week 3: Testing & Deployment
- **Member 1:** Simulation testing suite
- **Member 2:** Real Tello drone interface
- **Member 3:** Safe deployment protocols
- **Member 4:** Final report, video demonstrations

## Results

### Expected Performance Metrics

| Task | Success Rate | Avg Return | Position Error (m) |
|------|-------------|------------|-------------------|
| Hover | >95% | >450 | <0.05 |
| Waypoint | >85% | >400 | <0.10 |
| Trajectory | >75% | >350 | <0.15 |

### Trained Models

- `models/ppo/best_hover.zip`: Pure PPO for hovering (500K steps)
- `models/ppo/best_waypoint.zip`: PPO for waypoint navigation (500K steps)
- `models/hybrid/best_model.zip`: Hybrid controller (300K steps)

Models are saved in Stable-Baselines3 format and can be loaded with:

```python
from stable_baselines3 import PPO
model = PPO.load('models/hybrid/best_model.zip')
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_hybrid.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Domain Randomization

The system uses extensive domain randomization for robust sim-to-real transfer:

- **Mass variation:** ±20%
- **Inertia variation:** ±20%
- **Motor noise:** 5% of command
- **Wind disturbance:** Random forces up to 0.1N
- **Delay randomization:** 0-50ms action delay
- **Sensor noise:** Gaussian noise on observations

Enable/disable in config files:

```yaml
domain_randomization:
  enabled: true
  mass_range: [0.8, 1.2]
  inertia_range: [0.8, 1.2]
  motor_noise: 0.05
  wind_disturbance: 0.1
  action_delay: [0, 0.05]
  sensor_noise: 0.01
```

## Real Drone Deployment

### Safety Protocols

1. **Pre-flight Checklist:**
   - Battery >70%
   - Sufficient space (min 3m × 3m)
   - Emergency stop button ready
   - Camera calibration verified

2. **Test Mode:**
   - Start with `--test-mode safe`
   - Limited altitude (max 1m)
   - Reduced speed (50%)
   - Auto-landing on errors

3. **Full Deployment:**
   - Gradually increase complexity
   - Monitor telemetry in real-time
   - Human operator ready to intervene

### Example Deployment

```bash
# Safe test mode
python scripts/deploy_real.py \
    --model models/hybrid/best_model.zip \
    --test-mode safe \
    --max-altitude 1.0 \
    --max-duration 30

# Full deployment
python scripts/deploy_real.py \
    --model models/hybrid/best_model.zip \
    --task waypoint \
    --max-duration 60
```

## Troubleshooting

### Common Issues

**Issue:** PyBullet crashes on import  
**Solution:** Install with `pip install pybullet==3.2.5` (specific version)

**Issue:** Tello connection fails  
**Solution:** 
```bash
# Check WiFi connection to Tello network
# Verify UDP port 8889 is not blocked
python -c "from djitellopy import Tello; t = Tello(); t.connect()"
```

**Issue:** Low success rate in simulation  
**Solution:** 
- Increase training timesteps (try 1M instead of 500K)
- Reduce domain randomization intensity
- Tune reward function weights
- Check PID gains are properly tuned

**Issue:** Sim-to-real gap too large  
**Solution:**
- Increase domain randomization during training
- Add more realistic sensor noise
- Calibrate Tello camera and IMU
- Use system identification to match sim parameters

### Getting Help

- Check `docs/` for detailed guides
- Review training logs in `logs/`
- Open an issue on GitHub
- Contact team members (see Team Roles section)

## Contributing

This is a course project with a fixed team structure. For external contributors:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **gym-pybullet-drones:** Simulation environment ([GitHub](https://github.com/utiasDSL/gym-pybullet-drones))
- **Stable-Baselines3:** RL algorithms ([Docs](https://stable-baselines3.readthedocs.io/))
- **DJI Tello:** Real drone platform
- **Course Instructors:** For project guidance and support

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{drone_hybrid_rl_pid_2025,
  title={Drone Hybrid RL+PID Control System},
  author={Your Team Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/your-username/Task1_Drone_Hybrid_RL_PID}
}
```

## Contact

- **Project Lead:** [Name] - [email]
- **Technical Lead:** [Name] - [email]
- **Repository:** https://github.com/your-username/Task1_Drone_Hybrid_RL_PID

## Timeline

- **Week 1 (Days 1-7):** PID implementation and tuning
- **Week 2 (Days 8-14):** RL training and hybrid development
- **Week 3 (Days 15-21):** Testing, deployment, and documentation

## References

1. Panerati, J., et al. (2021). "Learning to Fly: A Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control"
2. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"
3. Tobin, J., et al. (2017). "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
4. Hwangbo, J., et al. (2019). "Learning Agile and Dynamic Motor Skills for Legged Robots"

---

**Last Updated:** November 2025  
**Version:** 1.0.0  
**Status:** Active Development
