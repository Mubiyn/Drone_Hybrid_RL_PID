# Getting Started - Drone Hybrid RL+PID Project

Complete setup guide for the Drone Hybrid RL+PID Control System project.

---

## Prerequisites

Before starting, ensure you have:
- macOS 14.1+ or Ubuntu 22.04+
- At least 5GB free disk space
- Internet connection for downloads

---

## Step 1: Install Python 3.10 via Conda

The `gym-pybullet-drones` library requires **Python 3.10** specifically.

### Install Miniconda (if not already installed)

**For macOS:**
```bash
# Download Miniconda installer for macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Make it executable
chmod +x Miniconda3-latest-MacOSX-arm64.sh

# Run installer
./Miniconda3-latest-MacOSX-arm64.sh

# Follow prompts, accept license, confirm install location
# When asked "Do you wish to update your shell profile to automatically initialize conda?" → yes

# Restart terminal or run:
source ~/.zshrc  # or ~/.bash_profile
```

**For Ubuntu:**
```bash
# Download Miniconda installer for Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make it executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Run installer
./Miniconda3-latest-Linux-x86_64.sh

# Follow prompts, restart terminal
source ~/.bashrc
```

### Verify Conda Installation
```bash
conda --version
# Should output: conda 24.x.x or similar
```

---

## Step 2: Create Python 3.10 Virtual Environment

Navigate to your project directory and create the environment:

```bash
# Navigate to project root
cd Drone_Hybrid_RL_PID

# Create conda environment with Python 3.10
conda create -n drone-rl-pid python=3.10 -y

# Activate the environment
conda activate drone-rl-pid

# Verify Python version
python --version
# Should output: Python 3.10.x
```

---

## Step 3: Install PyBullet

**IMPORTANT:** PyBullet must be installed via conda-forge BEFORE installing gym-pybullet-drones to avoid compilation errors.

### For macOS (especially Apple Silicon M1/M2/M3):

```bash
# Verify you're in the correct environment
conda activate drone-rl-pid
python --version  # Should show Python 3.10.x

# Install pybullet from conda-forge (has pre-built ARM64 binaries)
conda install -c conda-forge pybullet -y
```

### For Ubuntu/Linux:

```bash
# Verify you're in the correct environment
conda activate drone-rl-pid
python --version  # Should show Python 3.10.x

# Install pybullet from conda-forge (recommended for consistency)
conda install -c conda-forge pybullet -y
```

**Note for Ubuntu users:** If you prefer pip, you can use `pip install pybullet`, but conda-forge is recommended for compatibility.

### Verify PyBullet Installation

```bash
# Test import
python -c "import pybullet; print('PyBullet installed successfully!')"
```

---

## Step 4: Install gym-pybullet-drones

The gym-pybullet-drones library is located in the project directory at `gym-pybullet-drones/`.

### Install in Editable Mode

```bash
# Navigate to gym-pybullet-drones directory (inside project folder)
cd Drone_Hybrid_RL_PID/gym-pybullet-drones

# Upgrade pip
pip install --upgrade pip

# Install gym-pybullet-drones in editable mode
pip install -e .
```

**What this does:**
- Installs gym-pybullet-drones and its dependencies (except pybullet, already installed)
- Upgrades numpy, torch, stable-baselines3 to compatible versions
- Installs in editable mode so changes to the library are reflected immediately

### Verify Installation

```bash
# Test import
python -c "import gym_pybullet_drones; print('gym-pybullet-drones installed successfully!')"
```

---

## Step 5: Install Project Dependencies

Return to your project directory and install all required packages:

```bash
# Navigate back to project root
cd Drone_Hybrid_RL_PID

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

**What this installs:**
- Real drone interface: `djitellopy`, `opencv-python`
- Visualization: `matplotlib`, `seaborn`, `pandas`, `plotly`
- Utilities: `pyyaml`, `scipy`, `tqdm`, `imageio`, `pillow`
- Testing: `pytest`, `pytest-cov`
- Development: `jupyter`, `notebook`, `ipywidgets`
- Experiment tracking: `wandb`, `mlflow`, `tensorboard`
- Code quality: `black`, `flake8`, `mypy`
- **PPO Training (Week 2):** `stable-baselines3[extra]` includes:
  - `pygame` - for rendering and visualization
  - `ale-py` - Atari learning environment (SB3 dependency)
  - `rich` - beautiful terminal output for training progress

**Note:** The following are NOT in requirements.txt (installed separately):
- `pybullet` - installed via conda-forge
- `gym-pybullet-drones` - installed in editable mode
- Core ML libraries (`numpy`, `torch`, `gymnasium`, `stable-baselines3`) - installed with gym-pybullet-drones

---

## Step 6: Verify Complete Installation

Run the installation test script:

```bash
# Still in project root: Drone_Hybrid_RL_PID
python scripts/test_installation.py
```

**Expected Output:**
```
============================================================
Drone Hybrid RL+PID - Installation Test
============================================================

Python version: 3.10.15 | packaged by conda-forge | (main, Oct 16 2024, 01:24:20) [Clang 17.0.6 ]

--- Core Dependencies ---
✓ numpy (version: 2.2.6)
✓ torch (version: 2.9.1)
✓ gymnasium
✓ stable_baselines3

--- Simulation ---
pybullet build time: Oct 21 2025 17:40:50
✓ pybullet
✓ gym-pybullet-drones

--- Real Drone Interface ---
✓ djitellopy
✓ opencv-python

--- Utilities ---
✓ pyyaml
✓ matplotlib
✓ pandas
✓ scipy
✓ tqdm

--- Optional Dependencies ---
✓ wandb
✓ tensorboard
✓ plotly

============================================================
✓ All 13 required checks passed!

You're ready to start training!

Quick start:
  python scripts/train_pid.py
  python scripts/train_rl.py --timesteps 500000
```

---

## Step 7: Test Basic Simulation

Create a simple test script to verify gym-pybullet-drones works:

```bash
# Create a test script
cat > test_sim.py << 'EOF'
"""Test basic gym-pybullet-drones simulation"""
import gymnasium as gym
import gym_pybullet_drones
from gym_pybullet_drones.utils.enums import DroneModel, Physics

print("Creating environment...")
env = gym.make('hover-aviary-v0')

print("Resetting environment...")
obs, info = env.reset()

print("Running 100 random steps...")
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

print("✓ Simulation test passed!")
env.close()
EOF

# Run the test
python test_sim.py
```

**Expected Output:**
```
Creating environment...
pybullet build time: Oct 21 2025 17:40:50
[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:
[INFO] m 0.027000, L 0.039700,
[INFO] ixx 0.000014, iyy 0.000014, izz 0.000022,
...
Resetting environment...
Running 100 random steps...
✓ Simulation test passed!
```

**Note:** You may see some warnings about:
- `pkg_resources is deprecated` - This is a known warning from gym-pybullet-drones, safe to ignore
- `Box precision lowered` and `obs dtype` warnings - These are compatibility warnings between gymnasium and the environment, they don't affect functionality

The important part is seeing `✓ Simulation test passed!` at the end.

---

## Step 8: Review Project Structure

Your project is now set up with the following structure:

```
Drone_Hybrid_RL_PID/
├── README.md                 # Main documentation
├── GETTING_STARTED.md        # This file
├── QUICKSTART.md            # Quick reference
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment (alternative)
├── config/                  # Configuration files
│   ├── pid_hover_config.yaml
│   ├── rl_waypoint_config.yaml
│   ├── hybrid_trajectory_config.yaml
│   └── domain_randomization.yaml
├── src/                     # Source code (to implement)
│   ├── controllers/
│   ├── training/
│   ├── testing/
│   ├── real_drone/
│   └── utils/
├── scripts/                 # Training/testing scripts
│   ├── test_installation.py
│   └── train_pid.py
├── notebooks/              # Jupyter notebooks
├── models/                 # Saved models
├── data/                   # Training data
├── results/                # Results and plots
└── logs/                   # Training logs
```

---

## Step 9: Set Up Your Development Environment

### Option A: VS Code (Recommended)

1. Open VS Code
2. Open the project folder: `File > Open Folder` → select `Drone_Hybrid_RL_PID`
3. Install Python extension (if not already)
4. Select Python interpreter:
   - `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Linux)
   - Type "Python: Select Interpreter"
   - Choose `Python 3.10.x ('drone-rl-pid')`

### Option B: PyCharm

1. Open PyCharm
2. `File > Open` → select `Drone_Hybrid_RL_PID` folder
3. `Preferences > Project > Python Interpreter`
4. Click gear icon → `Add`
5. Select "Conda Environment" → "Existing environment"
6. Choose `/Users/MMD/miniconda3/envs/drone-rl-pid/bin/python`

---

## Step 10: Initialize Git (if not already done)

```bash
cd Drone_Hybrid_RL_PID

# Check if git is initialized
git status

# If not initialized, run:
git init
git add .
git commit -m "Initial project setup with complete environment"

# Link to your GitHub repository
git remote add origin https://github.com/Mubiyn/Drone_Hybrid_RL_PID.git
git push -u origin main
```

---

## Step 11: Run Example PID Controller

Test the gym-pybullet-drones examples:

```bash
# Navigate to gym-pybullet-drones examples (inside project folder)
cd /Drone_Hybrid_RL_PID/gym-pybullet-drones/gym_pybullet_drones/examples/

# Run PID control example
python pid.py
```

You should see a PyBullet simulation window open with a drone hovering using PID control.

---

## Daily Workflow

### Activating Your Environment

Every time you start working, activate the conda environment:

```bash
# Activate environment
conda activate drone-rl-pid

# Navigate to project
cd Drone_Hybrid_RL_PID

# Start working!
```

### Deactivating When Done

```bash
# Deactivate conda environment
conda deactivate
```

---

## Common Commands

### Check Installed Packages
```bash
conda activate drone-rl-pid
conda list  # or pip list
```

### Update a Package
```bash
pip install --upgrade package-name
```

### Add New Dependencies
```bash
# Install new package
pip install new-package

# Update requirements.txt
pip freeze > requirements.txt
```

### Clean PyCache
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

---

## Troubleshooting

### Issue: `conda: command not found`
**Solution:** Restart your terminal or run:
```bash
source ~/.zshrc  # macOS
source ~/.bashrc  # Linux
```

### Issue: PyBullet failed to build wheel (Most Common!)
**Symptoms:** Error when running `pip install -e .` showing "Failed building wheel for pybullet" with clang compiler errors

**Root Cause:** PyBullet is trying to compile from source instead of using pre-built binaries. This happens on Apple Silicon Macs when using pip.

**Solution (RECOMMENDED):**
```bash
# 1. Make sure conda environment is activated
conda activate drone-rl-pid

# 2. Verify Python version
python --version  # MUST show Python 3.10.x

# 3. Install pybullet from conda-forge (has pre-built ARM64 binaries)
conda install -c conda-forge pybullet -y

# 4. Then install gym-pybullet-drones
cd Drone_Hybrid_RL_PID/gym-pybullet-drones
pip install -e .
```

**Why conda-forge?** On Apple Silicon (M1/M2/M3), PyPI doesn't have pre-built wheels for pybullet, so pip tries to compile from source which fails. conda-forge provides pre-compiled binaries.

### Issue: Using system Python instead of conda
**Symptoms:** `python --version` shows 3.9.6 or wrong version
**Solution:** 
```bash
# Activate conda environment first!
conda activate drone-rl-pid

# Verify
python --version  # Should be 3.10.x
which python  # Should point to miniconda3/envs/drone-rl-pid/
```

### Issue: `gym_pybullet_drones` not found
**Solution:** Make sure you installed it in editable mode from the correct location:
```bash
conda activate drone-rl-pid
cd Drone_Hybrid_RL_PID/gym-pybullet-drones
pip install -e .
```

**Note:** The gym-pybullet-drones directory is INSIDE the project folder, not in ``

### Issue: PyBullet build errors on Ubuntu
**Solution:** Install build tools:
```bash
sudo apt install build-essential gcc g++ make
```

### Issue: Python version mismatch
**Solution:** Verify you're using Python 3.10:
```bash
conda activate drone-rl-pid
python --version  # Should be 3.10.x
```

### Issue: Import errors after installation
**Solution:** Reinstall in editable mode:
```bash
cd Drone_Hybrid_RL_PID
pip install -e .
```

---

## Next Steps

Now that your environment is set up, you're ready to start implementing:

1. **Week 1:** PID Controller Implementation
   - Review `Task1_Drone_Hybrid_RL_PID_Guide.md` for detailed instructions
   - Start with `src/controllers/pid_controller.py`
   - Test with `scripts/train_pid.py`

2. **Week 2:** RL Training
   - Implement PPO training
   - Set up domain randomization
   - Train hybrid controller

3. **Week 3:** Testing and Real Drone Deployment
   - Run OOD scenario tests
   - Analyze results
   - Deploy to real hardware

**Important Resources:**
- Implementation guide: `Task1_Drone_Hybrid_RL_PID_Guide.md`
- Quick reference: `QUICKSTART.md`
- Main documentation: `README.md`
- Project structure: `PROJECT_STRUCTURE.md`

---

## Environment Management

### List All Conda Environments
```bash
conda env list
```

### Remove Environment (if needed)
```bash
conda deactivate
conda remove -n drone-rl-pid --all
```

### Export Environment (for team sharing)
```bash
conda activate drone-rl-pid
conda env export > environment_exact.yml
```

### Create from Exported Environment
```bash
conda env create -f environment_exact.yml
```

---

## GPU Support (Optional)

If you have a CUDA-capable GPU and want faster training:

### For Ubuntu with NVIDIA GPU
```bash
conda activate drone-rl-pid

# Check CUDA availability
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is detected
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Summary Checklist

- [x] Conda installed and verified
- [x] Python 3.10 environment created (`drone-rl-pid`)
- [x] PyBullet installed via conda-forge
- [x] gym-pybullet-drones installed from `Drone_Hybrid_RL_PID/gym-pybullet-drones`
- [x] Project dependencies installed from `requirements.txt`
- [x] Installation test script passed (13/13 checks)
- [ ] Basic simulation test successful
- [ ] Git initialized and committed
- [ ] IDE configured with correct Python interpreter
- [ ] Ready to start Week 1 tasks!

## Key Installation Sequence

**Critical:** Follow this exact order to avoid errors:

1. Create conda environment with Python 3.10
2. **Install pybullet via conda-forge** (conda install -c conda-forge pybullet -y)
3. Install gym-pybullet-drones in editable mode (pip install -e .)
4. Install remaining dependencies (pip install -r requirements.txt)
5. Verify installation (python scripts/test_installation.py)

**Common Mistakes to Avoid:**
- ❌ Installing pybullet with pip on Apple Silicon (will fail to compile)
- ❌ Using base conda environment instead of drone-rl-pid
- ❌ Installing gym-pybullet-drones before pybullet
- ❌ Looking for gym-pybullet-drones in wrong directory (it's inside project folder)

---

**Congratulations! Your development environment is fully set up and ready for the project.** 🚀

For any issues, refer to the troubleshooting section or check the official gym-pybullet-drones repository: https://github.com/utiasDSL/gym-pybullet-drones
