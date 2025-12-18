# Installation Guide

Complete installation instructions for the Drone Hybrid RL-PID project.

---

## Prerequisites

### Required
- **Python**: 3.10 or higher
- **Operating System**: macOS, Linux, or Windows with WSL2
- **Disk Space**: ~5GB for environment and models

### Optional
- **CUDA**: 11.7+ for GPU-accelerated training (faster but not required)
- **DJI Tello Drone**: For Phase 2 hardware deployment
- **Motion Capture System**: OptiTrack for advanced tracking (optional)

---

## Installation Methods

### Option 1: Conda (Recommended)

Conda automatically handles package dependencies and is more reliable across platforms.

```bash
# 1. Clone repository
git clone https://github.com/Mubiyn/Drone_Hybrid_RL_PID.git
cd Drone_Hybrid_RL_PID

# 2. Create conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate drone-hybrid-rl

# 4. Install gym-pybullet-drones
cd gym-pybullet-drones
pip install -e .
cd ..

# 5. Verify installation
python scripts/test_installation.py
```

### Option 2: Pip with Virtual Environment

```bash
# 1. Clone repository
git clone https://github.com/Mubiyn/Drone_Hybrid_RL_PID.git
cd Drone_Hybrid_RL_PID

# 2. Create virtual environment
python3.10 -m venv venv

# 3. Activate environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Install gym-pybullet-drones
cd gym-pybullet-drones
pip install -e .
cd ..

# 6. Verify installation
python scripts/test_installation.py
```

---

## Platform-Specific Notes

### Apple Silicon (M1/M2/M3)

PyBullet may have issues with pip on Apple Silicon. Use conda instead:

```bash
# Install PyBullet via conda
conda install -c conda-forge pybullet

# Then proceed with normal installation
cd gym-pybullet-drones && pip install -e .
```

### Windows WSL2

Ensure you're using WSL2 (not WSL1) for OpenGL support:

```bash
# Check WSL version
wsl -l -v

# If WSL1, upgrade to WSL2
wsl --set-version Ubuntu 2
```

### Linux

Install system dependencies first:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip

# NVIDIA GPU support (optional)
sudo apt-get install -y nvidia-cuda-toolkit
```

---

## Verification

### Test Installation

```bash
python scripts/test_installation.py
```

Expected output:
```
✓ Python version: 3.10.x
✓ PyBullet: 3.2.x
✓ Stable-Baselines3: 2.x.x
✓ Gym: 0.21.0
✓ NumPy: 1.24.x
✓ PyTorch: 2.x.x
✓ gym-pybullet-drones installed
All checks passed!
```

### Test Simulation

```bash
# Run quick simulation test (30 seconds)
python src/testing/test_rl.py --task hover --duration 30
```

You should see a PyBullet window with a drone hovering.

### Test RL Model Loading

```bash
# Load and test a pre-trained model
python src/testing/test_rl.py --task circle
```

---

## Troubleshooting

### Issue: PyBullet Import Error

**Error**: `ModuleNotFoundError: No module named 'pybullet'`

**Solution**:
```bash
# Try conda installation
conda install -c conda-forge pybullet

# OR force reinstall with pip
pip install --force-reinstall pybullet
```

### Issue: Gym Version Conflict

**Error**: `gym 0.26.0 requires gym-notices>=0.0.4, but you have gym-notices 0.0.2`

**Solution**:
```bash
pip install gym==0.21.0 --force-reinstall
```

### Issue: CUDA Not Found (GPU Training)

**Error**: `CUDA not available`

**Note**: This is not critical - training will use CPU (slower but works).

**Solution** (for GPU acceleration):
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### Issue: gym-pybullet-drones Import Error

**Error**: `ModuleNotFoundError: No module named 'gym_pybullet_drones'`

**Solution**:
```bash
cd gym-pybullet-drones
pip install -e .
cd ..
```

### Issue: Permission Denied (Linux/macOS)

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
# Don't use sudo with pip in virtual environment
# Instead, ensure you're in the virtual environment:
conda activate drone-hybrid-rl  # OR
source venv/bin/activate
```

---

## Docker Installation (Alternative)

For a completely isolated environment:

```bash
# Build Docker image
docker build -t drone-hybrid-rl .

# Run container
docker run -it --rm \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/models:/workspace/models \
  drone-hybrid-rl

# Run tests inside container
python scripts/test_installation.py
```

See [Docker Guide](07_docker_guide.md) for details.

---

## Hardware Setup (Phase 2 Only)

### DJI Tello Drone

1. **Connect to Tello WiFi**:
   - Turn on Tello drone
   - Connect computer to `TELLO-XXXXXX` WiFi network

2. **Install Tello SDK**:
   ```bash
   pip install djitellopy
   ```

3. **Test connection**:
   ```bash
   python scripts/hardware/test_tello_setup.py
   ```

4. **Verify battery** (>50% recommended):
   ```bash
   python -c "from src.hardware.TelloWrapper import TelloWrapper; t = TelloWrapper(); print(f'Battery: {t.get_battery()}%')"
   ```

See [Hardware Setup](06_hardware_setup.md) for complete guide.

---

## Next Steps

After successful installation:

1. **[Getting Started](01_getting_started.md)** - Run your first experiment
2. **[Methodology](03_methodology.md)** - Understand the approach
3. **[Results](05_results.md)** - Explore performance metrics

---

## Dependencies Summary

### Core Dependencies
- `python >=3.10`
- `numpy >=1.24`
- `torch >=2.0`
- `pybullet >=3.2.5`
- `stable-baselines3 >=2.0`
- `gym ==0.21.0`

### Hardware (Optional)
- `djitellopy` - DJI Tello control

### Visualization
- `matplotlib >=3.7`
- `seaborn >=0.12`

### Full List
See [requirements.txt](../requirements.txt) or [environment.yml](../environment.yml)

---

**Having issues?** Check [System Architecture](04_architecture.md) for technical details or open an [issue](https://github.com/Mubiyn/Drone_Hybrid_RL_PID/issues).
