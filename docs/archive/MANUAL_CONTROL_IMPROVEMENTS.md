# Manual Data Collection Improvements

## Key Fixes Applied

### 1. Added `initial_position` tracking for MoCap
- Now tracks position relative to takeoff (like autonomous mode)
- Consistent coordinate frames between manual and autonomous data

### 2. Fixed `get_position()` call
- Removed invalid `rigid_body_id` parameter
- MocapWrapper.get_position() takes no arguments

### 3. **CRITICAL**: Actions must match PID output format
Your current manual collection records keyboard inputs as actions:
```python
action = [fb, lr, ud, yaw]  # Keyboard commands
```

But your PID outputs WORLD-FRAME velocities:
```python
action = [vx_world, vy_world, vz_world, yaw_rate]  # PID output
```

**This is a MISMATCH!** For offline RL to work, manual data must have the SAME action format as PID data.

## Recommended Workflow

Since autonomous PID control isn't working reliably, here's the best path forward:

### Option A: Collect Manual Data + Train Behavioral Cloning (SIMPLEST)

1. **Collect manual flights** (current code works fine)
   ```bash
   python src/real_drone/manual_data_collection.py --mocap
   ```

2. **Create behavioral cloning script** instead of offline RL
   - Train neural network to imitate your flying
   - Simpler than offline RL (no rewards needed)
   - Works with inconsistent action formats
   
3. **File**: `scripts/train_behavioral_cloning.py` (NEEDS CREATION)
   ```python
   # Load manual data
   # Train: obs → action (supervised learning)
   # No PID needed!
   ```

### Option B: Fix Manual Collection to Match PID Format

1. **During manual flight**, estimate what PID WOULD output:
   ```python
   # Get current state
   state = get_state()
   current_pos = state[0:3]
   
   # Where do you WANT to go? (user's intent)
   # This requires tracking a mental "target position"
   target_pos = current_pos + some_offset_based_on_keyboard
   
   # What would PID output?
   action_pid_style = kp * (target_pos - current_pos)  # World frame velocity
   ```

2. **Problem**: Hard to know user's target position from keyboard alone

### Option C: Hybrid - Manual Takeoff/Land, Auto Collection

Use manual control for safety (takeoff/land/emergency), but autonomous PID for data:
```python
# Manual takeoff
# Press 'A' → Start autonomous circle with PID
# Collect PID data (already working from autonomous script)
# Manual land
```

## RECOMMENDATION: Option A (Behavioral Cloning)

**Why**:
- Manual flying works NOW
- No coordinate frame issues
- Simpler training (supervised learning)
- Can collect data TODAY

**Steps**:
1. Fly manually for 5-10 minutes (multiple trajectories)
2. Create `scripts/train_bc.py` - train network to copy your actions
3. Test learned policy

**I can create the behavioral cloning script for you if you want this approach.**

