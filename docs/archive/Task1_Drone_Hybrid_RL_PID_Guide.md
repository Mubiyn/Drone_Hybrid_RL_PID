# Task 1: Hybrid RL and Residual PID for Quadrotor Trajectory Tracking

## Project Timeline: 3 Weeks (4 Team Members)

---

##  Project Overview

**Goal**: Compare the robustness of three drone control approaches when facing out-of-distribution dynamics:
1. Classical PID Controller
2. Pure Reinforcement Learning (PPO)
3. Hybrid PID + RL Residual Controller (SOTA)

**Key Question**: How do these controllers perform under conditions they weren't trained for (mass changes, wind, damaged motors)?

**Expected Outcome**: Demonstrate that hybrid controllers offer the best balance of stability and adaptability.

---

##  Success Metrics

### Quantitative Metrics
- **Trajectory Tracking Error**: RMSE between desired and actual position
- **Settling Time**: Time to reach waypoint (within 5cm tolerance)
- **Control Effort**: Average motor thrust % and saturation events
- **Success Rate**: % of trajectories completed without crashes
- **Robustness Score**: Performance degradation under OOD conditions

### Qualitative Metrics
- Smooth vs jerky control behavior
- Recovery time after disturbances
- Visual quality of trajectory tracking

---

##  Team Structure & Responsibilities

### **Member 1: Infrastructure Lead**
- Environment setup and simulation infrastructure
- Coordinate tool installation across team
- Hardware interface for real drone
- Documentation of setup procedures

### **Member 2: Classical Control Specialist**
- PID controller implementation and tuning
- Baseline performance evaluation
- Mathematical modeling of drone dynamics
- Domain randomization implementation

### **Member 3: RL Training Lead**
- Pure PPO controller implementation
- Hybrid PID+RL controller implementation
- Training pipeline setup and monitoring
- Hyperparameter tuning

### **Member 4: Testing & Analysis Lead**
- Experimental design for OOD scenarios
- Data collection and metrics computation
- Visualization and plotting
- Real drone testing coordination

**Note**: Everyone participates in real drone testing for safety and learning.

---

##  Week-by-Week Breakdown

---

## **WEEK 1: Setup, PID, and Training Infrastructure**

### **Day 1-2: Environment Setup (ALL MEMBERS)**

#### **Member 1 Tasks** (Lead)
1. **Install gym-pybullet-drones**
   ```bash
   git clone https://github.com/utiasDSL/gym-pybullet-drones.git
   cd gym-pybullet-drones
   pip install -e .
   ```

2. **Test basic simulation**
   ```python
   # test_simulation.py
   import gymnasium as gym
   import gym_pybullet_drones
   
   env = gym.make('ctrl-aviary-v0')
   obs = env.reset()
   for i in range(1000):
       obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
       env.render()
   env.close()
   ```

3. **Set up shared code repository**
   - Initialize Git repo
   - Create folder structure:
     ```
     project/
     ├── controllers/
     │   ├── pid_controller.py
     │   ├── ppo_controller.py
     │   └── hybrid_controller.py
     ├── training/
     │   ├── train_ppo.py
     │   └── train_hybrid.py
     ├── testing/
     │   ├── test_scenarios.py
     │   └── metrics.py
     ├── configs/
     │   └── default_config.yaml
     ├── results/
     └── real_drone/
     ```

4. **Hardware inventory & testing**
   - Document available drones (model, sensors)
   - Check battery status and chargers
   - Test communication (RC controller, WiFi)
   - Verify motion capture system or GPS availability

#### **Member 2 Tasks**
1. **Install Stable-Baselines3**
   ```bash
   pip install stable-baselines3[extra]
   pip install tensorboard
   ```

2. **Study drone dynamics**
   - Review quadrotor equations of motion
   - Understand thrust-to-weight ratios
   - Document state space (position, velocity, orientation, angular velocity)

3. **Create configuration file**
   ```yaml
   # configs/default_config.yaml
   drone:
     mass: 0.027  # kg
     arm_length: 0.0397  # m
     thrust_to_weight: 2.0
     max_rpm: 35000
   
   pid:
     pos_kp: [0.4, 0.4, 1.25]
     pos_ki: [0.05, 0.05, 0.05]
     pos_kd: [0.2, 0.2, 0.5]
     att_kp: [70, 70, 60]
     att_ki: [0.0, 0.0, 500]
     att_kd: [20, 20, 12]
   
   training:
     total_timesteps: 5000000
     learning_rate: 0.0003
     n_steps: 2048
     batch_size: 64
   ```

#### **Member 3 Tasks**
1. **Set up training monitoring**
   ```bash
   # Create TensorBoard logging
   mkdir -p logs/ppo logs/hybrid
   ```

2. **Study PPO algorithm**
   - Review Stable-Baselines3 PPO documentation
   - Understand policy networks, value functions
   - Read about Domain Randomization techniques

3. **Create training template**
   ```python
   # training/train_template.py
   from stable_baselines3 import PPO
   from stable_baselines3.common.vec_env import SubprocVecEnv
   import gymnasium as gym
   
   def make_env():
       def _init():
           env = gym.make('ctrl-aviary-v0')
           return env
       return _init
   
   # Create parallel environments for faster training
   n_envs = 8
   env = SubprocVecEnv([make_env() for _ in range(n_envs)])
   
   model = PPO(
       "MlpPolicy",
       env,
       learning_rate=3e-4,
       n_steps=2048,
       batch_size=64,
       n_epochs=10,
       verbose=1,
       tensorboard_log="./logs/ppo/"
   )
   ```

#### **Member 4 Tasks**
1. **Design test scenarios**
   ```python
   # testing/test_scenarios.py
   
   OOD_SCENARIOS = {
       'nominal': {
           'mass_multiplier': 1.0,
           'motor_efficiency': [1.0, 1.0, 1.0, 1.0],
           'wind_speed': 0.0
       },
       'heavy_payload': {
           'mass_multiplier': 1.2,  # +20% mass
           'motor_efficiency': [1.0, 1.0, 1.0, 1.0],
           'wind_speed': 0.0
       },
       'light_payload': {
           'mass_multiplier': 0.8,  # -20% mass
           'motor_efficiency': [1.0, 1.0, 1.0, 1.0],
           'wind_speed': 0.0
       },
       'damaged_motor': {
           'mass_multiplier': 1.0,
           'motor_efficiency': [1.0, 1.0, 0.7, 1.0],  # Motor 3 at 70%
           'wind_speed': 0.0
       },
       'strong_wind': {
           'mass_multiplier': 1.0,
           'motor_efficiency': [1.0, 1.0, 1.0, 1.0],
           'wind_speed': 2.0  # m/s
       },
       'combined_worst': {
           'mass_multiplier': 1.2,
           'motor_efficiency': [1.0, 1.0, 0.7, 1.0],
           'wind_speed': 1.5
       }
   }
   ```

2. **Create metrics calculation framework**
   ```python
   # testing/metrics.py
   import numpy as np
   
   def compute_rmse(desired, actual):
       """Compute Root Mean Square Error"""
       return np.sqrt(np.mean((desired - actual)**2))
   
   def compute_settling_time(positions, target, threshold=0.05):
       """Time to reach and stay within threshold of target"""
       errors = np.linalg.norm(positions - target, axis=1)
       # Find first time error goes below threshold and stays there
       pass
   
   def compute_control_effort(actions):
       """Average and peak control usage"""
       return {
           'mean': np.mean(np.abs(actions)),
           'max': np.max(np.abs(actions)),
           'saturation_pct': np.mean(np.abs(actions) > 0.95) * 100
       }
   ```

---

### **Day 3-5: PID Implementation (Member 2 Lead, All Support)**

#### **Member 2 Tasks** (Primary)
1. **Implement PID controller**
   ```python
   # controllers/pid_controller.py
   import numpy as np
   
   class PIDController:
       def __init__(self, config):
           # Position PIDs (3 separate PIDs for x, y, z)
           self.pos_kp = np.array(config['pid']['pos_kp'])
           self.pos_ki = np.array(config['pid']['pos_ki'])
           self.pos_kd = np.array(config['pid']['pos_kd'])
           
           # Attitude PIDs (roll, pitch, yaw)
           self.att_kp = np.array(config['pid']['att_kp'])
           self.att_ki = np.array(config['pid']['att_ki'])
           self.att_kd = np.array(config['pid']['att_kd'])
           
           # Error accumulation
           self.pos_integral = np.zeros(3)
           self.att_integral = np.zeros(3)
           self.prev_pos_error = np.zeros(3)
           self.prev_att_error = np.zeros(3)
           
           self.dt = 1/240.0  # Simulation timestep
       
       def compute_control(self, state, target_pos, target_vel=None, target_yaw=0):
           """
           Args:
               state: dict with 'pos', 'vel', 'rpy', 'ang_vel'
               target_pos: desired [x, y, z]
               target_vel: desired velocity (optional)
               target_yaw: desired yaw angle
           Returns:
               motor_rpms: [rpm1, rpm2, rpm3, rpm4]
           """
           # Position control (outer loop)
           pos_error = target_pos - state['pos']
           self.pos_integral += pos_error * self.dt
           pos_derivative = (pos_error - self.prev_pos_error) / self.dt
           
           # PID output: desired acceleration
           des_acc = (self.pos_kp * pos_error + 
                     self.pos_ki * self.pos_integral +
                     self.pos_kd * pos_derivative)
           
           if target_vel is not None:
               des_acc += self.pos_kd * (target_vel - state['vel'])
           
           # Convert desired acceleration to desired attitude
           des_thrust = (des_acc[2] + 9.81) * self.mass
           des_roll = np.arcsin(-des_acc[1] / 9.81)
           des_pitch = np.arcsin(des_acc[0] / 9.81)
           
           des_attitude = np.array([des_roll, des_pitch, target_yaw])
           
           # Attitude control (inner loop)
           att_error = des_attitude - state['rpy']
           self.att_integral += att_error * self.dt
           att_derivative = (att_error - self.prev_att_error) / self.dt
           
           des_torques = (self.att_kp * att_error +
                         self.att_ki * self.att_integral +
                         self.att_kd * att_derivative)
           
           # Convert thrust + torques to motor RPMs
           motor_rpms = self._thrust_torques_to_rpms(des_thrust, des_torques)
           
           # Update previous errors
           self.prev_pos_error = pos_error
           self.prev_att_error = att_error
           
           return motor_rpms
       
       def _thrust_torques_to_rpms(self, thrust, torques):
           """Convert desired thrust and torques to motor RPMs"""
           # This uses the quadrotor mixing matrix
           # Implementation specific to drone configuration
           pass
       
       def reset(self):
           """Reset integral terms"""
           self.pos_integral = np.zeros(3)
           self.att_integral = np.zeros(3)
           self.prev_pos_error = np.zeros(3)
           self.prev_att_error = np.zeros(3)
   ```

2. **Manual tuning procedure**
   - Start with Kp only (Ki=Kd=0)
   - Increase Kp until oscillations appear
   - Add Kd to dampen oscillations
   - Add small Ki to eliminate steady-state error
   - Tune position loop first, then attitude loop

3. **Create tuning visualization script**
   ```python
   # Test PID response to step input
   import matplotlib.pyplot as plt
   
   def plot_step_response(controller, target_pos):
       """Plot position vs time for step response"""
       # Useful for visualizing overshoot, settling time
       pass
   ```

#### **Member 1 Tasks** (Support)
- Create wrapper for easy controller testing
- Set up recording of trajectories for analysis
- Help debug PID numerical issues (integral windup, derivative kick)

#### **Member 3 Tasks** (Support)
- Study PID theory to understand hyperparameters
- Help with testing different PID gains
- Document best PID gains found

#### **Member 4 Tasks** (Support)
- Run systematic tests of different PID gains
- Record performance metrics for each configuration
- Create comparison plots

**Deliverable**: Working PID controller with documented gains that can hover and track waypoints.

---

### **Day 6-7: Domain Randomization Setup (Member 2 Lead)**

#### **Member 2 Tasks**
1. **Implement domain randomization wrapper**
   ```python
   # training/domain_randomization.py
   import gymnasium as gym
   import numpy as np
   
   class DomainRandomizationWrapper(gym.Wrapper):
       def __init__(self, env, randomize_params):
           super().__init__(env)
           self.randomize_params = randomize_params
       
       def reset(self, **kwargs):
           # Randomize physics parameters at each episode start
           if self.randomize_params['mass']:
               mass_range = (0.024, 0.030)  # ±10% of nominal 0.027kg
               new_mass = np.random.uniform(*mass_range)
               self.env.set_mass(new_mass)
           
           if self.randomize_params['inertia']:
               inertia_mult = np.random.uniform(0.9, 1.1, size=3)
               self.env.set_inertia(self.env.nominal_inertia * inertia_mult)
           
           if self.randomize_params['motor_constants']:
               motor_mult = np.random.uniform(0.95, 1.05, size=4)
               self.env.set_motor_constants(motor_mult)
           
           if self.randomize_params['drag']:
               drag_mult = np.random.uniform(0.8, 1.2)
               self.env.set_drag_coefficient(drag_mult)
           
           if self.randomize_params['wind']:
               wind_vel = np.random.uniform(-0.5, 0.5, size=3)
               self.env.set_wind(wind_vel)
           
           return self.env.reset(**kwargs)
   ```

2. **Test randomization effects**
   - Verify parameters are actually changing
   - Ensure changes are realistic (drone still flyable)
   - Document range of randomization

#### **All Members**
- Discuss and agree on randomization ranges
- Test that PID still works under randomization
- Decide on DR intensity (conservative vs aggressive)

---

## **WEEK 2: RL Training and Hybrid Implementation**

### **Day 8-10: Pure PPO Training (Member 3 Lead)**

#### **Member 3 Tasks** (Primary)
1. **Create custom reward function**
   ```python
   # training/reward_functions.py
   import numpy as np
   
   def compute_reward(state, action, target_pos, target_vel=None):
       """
       Reward function for trajectory tracking
       """
       # Position error penalty
       pos_error = np.linalg.norm(state['pos'] - target_pos)
       pos_reward = -pos_error * 10.0
       
       # Velocity error penalty (if target velocity provided)
       if target_vel is not None:
           vel_error = np.linalg.norm(state['vel'] - target_vel)
           vel_reward = -vel_error * 2.0
       else:
           vel_reward = 0
       
       # Orientation penalty (want to stay level)
       rpy = state['rpy']
       orientation_penalty = -(abs(rpy[0]) + abs(rpy[1])) * 5.0
       
       # Control effort penalty (discourage aggressive control)
       control_penalty = -np.sum(np.abs(action)) * 0.01
       
       # Survival bonus
       survival_bonus = 0.1
       
       # Termination penalties
       if pos_error > 2.0:  # Too far from target
           return -100.0
       if abs(rpy[0]) > 0.5 or abs(rpy[1]) > 0.5:  # Tipped over
           return -100.0
       
       total_reward = (pos_reward + vel_reward + 
                      orientation_penalty + control_penalty + 
                      survival_bonus)
       
       return total_reward
   ```

2. **Implement training script**
   ```python
   # training/train_ppo.py
   from stable_baselines3 import PPO
   from stable_baselines3.common.vec_env import SubprocVecEnv
   from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
   import gymnasium as gym
   from domain_randomization import DomainRandomizationWrapper
   
   def make_train_env(rank):
       def _init():
           env = gym.make('ctrl-aviary-v0')
           env = DomainRandomizationWrapper(env, randomize_params={
               'mass': True,
               'inertia': True,
               'motor_constants': True,
               'drag': True,
               'wind': True
           })
           return env
       return _init
   
   # Create parallel environments
   n_envs = 8
   train_env = SubprocVecEnv([make_train_env(i) for i in range(n_envs)])
   
   # Callbacks for saving checkpoints
   checkpoint_callback = CheckpointCallback(
       save_freq=100000,
       save_path='./models/ppo/',
       name_prefix='ppo_drone'
   )
   
   # Create PPO model
   model = PPO(
       "MlpPolicy",
       train_env,
       learning_rate=3e-4,
       n_steps=2048,
       batch_size=64,
       n_epochs=10,
       gamma=0.99,
       gae_lambda=0.95,
       clip_range=0.2,
       ent_coef=0.0,
       verbose=1,
       tensorboard_log="./logs/ppo/"
   )
   
   # Train
   model.learn(
       total_timesteps=5000000,
       callback=checkpoint_callback
   )
   
   model.save("models/ppo_final")
   ```

3. **Monitor training**
   ```bash
   # Launch TensorBoard
   tensorboard --logdir=./logs/ppo/
   ```
   - Watch episode reward curve
   - Check for learning (reward should increase)
   - Monitor policy loss, value loss
   - Check for catastrophic forgetting

4. **Train multiple seeds**
   - Run training 3 times with different random seeds
   - This helps assess variance in final performance

#### **Member 1 Tasks** (Support)
- Set up GPU monitoring (nvidia-smi)
- Ensure training doesn't crash overnight
- Create backup system for checkpoints

#### **Member 2 Tasks** (Support)
- Compare PPO behavior to PID during training
- Identify when PPO becomes "stable enough"
- Help tune reward function weights

#### **Member 4 Tasks** (Support)
- Evaluate checkpoints periodically
- Plot learning curves
- Document training hyperparameters and outcomes

**Deliverable**: Trained PPO model that can hover and track trajectories (may be unstable under OOD).

---

### **Day 11-14: Hybrid Controller (Member 3 Lead, Member 2 Support)**

#### **Member 3 Tasks** (Primary)
1. **Implement hybrid architecture**
   ```python
   # controllers/hybrid_controller.py
   from pid_controller import PIDController
   import numpy as np
   
   class HybridController:
       def __init__(self, pid_config, rl_model_path):
           self.pid = PIDController(pid_config)
           self.rl_model = load_model(rl_model_path)  # Trained RL policy
       
       def compute_control(self, state, target_pos, target_vel=None, target_yaw=0):
           """
           Hybrid control: u_total = u_PID + u_RL_residual
           """
           # Get PID base control
           u_pid = self.pid.compute_control(state, target_pos, target_vel, target_yaw)
           
           # RL observes state + PID action
           rl_obs = self._construct_rl_observation(state, target_pos, u_pid)
           
           # RL outputs residual correction
           u_residual, _ = self.rl_model.predict(rl_obs, deterministic=True)
           
           # Combine
           u_total = u_pid + u_residual
           
           # Clip to valid motor range
           u_total = np.clip(u_total, 0, MAX_RPM)
           
           return u_total
       
       def _construct_rl_observation(self, state, target, pid_action):
           """
           RL needs to see:
           - Current state
           - Target
           - What PID is trying to do
           """
           obs = np.concatenate([
               state['pos'],
               state['vel'],
               state['rpy'],
               state['ang_vel'],
               target,
               pid_action
           ])
           return obs
   ```

2. **Create hybrid training environment**
   ```python
   # training/train_hybrid.py
   
   class HybridTrainingEnv(gym.Env):
       """
       Custom environment where RL learns residual on top of PID
       """
       def __init__(self, base_env, pid_controller):
           self.base_env = base_env
           self.pid = pid_controller
           
           # RL action space: residual corrections (smaller than full control)
           self.action_space = gym.spaces.Box(
               low=-0.3, high=0.3, shape=(4,), dtype=np.float32
           )
           
           # Observation includes PID's intended action
           self.observation_space = gym.spaces.Box(
               low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
           )
       
       def step(self, rl_residual):
           state = self.get_state()
           target = self.get_current_target()
           
           # PID computes base control
           pid_action = self.pid.compute_control(state, target)
           
           # Combine with RL residual
           combined_action = pid_action + rl_residual
           combined_action = np.clip(combined_action, 0, MAX_RPM)
           
           # Execute in environment
           next_state, reward, done, info = self.base_env.step(combined_action)
           
           # Augment observation with PID action
           obs = self._make_observation(next_state, pid_action)
           
           return obs, reward, done, info
       
       def _make_observation(self, state, pid_action):
           return np.concatenate([
               state['pos'], state['vel'], state['rpy'], 
               state['ang_vel'], pid_action
           ])
   ```

3. **Train hybrid controller**
   ```python
   # Training hybrid is similar to pure PPO but:
   # 1. Action space is smaller (just residuals)
   # 2. Observation includes PID actions
   # 3. Should converge faster (starting from PID baseline)
   
   hybrid_env = SubprocVecEnv([make_hybrid_env(i) for i in range(8)])
   
   model = PPO(
       "MlpPolicy",
       hybrid_env,
       learning_rate=3e-4,
       n_steps=2048,
       # ... same hyperparameters as pure PPO
   )
   
   model.learn(total_timesteps=2000000)  # Shorter training!
   model.save("models/hybrid_final")
   ```

#### **Member 2 Tasks** (Support)
- Ensure PID is properly integrated
- Debug issues with PID+RL combination
- Verify that residuals are reasonable (not too large)

#### **Member 1 Tasks** (Support)
- Monitor hybrid training
- Compare training speed to pure PPO

#### **Member 4 Tasks** (Support)
- Test intermediate hybrid models
- Visualize what residuals look like
- Check if hybrid outperforms both PID and PPO

**Deliverable**: Trained hybrid controller that combines PID stability with RL adaptability.

---

## **WEEK 3: Testing, Analysis, and Real Drone Deployment**

### **Day 15-17: Comprehensive Testing (Member 4 Lead, All Participate)**

#### **Member 4 Tasks** (Primary)
1. **Automated testing framework**
   ```python
   # testing/run_experiments.py
   import json
   from test_scenarios import OOD_SCENARIOS
   
   def test_controller(controller_name, controller, scenarios, n_trials=5):
       """
       Test a controller across multiple scenarios
       """
       results = {}
       
       for scenario_name, params in scenarios.items():
           print(f"Testing {controller_name} on {scenario_name}...")
           
           scenario_results = []
           for trial in range(n_trials):
               # Set environment parameters
               env.set_physics_params(params)
               
               # Run episode
               metrics = run_episode(env, controller)
               scenario_results.append(metrics)
           
           # Aggregate across trials
           results[scenario_name] = {
               'rmse_mean': np.mean([m['rmse'] for m in scenario_results]),
               'rmse_std': np.std([m['rmse'] for m in scenario_results]),
               'success_rate': np.mean([m['success'] for m in scenario_results]),
               'settling_time': np.mean([m['settling_time'] for m in scenario_results])
           }
       
       return results
   
   # Test all three controllers
   pid_results = test_controller("PID", pid_controller, OOD_SCENARIOS)
   ppo_results = test_controller("PPO", ppo_controller, OOD_SCENARIOS)
   hybrid_results = test_controller("Hybrid", hybrid_controller, OOD_SCENARIOS)
   
   # Save results
   with open('results/experiment_results.json', 'w') as f:
       json.dump({
           'PID': pid_results,
           'PPO': ppo_results,
           'Hybrid': hybrid_results
       }, f, indent=2)
   ```

2. **Run experiments**
   - Test each controller on all 6 OOD scenarios
   - 5 trials per scenario (30 trials per controller)
   - Total: 90 trials × 3 controllers = 270 trials
   - Record videos of interesting cases (successes and failures)

#### **Member 1, 2, 3 Tasks** (Support)
- Help run experiments in parallel
- Monitor for crashes or bugs
- Take notes on qualitative observations

#### **All Members**
- Daily stand-up: share observations
- Discuss unexpected behaviors
- Decide if any re-training is needed

**Deliverable**: Complete dataset of performance metrics across all scenarios.

---

### **Day 18-19: Data Analysis and Visualization (Member 4 Lead)**

#### **Member 4 Tasks** (Primary)
1. **Create comparison plots**
   ```python
   # analysis/plot_results.py
   import matplotlib.pyplot as plt
   import seaborn as sns
   import json
   
   # Load results
   with open('results/experiment_results.json', 'r') as f:
       results = json.load(f)
   
   # Plot 1: Tracking error across scenarios
   scenarios = list(results['PID'].keys())
   pid_errors = [results['PID'][s]['rmse_mean'] for s in scenarios]
   ppo_errors = [results['PPO'][s]['rmse_mean'] for s in scenarios]
   hybrid_errors = [results['Hybrid'][s]['rmse_mean'] for s in scenarios]
   
   x = np.arange(len(scenarios))
   width = 0.25
   
   fig, ax = plt.subplots(figsize=(12, 6))
   ax.bar(x - width, pid_errors, width, label='PID', color='blue')
   ax.bar(x, ppo_errors, width, label='PPO', color='orange')
   ax.bar(x + width, hybrid_errors, width, label='Hybrid', color='green')
   
   ax.set_ylabel('RMSE (m)')
   ax.set_title('Tracking Error Across OOD Scenarios')
   ax.set_xticks(x)
   ax.set_xticklabels(scenarios, rotation=45)
   ax.legend()
   plt.tight_layout()
   plt.savefig('results/tracking_error_comparison.png')
   
   # Plot 2: Success rate heatmap
   # Plot 3: Control effort comparison
   # Plot 4: Robustness degradation (performance vs. disturbance magnitude)
   ```

2. **Statistical analysis**
   - T-tests between controllers
   - Compute effect sizes
   - Identify statistically significant differences

3. **Create summary tables**
   - Best controller per scenario
   - Overall robustness ranking
   - Trade-offs analysis

#### **Member 1, 2, 3 Tasks**
- Review plots and suggest improvements
- Help interpret results
- Draft findings section

**Deliverable**: Complete set of figures and statistical analysis.

---

### **Day 20-21: Real Drone Testing (ALL MEMBERS)**

#### **Safety Protocol (Member 1 Lead)**
1. **Pre-flight checks**
   - Battery voltage > 11.1V
   - All propellers secure and undamaged
   - RC controller failsafe configured
   - Geofencing enabled (max altitude, radius)
   - Emergency stop tested

2. **Testing environment**
   - Indoor flight area with safety nets
   - Motion capture system calibrated (if available)
   - Clear of obstacles and people

3. **Progressive testing**
   - Day 1: PID controller only (baseline, ensure hardware works)
   - Day 2: PPO and Hybrid controllers

#### **Testing Procedure**
**Test 1: Hover Stability**
- Command: hover at [0, 0, 1] for 30 seconds
- Measure: position variance
- Controllers: PID → PPO → Hybrid

**Test 2: Waypoint Navigation**
- Command: visit waypoints [1,0,1] → [1,1,1] → [0,1,1] → [0,0,1]
- Measure: tracking error, settling time
- Controllers: PID → PPO → Hybrid

**Test 3: Payload Disturbance** (if safe)
- Add small payload (+20g)
- Repeat hover and waypoint tests
- This simulates "heavy_payload" OOD scenario

**Test 4: Wind Disturbance** (if feasible)
- Use fans to create wind
- Test hover stability
- Observe recovery behavior

#### **Data Collection**
- Record flight logs (onboard sensors)
- Capture video (external cameras)
- Motion capture data (if available)
- Manual observations (smoothness, aggressiveness)

#### **All Members Roles**
- **Member 1**: Pilot (RC controller, safety override)
- **Member 2**: Data logging (computer, sensors)
- **Member 3**: Video recording (camera operator)
- **Member 4**: Note-taking (observations, timing)

**Deliverable**: Real-world performance data and comparison videos.

---

##  Final Deliverables (Day 21)

### **1. Code Repository**
- All controller implementations
- Training scripts with configs used
- Testing framework
- Analysis scripts

### **2. Experimental Results**
- `experiment_results.json`: Raw data
- Figures: comparison plots (8-10 figures)
- Videos: simulation + real drone tests
- Statistical analysis summary

### **3. Written Report** (8-12 pages)
**Structure:**
1. **Introduction** (Member 2)
   - Problem statement
   - Why hybrid controllers?
   - Research question

2. **Background** (Member 1)
   - PID control theory
   - Reinforcement learning (PPO)
   - Domain randomization
   - Related work

3. **Methods** (Member 3)
   - Experimental design
   - Controller implementations
   - Training procedures
   - OOD scenarios

4. **Results** (Member 4)
   - Simulation results (plots)
   - Real drone results
   - Statistical comparisons

5. **Discussion** (All)
   - Interpretation of results
   - Why hybrid wins (or doesn't)
   - Limitations
   - Sim-to-real gap analysis

6. **Conclusion** (Member 2)
   - Key findings
   - Future work

### **4. Presentation** (15-20 minutes)
- Slides: 12-15 slides
- Live demo video (2-3 minutes)
- Q&A preparation

---

## 🔧 Troubleshooting Guide

### **Issue: PID is unstable**
- **Symptoms**: Oscillations, overshooting
- **Solutions**: 
  - Reduce Kp gains
  - Increase Kd (damping)
  - Check for derivative kick (use filtered derivative)
  - Verify timestep is correct

### **Issue: PPO not learning**
- **Symptoms**: Reward not increasing, drone crashes immediately
- **Solutions**:
  - Simplify reward (remove some terms)
  - Increase episode length
  - Check for reward scaling issues (too large/small)
  - Reduce domain randomization initially
  - Start from easier task (hover before waypoints)

### **Issue: Hybrid worse than PID**
- **Symptoms**: Hybrid controller less stable
- **Solutions**:
  - Residual action space too large (reduce limits)
  - RL fighting PID (adjust reward to encourage cooperation)
  - Insufficient training (train longer)
  - Check observation normalization

### **Issue: Sim-to-real gap too large**
- **Symptoms**: Works in sim, fails on real drone
- **Solutions**:
  - Increase domain randomization intensity
  - Add sensor noise to simulation
  - Model actuator delays
  - Collect real data and fine-tune
  - Use conservative policies (lower gains)

### **Issue: Real drone crashes**
- **Symptoms**: Unpredictable behavior, instability
- **Solutions**:
  - Always test PID first (known baseline)
  - Start with very conservative gains
  - Test in simulation with real drone's exact parameters
  - Use RC override at first sign of trouble
  - Check battery voltage (low battery = poor performance)

---

## 📚 Key Resources

### **Documentation**
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

### **Papers to Read**
1. "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics" (2017)
2. "Learning to Fly by Crashing" (2017)
3. "Residual Policy Learning" (2018)
4. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (2017)

### **Video Tutorials**
- PID tuning for drones (YouTube)
- PPO algorithm explained (YouTube)
- Gym custom environments (YouTube)

---

##  Success Checklist

- [ ] Environment installed and tested (Day 1)
- [ ] PID controller working in sim (Day 5)
- [ ] Domain randomization tested (Day 7)
- [ ] PPO training launched (Day 8)
- [ ] PPO model converged (Day 11)
- [ ] Hybrid controller implemented (Day 12)
- [ ] Hybrid training complete (Day 14)
- [ ] All OOD tests run (Day 17)
- [ ] Results analyzed and plotted (Day 19)
- [ ] Real drone tested safely (Day 21)
- [ ] Report drafted (Day 21)

---

##  Expected Findings

Based on literature, you should observe:

1. **PID Performance**
   - Excellent on nominal conditions
   - Degrades significantly (+50-100% error) on OOD
   - Fast response but no adaptation

2. **Pure PPO Performance**
   - Better OOD generalization if DR is strong
   - May be unstable or unpredictable
   - Can be overly aggressive

3. **Hybrid Performance**
   - Best overall: PID-level stability + RL adaptability
   - Graceful degradation under OOD
   - Smooth control with intelligent corrections
   - **~20-40% better tracking error than PID on OOD scenarios**

4. **Key Insight**
   - Hybrid combines model-based (PID) and model-free (RL) strengths
   - RL residual learns to correct systematic PID errors
   - Domain randomization is crucial for real-world transfer

---

##  Tips for Success

1. **Start simple**: Get basic hover working before complex trajectories
2. **Visualize everything**: Plot trajectories, rewards, errors constantly
3. **Save checkpoints**: Don't lose training progress
4. **Test incrementally**: Don't wait until the end to test integration
5. **Communicate daily**: 15-min stand-ups to coordinate
6. **Document as you go**: Write methods while implementing
7. **Safety first**: Never rush real drone tests
8. **Plan for delays**: If training is slow, reduce total timesteps

---

##  Division of Labor Summary

| Member | Primary Role | Time Allocation |
|--------|-------------|-----------------|
| **Member 1** | Infrastructure & Hardware | 40% setup, 30% support, 30% real testing |
| **Member 2** | PID & Classical Control | 50% PID, 30% DR, 20% analysis |
| **Member 3** | RL Training | 70% training, 30% support |
| **Member 4** | Testing & Analysis | 60% testing, 40% visualization |

**Everyone** contributes to:
- Real drone testing (safety requires full team)
- Report writing (divided by sections)
- Debugging and troubleshooting
- Final presentation preparation

---
