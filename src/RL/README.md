# 🚁 Drone RL + PID Controller  + Hybrid RL 
### Reinforcement Learning for Precise Trajectory Tracking Using PyBullet + PPO 

This repository contains a full **Reinforcement Learning ** control system for trajectory-tracking quadcopters.  
The environment is built on **gym-pybullet-drones**, with a custom modified RL environment that teaches a Crazyflie-style drone to fly:

- ✔ Circles  
- ✔ Figure-8 paths  
- ✔ Four-point patterns  
- ✔ “Go-To” single-point moves  


---
```bash
train  python .\scripts\trainer.py # make sure you set a task to train [circle, four_points, figure8,goto]
test src/testing/tester.py 
```


### **Single-Task & Multi-Task Training**
The environment supports:
- **Single-task RL** (train only `circle`, only `four_points`, etc.)
- **Sequential training** (train circle → load model → train figure8 → …)

Each task has its own set of waypoints and reward shaping.

---

### **Waypoint-Based Motion**
Each trajectory is expressed as a list of waypoints:

The agent receives:
- Current position  
- Velocity  
- Orientation  
- Angular velocity  
- Relative vector to current waypoint  

---


### **4. Live Training Plot**
A live matplotlib window shows:
- Rolling mean episode reward  
- Training stability  
- Improvements over time  

---