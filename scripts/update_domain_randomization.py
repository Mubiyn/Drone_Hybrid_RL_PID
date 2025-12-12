#!/usr/bin/env python3
"""
Update Domain Randomization Parameters from Real Tello Data

After collecting real flight data, use this to extract actual Tello dynamics
and update the simulation's domain randomization to match reality.

Usage:
    python scripts/update_domain_randomization.py data/tello_flights/*.pkl
"""

import pickle
import numpy as np
from pathlib import Path
import sys


def analyze_flight_data(filepath):
    """Extract dynamics parameters from flight data"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    states = data['states']
    actions = data['actions']
    timestamps = np.array(data['timestamps'])
    
    velocities = states[:, 6:9]
    
    # Max velocity
    speeds = np.linalg.norm(velocities, axis=1)
    max_speed = speeds.max()
    
    # Acceleration
    dt = np.diff(timestamps)
    dv = np.diff(velocities, axis=0)
    accelerations = dv / dt[:, np.newaxis]
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)
    max_accel = np.percentile(accel_magnitudes, 95)  # Use 95th percentile to avoid spikes
    
    return {
        'max_speed': max_speed,
        'max_accel': max_accel,
        'duration': timestamps[-1] - timestamps[0],
        'samples': len(states)
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/update_domain_randomization.py <flight_data_files>")
        sys.exit(1)
    
    files = sys.argv[1:]
    
    print("="*80)
    print("DOMAIN RANDOMIZATION PARAMETER EXTRACTION")
    print("="*80)
    
    # Analyze all flights
    results = []
    for filepath in files:
        try:
            result = analyze_flight_data(filepath)
            result['file'] = Path(filepath).name
            results.append(result)
            print(f"✓ {result['file']}: max_vel={result['max_speed']:.2f} m/s, max_accel={result['max_accel']:.2f} m/s²")
        except Exception as e:
            print(f"✗ {filepath}: {e}")
    
    if not results:
        print("\n❌ No valid flight data!")
        return
    
    # Aggregate statistics
    max_speeds = [r['max_speed'] for r in results]
    max_accels = [r['max_accel'] for r in results]
    
    print("\n" + "="*80)
    print("MEASURED TELLO PARAMETERS")
    print("="*80)
    
    print(f"\nVelocity limits:")
    print(f"  Mean: {np.mean(max_speeds):.3f} m/s")
    print(f"  Max:  {np.max(max_speeds):.3f} m/s")
    print(f"  Min:  {np.min(max_speeds):.3f} m/s")
    
    print(f"\nAcceleration limits:")
    print(f"  Mean: {np.mean(max_accels):.3f} m/s²")
    print(f"  Max:  {np.max(max_accels):.3f} m/s²")
    
    print(f"\nMass (from DJI specs):")
    print(f"  Tello: 80g")
    print(f"  CF2X (sim): 27g")
    print(f"  Ratio: {80/27:.2f}x heavier")
    
    # Compute domain randomization parameters
    print("\n" + "="*80)
    print("RECOMMENDED DOMAIN RANDOMIZATION UPDATES")
    print("="*80)
    
    # Current simulation (CF2X)
    cf2x_mass = 0.027  # kg
    cf2x_inertia = [1.4e-5, 1.4e-5, 2.17e-5]
    cf2x_max_vel = 2.0  # m/s (approximate from sim)
    
    # Measured Tello
    tello_mass = 0.080  # kg
    tello_max_vel = np.mean(max_speeds)
    tello_max_accel = np.mean(max_accels)
    
    # Mass ratio
    mass_ratio = tello_mass / cf2x_mass
    
    # Inertia scales roughly with mass for similar geometry
    tello_inertia_est = [i * mass_ratio for i in cf2x_inertia]
    
    print(f"\nOLD (CF2X-centered) domain randomization:")
    print(f"  mass: {cf2x_mass:.4f} kg ± 30% → [{cf2x_mass*0.7:.4f}, {cf2x_mass*1.3:.4f}] kg")
    print(f"  inertia: {cf2x_inertia} ± 30%")
    print(f"  max_vel: ~{cf2x_max_vel:.1f} m/s")
    
    print(f"\nNEW (Tello-centered) domain randomization:")
    print(f"  mass: {tello_mass:.4f} kg ± 20% → [{tello_mass*0.8:.4f}, {tello_mass*1.2:.4f}] kg")
    print(f"  inertia: {tello_inertia_est} ± 20%")
    print(f"  max_vel: ~{tello_max_vel:.1f} m/s (enforce in action clipping)")
    print(f"  max_accel: ~{tello_max_accel:.1f} m/s² (implicit from mass/thrust)")
    
    # Generate code update
    print("\n" + "="*80)
    print("CODE UPDATE FOR src/envs/HybridAviary.py")
    print("="*80)
    
    print(f"""
Update _randomize_dynamics() method:

def __init__(self, ...):
    ...
    # Domain Randomization - UPDATED FOR TELLO
    self.domain_randomization = domain_randomization
    self.original_mass = {tello_mass:.4f}  # Tello mass (was 0.027 for CF2X)
    self.original_inertia = {tello_inertia_est}  # Estimated Tello inertia
    
    # Velocity limits (for action clipping)
    self.max_velocity = {tello_max_vel:.2f}  # Measured from real Tello

def _randomize_dynamics(self):
    # Mass ± 20% (Tighter range for real Tello)
    mass_scale = np.random.uniform(0.8, 1.2)
    new_mass = self.original_mass * mass_scale
    
    # Inertia ± 20%
    inertia_scale = np.random.uniform(0.8, 1.2)
    new_inertia = [i * inertia_scale for i in self.original_inertia]
    
    for i in range(self.NUM_DRONES):
        p.changeDynamics(self.DRONE_IDS[i], -1, 
                        mass=new_mass, 
                        localInertiaDiagonal=new_inertia, 
                        physicsClientId=self.CLIENT)
    
    # Optional: Add actuator delay (to model ~10-30ms hardware lag)
    # self.action_delay_buffer = []  # Store delayed actions

def _apply_action_clipping(self, velocity_cmd):
    '''Clip to measured Tello limits'''
    velocity_cmd = np.clip(velocity_cmd, -self.max_velocity, self.max_velocity)
    return velocity_cmd
""")
    
    print("\n" + "="*80)
    print("IMPACT ON TRAINING")
    print("="*80)
    
    print(f"""
WHY this matters:

1. MASS DIFFERENCE:
   - Old randomization: 27g ± 30% → [18.9g, 35.1g]
   - New randomization: 80g ± 20% → [64g, 96g]
   - Your Tello (80g) was OUTSIDE old randomization range!
   
2. VELOCITY LIMITS:
   - Old simulation: No limit (~2+ m/s)
   - Measured Tello: {tello_max_vel:.2f} m/s max
   - This affects optimal policy (can't plan for speeds > {tello_max_vel:.2f} m/s)

3. INERTIA (affects angular dynamics):
   - Old: {cf2x_inertia}
   - New: {tello_inertia_est}
   - 3x difference in rotational response!

RECOMMENDED NEXT STEPS:

1. Update HybridAviary.py with new parameters above
2. Re-train hybrid model with Tello-centered domain randomization
3. Evaluate: Old hybrid vs New hybrid vs PID on real Tello
4. Expected: New hybrid should transfer better (smaller reality gap)

ALTERNATIVE (if you don't want to re-train):

Use the autonomous PID data collection + offline RL fine-tuning
to adapt the existing hybrid model to real Tello dynamics.
This is faster than full re-training.
""")


if __name__ == '__main__':
    main()
